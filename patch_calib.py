import os

with open("src/processing/calibration.py", "r") as f:
    content = f.read()

# Find the start of the Corner extraction section
split_marker = "# ── Corner extraction ───────────────────────────────────────────────────────"
parts = content.split(split_marker)

if len(parts) != 2:
    print("Could not find the split marker")
    exit(1)

new_content = parts[0] + split_marker + """

def _get_charuco_board(board_size, square_size, marker_size):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(board_size, square_size, marker_size, dictionary)
    return board, dictionary

def extract_corners_from_video_pair(
    left_path: str,
    right_path: str,
    board_size: Tuple[int, int] = (10, 7),
    square_size: float = 0.025,
    marker_size: float = 0.015,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)

    if not cap_l.isOpened():
        raise FileNotFoundError(f"Cannot open left video: {left_path}")
    if not cap_r.isOpened():
        raise FileNotFoundError(f"Cannot open right video: {right_path}")

    left_corners: List[np.ndarray] = []
    left_ids: List[np.ndarray] = []
    right_corners: List[np.ndarray] = []
    right_ids: List[np.ndarray] = []
    image_size = None
    frame_idx = 0

    board, dictionary = _get_charuco_board(board_size, square_size, marker_size)
    detectorParams = cv2.aruco.DetectorParameters()
    charucoDetector = cv2.aruco.CharucoDetector(board)

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        if image_size is None:
            image_size = (frame_l.shape[1], frame_l.shape[0])

        grey_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        grey_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        charucoCorners_l, charucoIds_l, _, _ = charucoDetector.detectBoard(grey_l)
        charucoCorners_r, charucoIds_r, _, _ = charucoDetector.detectBoard(grey_r)

        # Ensure enough corners were found in both
        if charucoCorners_l is not None and charucoCorners_r is not None and len(charucoCorners_l) > 6 and len(charucoCorners_r) > 6:
            # For stereo calibration to work easily, we need the SAME corners detected in both cameras
            # Find the intersection of detected IDs
            common_ids = np.intersect1d(charucoIds_l, charucoIds_r)
            if len(common_ids) > 6:
                # Filter to only keep common corners
                filt_corners_l = []
                filt_ids_l = []
                for i, id_val in enumerate(charucoIds_l):
                    if id_val[0] in common_ids:
                        filt_corners_l.append(charucoCorners_l[i])
                        filt_ids_l.append(id_val)
                
                filt_corners_r = []
                filt_ids_r = []
                for i, id_val in enumerate(charucoIds_r):
                    if id_val[0] in common_ids:
                        filt_corners_r.append(charucoCorners_r[i])
                        filt_ids_r.append(id_val)
                
                # Sort them so they align
                sorted_l = sorted(zip(filt_ids_l, filt_corners_l), key=lambda x: x[0][0])
                sorted_r = sorted(zip(filt_ids_r, filt_corners_r), key=lambda x: x[0][0])

                left_corners.append(np.array([x[1] for x in sorted_l]))
                left_ids.append(np.array([x[0] for x in sorted_l]))
                right_corners.append(np.array([x[1] for x in sorted_r]))
                right_ids.append(np.array([x[0] for x in sorted_r]))

        frame_idx += 1

    cap_l.release()
    cap_r.release()

    logger.info(
        "Found %d valid stereo pairs out of %d frames",
        len(left_corners), frame_idx,
    )
    return left_corners, left_ids, right_corners, right_ids, image_size or (1280, 720)


# ── Calibration ─────────────────────────────────────────────────────────────

def calibrate_stereo(
    left_video_path: str,
    right_video_path: str,
    board_size: Tuple[int, int] = (10, 7),
    square_size: float = 0.025,
    marker_size: float = 0.015,
) -> CalibrationResult:
    result = CalibrationResult(
        board_size=board_size,
        square_size=square_size,
        marker_size=marker_size,
    )

    # Step 1: extract corners
    left_corners, left_ids, right_corners, right_ids, img_size = extract_corners_from_video_pair(
        left_video_path, right_video_path, board_size, square_size, marker_size
    )
    result.image_size = img_size
    result.num_valid_pairs = len(left_corners)

    if len(left_corners) < 5:
        raise ValueError(
            f"Only {len(left_corners)} valid stereo pairs found — need at least 5 "
            "for a reliable calibration. Ensure the ChArUco board is visible in "
            "both cameras simultaneously."
        )

    board, dictionary = _get_charuco_board(board_size, square_size, marker_size)

    # Step 2: individual camera calibration
    logger.info("Calibrating left camera …")
    result.rms_left, result.K_left, result.dist_left, _, _ = cv2.aruco.calibrateCameraCharuco(
        left_corners, left_ids, board, img_size, None, None
    )
    logger.info("Left camera RMS: %.4f", result.rms_left)

    logger.info("Calibrating right camera …")
    result.rms_right, result.K_right, result.dist_right, _, _ = cv2.aruco.calibrateCameraCharuco(
        right_corners, right_ids, board, img_size, None, None
    )
    logger.info("Right camera RMS: %.4f", result.rms_right)

    # Step 3: stereo calibration
    logger.info("Running stereo calibration …")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # For stereoCalibrate with ChArUco, we need to map the IDs back to object points
    object_points = []
    left_corners_stereo = []
    right_corners_stereo = []

    board_obj_pts = board.getChessboardCorners()
    for i in range(len(left_corners)):
        obj_pts_frame = np.array([board_obj_pts[id[0]] for id in left_ids[i]])
        object_points.append(obj_pts_frame)
        left_corners_stereo.append(left_corners[i])
        right_corners_stereo.append(right_corners[i])

    result.rms_stereo, _, _, _, _, result.R, result.T, _, _ = cv2.stereoCalibrate(
        object_points,
        left_corners_stereo,
        right_corners_stereo,
        result.K_left,
        result.dist_left,
        result.K_right,
        result.dist_right,
        img_size,
        criteria=criteria,
        flags=flags,
    )
    logger.info("Stereo RMS: %.4f", result.rms_stereo)

    # Step 4: stereo rectification
    logger.info("Computing stereo rectification …")
    result.R1, result.R2, result.P1, result.P2, result.Q, _, _ = cv2.stereoRectify(
        result.K_left, result.dist_left,
        result.K_right, result.dist_right,
        img_size, result.R, result.T,
        alpha=0,
    )

    result.mean_reprojection_error = result.rms_stereo

    logger.info("Calibration complete. %d pairs, stereo RMS=%.4f",
                result.num_valid_pairs, result.rms_stereo)

    return result
"""

with open("src/processing/calibration.py", "w") as f:
    f.write(new_content)
