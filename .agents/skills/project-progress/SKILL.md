---
name: project-progress
description: Tracks the current implementation progress of the Unitree G1-Edu Benchmarking pipeline project.
---

# Project Implementation Progress

**ATTENTION AI AGENTS**: 
1. Use this document to understand the current state of the project and align on the immediate next steps. Do not duplicate completed work.
2. **CRITICAL RULE**: You MUST update this document to reflect any newly completed work or changed priorities **every time before pushing code or finalizing your tasks**. Keep the status accurate.
Based on the `project-context` skill, here is the current global status of the implementation:

## Phase 1 (COMPLETE)

*   **Part 1: Project Objective:** Understood and ongoing.
*   **Part 2: Telemetry Ingestion & Storage:** **[DONE]** `live_subscriber.py`, `exporter.py`, and `data_models.py` are implemented for capturing hardware data via CycloneDDS and exporting it to Pandas/Parquet.
*   **Part 3: Data Processing & Synchronization (Post-Run):** **[DONE]** Zero-phase digital filtering (`filter.py`) and timestamp alignment/merging (`synchronizer.py`) are implemented and validated.
*   **Part 4: Feature Engineering (Key Metrics):** **[DONE]** Metric extractors for control precision (RMSE), Cost of Transport (CoT), control latency, hardware stress, and stability (IMU and CoM variance) are implemented in `metrics.py` and `stability.py` and covered by unit tests in `test_features.py`.
*   **Part 5: Model Training Data & Baselines:** **[DONE]** Defined static baseline bounds for optimal RL/humanoid target limits and updated Class 2 and 3 with empirical data from the Humanoid Everyday Dataset (`baselines.py`).
*   **Part 6: Classification Architecture:** **[DONE]** Implemented the supervised rule-based scorer in `rules.py` and integrated it.
*   **Part 7: Web Infrastructure & Integration:** **[DONE]** Built a premium modern local dashboard (`src/web`) with a FastAPI backend. The `/api/upload` endpoint accepts `.parquet` telemetry logs, parses them with `pd.read_parquet`, runs the full feature extraction pipeline, and classifies the result with the real rule-based engine.
*   **Part 8: End-to-End Testing:** **[DONE]** A synthetic `.parquet` generator (`scripts/generate_test_parquet.py`) and a full integration test suite (`tests/test_integration.py`) were implemented. All 10 integration tests pass.

> [!WARNING]
> **Phase 1 Limitation Discovered:** The Phase 1 metrics (RMSE, CoT, torque stress, IMU variance) require raw hardware telemetry (torques, voltages, IMU data) that is only available from our own CycloneDDS subscriber. Real-world public datasets (HuggingFace / LeRobot) only contain joint positions and actions. This caused all metrics to return 0.0 and a false "perfect" score when uploading real files. Phase 2 addresses this fundamental limitation.

## Phase 2 (IN PROGRESS — Biomechanical Scoring & 3D Playback)

Phase 2 pivots to clinically-grounded biomechanical metrics inspired by pediatric gait analysis. These metrics evaluate intrinsic movement quality using only joint angle trajectories — exactly what public datasets provide.

*   **Part 1: Data Ingestion Overhaul:** **[DONE]** Built a `SchemaMapper` to auto-detect and normalise column names from HuggingFace/LeRobot Parquet formats into our internal schema.
*   **Part 2: Biomechanical Metrics Engine:** **[DONE]** Implemented new metrics: Smoothness (LDLJ), Symmetry Index, Periodicity, Range of Motion, and Spectral Arc Length (SPARC). Updated baselines and classifier to handle new bounds and task-specific weight profiles. Updated frontend UI.
*   **Part 3: 3D Movement Playback Viewer:** **[DONE]** Integrated Three.js and URDF loader to display G1 3D model. Back-end downsamples kinematics data and streams it to the browser. Added a "Testing Only" tab to view movement without running classification scoring.
*   **Part 4: Per-Task Dashboard Tabs:** **[DONE]** Added task categorisation (Walking, Reaching, Manipulation) via user dropdown on upload, with per-task weight profiles for the classifier.

## Phase 3 (IN PROGRESS — Task-Specific Metrics & Architectural Refactor)

Building off the recent architectural shift, the dashboard must dynamically display different metrics for different tasks, hiding irrelevant ones (e.g. Symmetry for Manipulation), and introducing new key evaluation actions like Jumping and Transitions.

## Immediate Next Steps

**Status: Phase 3 implementation is 100% complete!**

1. Refactored the backend API and frontend UI to dynamically hide/show specific metrics based on the dropdown selection.
2. Implemented backend logic for `Jumping` metrics (Flight Time, Peak Z-Axis Accel, Landing Jerk) and updated `rules.py`.
3. Implemented backend logic for `Transitions` (CoM Oscillation, Transition Time).
4. Polished UI color palette to match the dark slate and cyan/emerald theme requested by the user.

## Phase 4 (PLANNED — Advanced Dashboard Features)

Based on recent alignment, the following features are planned for implementation:
1. **[DONE]** **Interactive Time-Series Charts:** Added Chart.js to plot raw telemetry synced with 3D playback.
2. **[DONE]** **Anomaly Markers:** Overlaying red timeline markers at critical events (e.g., peak acceleration, maximum jerk).
3. **[DONE]** **Side-by-Side Comparison:** Allowing dual `.parquet` uploads for A/B testing of policies.
4. **[DONE]** **PDF/CSV Export:** Generating downloadable reports for standard Hugging Face model cards using `html2canvas` and `jspdf`.

## Immediate Next Steps

**Status: Phase 4 is fully complete! All planned dashboard features have been implemented, including the recent Log Library addition.**

*   *(Resolved)* **Schema Mismatch:** Fixed in Phase 2, Part 1. The dashboard now properly reads HuggingFace `.parquet` schemas.
*   *(Resolved)* **Multi-Episode Playback Compression:** Multi-episode `.parquet` datasets (like `file-000.parquet` with 190+ episodes) were previously being squashed into a single playback timeline, causing the robot to rapidly jump across completely different episodes in high speed. `app.py` has been updated to automatically filter to just `episode_index == 0`, ensuring clean, coherent metrics and playback.
*   *(Resolved)* **UI Enhancements:** The PDF export was refined to hide active tooltips/checkboxes and seamlessly display Chart.js titles directly inside the graphs for clarity. The file upload drop zones were also styled to default to a clear red border and turn green upon successful upload to provide better user feedback.
*   *(Resolved)* **Log Library (File Box):** Added a frontend-only session history vault that stores all uploaded logs during a session. Users can directly upload multiple logs from the library modal. After selecting logs, users can view them in a leaderboard and dynamically change the evaluated task via a dropdown (which uses a new `/api/reclassify` endpoint) to instantly recalculate and re-rank scores on the fly without re-uploading.
*   *(Resolved)* **UI Adjustments:** Made the baseline policy upload box smaller than the primary policy box to emphasize that it is optional.

## Phase 5 (PLANNED — Pivot to Two-Camera Black-Box Benchmarking)

We are fundamentally pivoting the architecture from ingesting internal telemetry `.parquet` logs to a passive, external, Visual "Black-Box" evaluation system. This enables evaluating any general-purpose robot without internal access, mapping performance to pediatric developmental milestones.

**Key Decisions:**
*   **Remove Legacy Data:** Completely remove the existing `.parquet` telemetry pipeline.
*   **Computer Vision:** Two-camera stereo setup. OpenCV for calibration, **MMPose with a ViTPose++ backbone** for 2D-to-3D triangulation.
*   *(Note: Acoustic inference and Conversational logic/STT evaluation have been skipped/deferred).*

To ensure smooth execution, this phase is broken down into sequential steps:

*   **Step 5.1: Repository Cleanup & Foundation:** Remove all `.parquet` logic and build basic video upload endpoints.
*   **Step 5.2: Stereo Calibration & Mocking:** Build the synthetic checkerboard generator and OpenCV calibration pipeline.
*   **Step 5.3: 3D Pose Triangulation:** Integrate MMPose/ViTPose++ and triangulate 2D joints into 3D world coordinates.

## Immediate Next Steps

**Status: The entire Phase 5 architectural pivot is complete.**

*   *(Resolved)* **Step 5.1 (Cleanup):** Stripped out the existing `.parquet` logic from `src/ingestion`, `src/processing`, `src/features`, and `src/web`. Set up basic dual-camera MP4 upload endpoints.
*   *(Resolved)* **Step 5.2 (Calibration):** Implemented the synthetic checkerboard video mock generator (`scripts/generate_mock_calibration.py`), the OpenCV stereo calibration module (`src/processing/calibration.py`), `/api/calibrate` and `/api/calibration_status` API endpoints, a calibration results visualisation UI (`calibration.html` with Three.js 3D camera placement), and 8 passing unit tests validating recovered R/T/K against ground truth.
*   *(Resolved)* **Step 5.3 (3D Pose Triangulation):** Integrated MMPose with a ViTPose++ backbone for 2D pose estimation. Implemented 3D triangulation (`src/processing/triangulation.py`) using DLT. Refactored biomechanics to operate on 3D pose arrays. Wired the full pipeline into the `/api/upload_av` backend endpoint and restored the original `RuleBasedClassifier` metric scoring.
*   *(Resolved)* **Step 6 (Verification & Finalization):** Built the final visualization UI on the dashboard. Modified the frontend to natively parse the 3D keypoints from the API and render a COCO skeleton in 3D using Three.js. Implemented a glassmorphism results dashboard to beautifully display the extracted biomechanical metrics and dynamic classification tier badge.
*   *(Resolved)* **Step 7 (ChArUco Upgrade):** Upgraded the fragile standard checkerboard pipeline to use ChArUco boards. Re-wrote `calibration.py` to use `cv2.aruco.CharucoDetector`, added a `generate_charuco.py` script for users to print robust calibration boards, and updated the UI forms and documentation to accept the new `marker_size` parameter.
*   *(Resolved)* **Step 8 (Legacy Log Pipeline Restoration):** Restored the old Parquet telemetry ingestion pipeline, 1D biomechanics extractors, and Log Library UI as a secondary 'Legacy Log Pipeline' tab, allowing users to evaluate either AV or Parquet logs simultaneously.
*   *(Resolved)* **Step 9 (1-Camera Monocular Pivot):** Completely removed the 2-camera stereo calibration and triangulation pipeline. The system now exclusively uses a single-camera monocular pipeline, inferring depth by walking the kinematic tree against known G1 bone-lengths (`src/processing/monocular_depth.py`). The dashboard UI has been streamlined to accept a single camera feed.
*   *(Resolved)* **Step 10 (Background Job Framework & GPU Acceleration):** Upgraded the monocular pipeline's web endpoint to use FastAPI `BackgroundTasks` with an in-memory `JobStore` class. This provides non-blocking `/api/upload_av` and a new `/api/job_status` polling endpoint. The frontend UI was updated to include a dynamic progress bar while processing video streams. Additionally, the backend was refactored to auto-detect and use PyTorch GPU acceleration (`cuda`/`mps`) instead of strictly CPU, significantly reducing inference time.
*   *(Resolved)* **Step 11 (Bug Fixes & UI Polish):** Resolved an mmengine `DefaultScope` thread race condition by implementing a global Inference Lock and Singleton `PoseEstimator`. Added a "Reset View" button to the 3D visualizer.
*   *(Resolved)* **Step 12 (MotionBERT Integration):** Swapped the custom kinematic monocular depth tree for a state-of-the-art MotionBERT pose lifter. Fixed the COCO-to-H36M format mismatch bug that was garbling the axes, and applied `norm_pose_2d=True` to prevent the output skeleton from improperly scaling up/down in size as the subject approached the camera.
*   *(Resolved)* **Step 13 (3D Playback Jitter Fix):** Actually implemented and applied the temporal zero-phase Butterworth filter (`TelemetryFilter`) to the 3D output of MotionBERT. This resolves a significant bug where joints were vibrating and jittering violently during 3D playback.
*   *(Resolved)* **Step 14 (Generalized Humanoid Fine-Tuning Pipeline):** Created a full data collection and fine-tuning pipeline to solve the domain gap between humans and humanoid robots. Added `scripts/scrape_humanoids.py` to extract diverse frames from YouTube via `yt-dlp`, and `scripts/finetune_vitpose.py` to programmatically fine-tune `ViTPose-small` on custom COCO annotations. Updated `src/processing/pose_estimation.py` to automatically load `vitpose_humanoid.pth` if present.
*   *(Resolved)* **Step 14.1 (Finetuning Bug Fixes):** Successfully patched multiple MMPose installation issues related to `xtcocotools` dependencies in Python 3.13 by creating a `pycocotools` shim. Disabled strict `COCOeval` validation in `finetune_vitpose.py` to bypass evaluator argument mismatches and ensure smooth, uninterupted fine-tuning completion.
*   *(Resolved)* **Step 15 (Stereo Camera Pipeline & Video Recording):** Integrated the stereo camera triangulation pipeline from `rt-pose-atao` into the devbench. Created `src/processing/stereo_core.py` with pure-numpy stereo math (frame splitting, disparity-based depth triangulation, pelvis matching). Updated `pose_estimation.py` to accept a `stereo` flag + `StereoConfig` (baseline_mm, focal_length_px): when enabled, each side-by-side frame is split, 2D pose is extracted from both halves, the pelvis is triangulated for metric depth, and the MotionBERT 3D output is scaled to real-world coordinates. Updated `/api/upload_av` to accept `stereo`, `baseline`, and `focal_length` form fields, and added `.webm` support for browser-recorded video. The frontend now offers a tabbed "Upload Video" / "Record Video" interface — the Record tab uses WebRTC `getUserMedia` + `MediaRecorder` for in-browser video capture with a live preview, pulse indicator, and timer. A collapsible "Stereo Camera" settings panel exposes editable Baseline (mm) and Focal Length (px) fields. Results display a "Monocular" or "Stereo-Fused" pipeline badge on the 3D viewer.
*   *(Resolved)* **Step 16 (MotionAGFormer Migration):** Replaced the heavy `MotionBERT` lifter with a lightweight, causal `MotionAGFormer` implementation from `rt-pose-atao`. Vendored the `MotionAGFormer-S` weights and source code, completely removing the `mmpose` sliding-window sequence overhead. Refactored `pose_estimation.py` to process 3D poses causally, frame-by-frame, using a robust pure-numpy `SlidingWindow` and `JointHold` buffer to guard against absent detections, significantly increasing processing speed and averting empty-stack crashes.
*   *(Resolved)* **Step 16.1 (ViTPose Model Instantiation Bug Fix):** Fixed a `pos_embed` shape mismatch error that caused video processing to crash. When loading the fine-tuned custom humanoid checkpoint (which was trained on `ViTPose-small`), the `PoseEstimator` was erroneously attempting to instantiate a `ViTPose-huge` architecture, leading `timm` to miscalculate expected positional embedding dimensions (`192` vs `169`). Patched `pose_estimation.py` to correctly load the `ViTPose-small` config when using the custom humanoid checkpoint.
*   *(Resolved)* **Step 17 (UI Tweaks & 2D Playback Bugfix):** Added descriptive tooltip icons to all metric statlines in the dashboard, linked a new `scoring.html` page to explain how metric scoring weights are dynamically applied per task, and fixed a 2D playback overlay bug where a black screen occurred due to canvas coordinate mismatches with `object-fit: contain` scaling, along with a `loadedmetadata` browser race condition. Enabled dual 2D/3D playback side-by-side so MP4 uploads display both views simultaneously. Removed the duplicate "Walk Cycle Grade" UI badge, merging its biomechanical weights into the core Classification Score.
*   *(Resolved)* **Step 18 (Fall Detection & Overlay Sync):** Implemented a Fall Detection rule for the walking task that monitors the vertical clearance between the pelvis/hips and the lowest ankle joint. If the torso drops within 20cm of the ground, the system immediately returns a "Fall Detected" tier with a score of 0.0. Added a 2D confidence array filter and a frame-to-frame teleportation detector (which tracks 2D center-of-mass shifts) to both pipelines to prevent false "Fall Detected" positives when the subject walks out of frame and a static object (like a chair) is erroneously tracked with high confidence by ViTPose. Fixed a desynchronization bug in the 2D skeleton overlay where variable framerate videos caused the tracking to move ahead of the subject; the canvas now maps frames using a strict percentage of the video's exact duration. Added an `isFinite()` fallback for Chromium's `Infinity` duration bug on short `<video>` blobs to prevent frozen skeleton overlays. Additionally, injected `loop playsinline` attributes to the 2D video element so short playback tests match the infinite loop behavior of the 3D viewer rather than abruptly stopping and appearing broken. Finally, implemented a pure-JS virtual timeline fallback that manually drives the 2D skeleton playback and sizes the canvas if the browser completely fails to decode the `<video>` track (e.g., HEVC/H.265 videos on Chromium), allowing the AI analysis skeleton to still perfectly animate against a black background.
*   *(Resolved)* **Step 19 (Metrics Cleanup):** Cleaned up the web interface by removing all biomechanical metrics that were no longer actively weighted or utilized in the current simplified grading logic (e.g., LDLJ, SPARC smoothness, Symmetry Index, Periodicity, Range of Motion, and jump/transition metrics). Reduced the `app.py` backend payload and `script.js` mapping logic to exclusively compute and serve the core metrics: Foot Clearance, Stride Length, Walking Speed, and Torso Oscillation.
*   *(Resolved)* **Step 20 (2D Jumping Metrics):** Fixed an issue in the 2D video processing pipeline where jumping metrics (Flight Time, Peak Z-Accel, Landing Jerk) were hardcoded to zero. Implemented `compute_jumping_metrics_2d` to scale 2D pixel keypoints to metric units via shoulder-to-elbow reference length, allowing the 2D Fast Mode to correctly evaluate and score jumping sequences.
*   *(Resolved)* **Step 21 (Jumping Metrics Jitter & G-Force Conversion Fix):** Fixed a critical issue in the 2D jumping metrics where unfiltered high-frequency pixel jitter caused massive acceleration noise spikes (e.g. 47 m/s^2) and incorrect flight time duration detection. Applied the zero-phase `TelemetryFilter` to 2D vertical coordinates before taking derivatives. Additionally, patched both the 2D and 3D pipelines to convert the raw m/s^2 acceleration and m/s^3 jerk into Gs and G/s respectively (by dividing by 9.81) to match the UI's display expectations. Finally, updated the ideal/acceptable scoring bounds in `baselines.py` to correctly grade the new G-force values.
*   *(Resolved)* **Step 22 (Robust Flight Time Calculation):** Replaced the naive and fragile acceleration/jerk peak detection logic for calculating flight time. The system now uses robust 1D kinematic physics based on the vertical position's apex: it calculates takeoff as the moment of maximum upward velocity before the apex, and landing as the moment of maximum downward velocity after the apex. This completely resolves bugs where minor acceleration spikes during push-off caused the flight time to incorrectly register as only a single frame (0.10s).
*   *(Resolved)* **Step 23 (Metrics Classification Key Sync):** Updated the `CLASS_1` through `CLASS_3` thresholds and the `TASK_WEIGHTS` in `baselines.py` to use the explicitly unit-suffixed keys (`flight_time_s` and `peak_z_accel_g`). This fixes a bug where the scoring classifier silently ignored these two metrics because it was looking for the old unit-less keys, causing only the Landing Jerk to receive a score contribution.
*   *(Resolved)* **Step 24 (Classification Tier Rename):** Renamed the three performance classification tiers across the entire codebase. 'Superhuman/Industrial' is now 'Adult', 'Research' is now 'Adolescent', and 'Experimental' is now 'Infant'. Updated the grading UI colors, documentation, baselines, and test suites to reflect these new biological analogies.
*   *(Resolved)* **Step 25 (Manipulation Task):** Added a "Manipulation (2D Only)" task that isolates the upper body tracking by zeroing out lower body joints (hips, knees, ankles) to prevent tracking errors when occluded by a table. The UI automatically enforces 2D Overlay mode when this task is selected.

- [X] **Object Tracking**: Implemented basic HSV tracking for red and white blocks in `pose_estimation.py` and visual rendering in the 2D overlay.
- [X] **Kinematic Scoring**: Implemented `compute_manipulation_metrics_2d` to extract upper-body metrics like Wrist Jerk and block manipulation metrics (displacement, wrist distance) for evaluation of the manipulation task with physical blocks.
- [X] **Manual Bounding Box**: Added a UI flow that allows users to draw a bounding box around the robot to force the pose estimator to ignore background noise and heavily occluded setups (like tables).
- [X] **Localhost Deployment**: Updated `src/web/app.py` to host on `localhost:3000` instead of `0.0.0.0` as per user request.
- [X] **Keypoint Teleportation Filter**: Added a per-keypoint jump filter in `PoseEstimator` that drops keypoints (sets confidence to 0) if they jump more than 15% of the frame dimension between frames.
- [X] **Manipulation Dashboard**: Added the calculated manipulation metrics (displacement, wrist distance) to the frontend dashboard UI and fixed a bounding logic bug in `RuleBasedClassifier` that caused metric scores to incorrectly exceed 1.0.
- [X] **Tracking Bug Fixes**: Fixed a bug where `pose_estimation.py` was loading stock ViTPose-huge instead of the fine-tuned humanoid model, causing it to track humans in the background. Also fixed the `jump_threshold` bug that permanently locked out fast-moving joints (like wrists) from tracking recovery.
- [X] **Scoring Documentation**: Updated `scoring.html` to correctly document the new manipulation metrics and weights.
