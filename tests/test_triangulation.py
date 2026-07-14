import numpy as np
import pytest
from src.processing.calibration import CalibrationResult
from src.processing.triangulation import triangulate_points, undistort_points, triangulate_pose_sequence

def test_triangulate_points():
    # Construct a simple synthetic stereo setup
    # Both cameras point forward (Z direction)
    # Left camera is at origin (0, 0, 0)
    # Right camera is at (0.1, 0, 0) (baseline = 0.1m along X)
    
    fx, fy, cx, cy = 1000.0, 1000.0, 640.0, 360.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Projection matrix P = K @ [R | T]
    # Left camera
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    P1 = K @ np.hstack((R1, T1))
    
    # Right camera
    R2 = np.eye(3)
    T2 = np.array([[-0.1], [0.0], [0.0]]) # Right camera moved right by 0.1 means world moved left by -0.1 relative to it
    P2 = K @ np.hstack((R2, T2))
    
    # Create a 3D point (0.05, 0.2, 2.0)
    pt3d = np.array([[0.05, 0.2, 2.0]])
    
    # Project into Left camera
    pt3d_hom = np.array([[0.05], [0.2], [2.0], [1.0]])
    pt2d_l_hom = P1 @ pt3d_hom
    pt2d_l = (pt2d_l_hom[:2] / pt2d_l_hom[2]).T # (1, 2)
    
    # Project into Right camera
    pt2d_r_hom = P2 @ pt3d_hom
    pt2d_r = (pt2d_r_hom[:2] / pt2d_r_hom[2]).T # (1, 2)
    
    # Triangulate
    recovered_3d = triangulate_points(pt2d_l, pt2d_r, P1, P2)
    
    np.testing.assert_allclose(recovered_3d, pt3d, atol=1e-5)

def test_triangulate_pose_sequence():
    # Setup similar to above, but with the CalibrationResult object
    fx, fy, cx, cy = 1000.0, 1000.0, 640.0, 360.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((np.eye(3), np.array([[-0.1], [0.0], [0.0]])))
    
    cal = CalibrationResult()
    cal.K_left = K
    cal.dist_left = np.zeros(5)
    cal.K_right = K
    cal.dist_right = np.zeros(5)
    cal.R1 = np.eye(3)
    cal.R2 = np.eye(3)
    cal.P1 = P1
    cal.P2 = P2
    
    # Sequence of 3 frames, 2 joints
    T, J = 3, 2
    pts3d = np.array([
        [[0.0, 0.0, 1.0], [0.1, 0.1, 1.5]],
        [[0.05, 0.0, 1.0], [0.1, 0.2, 1.5]],
        [[0.1, 0.0, 1.0], [0.1, 0.3, 1.5]]
    ]) # (3, 2, 3)
    
    # Project points manually to create synthetic 2D observations
    pts2d_l = np.zeros((T, J, 2))
    pts2d_r = np.zeros((T, J, 2))
    
    for t in range(T):
        for j in range(J):
            pt = np.array([pts3d[t, j, 0], pts3d[t, j, 1], pts3d[t, j, 2], 1.0]).reshape(4, 1)
            
            p_l = P1 @ pt
            pts2d_l[t, j] = (p_l[:2] / p_l[2]).flatten()
            
            p_r = P2 @ pt
            pts2d_r[t, j] = (p_r[:2] / p_r[2]).flatten()
            
    conf_l = np.ones((T, J))
    conf_r = np.ones((T, J))
    
    # Make one joint in one frame low confidence
    conf_l[1, 1] = 0.1 
    
    recovered_poses, valid_mask = triangulate_pose_sequence(
        pts2d_l, pts2d_r, cal, confidence_left=conf_l, confidence_right=conf_r, min_confidence=0.3
    )
    
    assert recovered_poses.shape == (T, J, 3)
    assert valid_mask.shape == (T, J)
    
    # Frame 0: both valid
    assert valid_mask[0, 0] == True
    assert valid_mask[0, 1] == True
    np.testing.assert_allclose(recovered_poses[0], pts3d[0], atol=1e-5)
    
    # Frame 1: joint 1 invalid
    assert valid_mask[1, 0] == True
    assert valid_mask[1, 1] == False
    np.testing.assert_allclose(recovered_poses[1, 0], pts3d[1, 0], atol=1e-5)
    assert np.isnan(recovered_poses[1, 1]).all()
    
    # Frame 2: both valid
    assert valid_mask[2, 0] == True
    assert valid_mask[2, 1] == True
    np.testing.assert_allclose(recovered_poses[2], pts3d[2], atol=1e-5)
