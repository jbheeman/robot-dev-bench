"""
Test: Stereo Calibration Pipeline

Generates synthetic stereo checkerboard videos with known ground-truth
camera parameters, runs the calibration pipeline, and verifies the
recovered R and T against the ground truth.
"""

import json
import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_mock_calibration import generate as generate_mock
from src.processing.calibration import calibrate_stereo, CalibrationResult


@pytest.fixture(scope="module")
def calibration_data(tmp_path_factory):
    """Generate mock videos and run calibration once for the whole module."""
    out_dir = str(tmp_path_factory.mktemp("mock_cal"))
    gt = generate_mock(out_dir, num_frames=60, fps=15, seed=42)

    left_path = os.path.join(out_dir, "calibration_left.mp4")
    right_path = os.path.join(out_dir, "calibration_right.mp4")

    result = calibrate_stereo(
        left_path, right_path,
        board_size=(gt["board_size"][0], gt["board_size"][1]),
        square_size=gt["square_size"],
    )

    return result, gt


class TestCalibrationPipeline:
    """Validates the calibration pipeline against synthetic ground truth."""

    def test_enough_pairs_detected(self, calibration_data):
        """At least 5 valid stereo pairs must be found."""
        result, _ = calibration_data
        assert result.num_valid_pairs >= 5, (
            f"Only {result.num_valid_pairs} pairs found — expected ≥5"
        )

    def test_stereo_rms_acceptable(self, calibration_data):
        """Stereo RMS reprojection error should be < 1.0 px for synthetic data."""
        result, _ = calibration_data
        assert result.rms_stereo < 1.0, (
            f"Stereo RMS {result.rms_stereo:.4f} exceeds threshold of 1.0 px"
        )

    def test_translation_direction(self, calibration_data):
        """
        The recovered T should point predominantly along X (horizontal baseline).
        Ground truth: T = [0.12, 0.0, 0.0].
        """
        result, gt = calibration_data
        T = result.T.flatten()
        gt_T = np.array(gt["T"]).flatten()

        # The dominant component should be X
        assert abs(T[0]) > abs(T[1]), "X should dominate over Y"
        assert abs(T[0]) > abs(T[2]), "X should dominate over Z"

    def test_translation_magnitude(self, calibration_data):
        """
        The baseline distance should be close to the ground truth (0.12 m).
        Allow 20% tolerance for synthetic rendering artefacts.
        """
        result, gt = calibration_data
        recovered_baseline = np.linalg.norm(result.T)
        gt_baseline = np.linalg.norm(np.array(gt["T"]))

        ratio = recovered_baseline / gt_baseline
        assert 0.8 < ratio < 1.2, (
            f"Baseline ratio {ratio:.3f} outside [0.8, 1.2] — "
            f"recovered={recovered_baseline:.4f}, gt={gt_baseline:.4f}"
        )

    def test_rotation_near_identity(self, calibration_data):
        """
        Ground truth R is identity (parallel cameras).
        The recovered R should be close to identity.
        """
        result, _ = calibration_data
        R_diff = result.R - np.eye(3)
        max_diff = np.max(np.abs(R_diff))
        assert max_diff < 0.15, (
            f"Max deviation from identity R is {max_diff:.4f} — expected < 0.15"
        )

    def test_intrinsic_focal_length(self, calibration_data):
        """
        Ground truth focal length is 800 px.
        Recovered fx/fy should be within 10% of that.
        """
        result, gt = calibration_data
        gt_fx = gt["K_left"][0][0]

        for name, K in [("left", result.K_left), ("right", result.K_right)]:
            fx = K[0, 0]
            fy = K[1, 1]
            assert abs(fx - gt_fx) / gt_fx < 0.10, (
                f"{name} fx={fx:.1f} deviates >10% from gt={gt_fx:.1f}"
            )
            assert abs(fy - gt_fx) / gt_fx < 0.10, (
                f"{name} fy={fy:.1f} deviates >10% from gt={gt_fx:.1f}"
            )

    def test_save_and_load_roundtrip(self, calibration_data, tmp_path):
        """CalibrationResult should survive a JSON save/load cycle."""
        result, _ = calibration_data
        path = str(tmp_path / "cal.json")
        result.save(path)

        loaded = CalibrationResult.load(path)
        np.testing.assert_allclose(loaded.K_left, result.K_left, atol=1e-6)
        np.testing.assert_allclose(loaded.R, result.R, atol=1e-6)
        np.testing.assert_allclose(loaded.T, result.T, atol=1e-6)
        assert loaded.num_valid_pairs == result.num_valid_pairs

    def test_rectification_matrices_exist(self, calibration_data):
        """Stereo rectification matrices should be populated."""
        result, _ = calibration_data
        assert result.R1 is not None, "R1 not computed"
        assert result.R2 is not None, "R2 not computed"
        assert result.P1 is not None, "P1 not computed"
        assert result.P2 is not None, "P2 not computed"
        assert result.Q is not None, "Q not computed"
