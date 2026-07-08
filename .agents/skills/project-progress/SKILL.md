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

**Status: Phase 3 Setup**

1. Refactor the backend API and frontend UI to dynamically hide/show specific metrics based on the dropdown selection.
2. Implement backend logic for `Jumping` metrics (Flight Time, Peak Z-Axis Accel, Landing Jerk) and update `rules.py`.
3. Implement backend logic for `Transitions` (CoM Oscillation, Transition Time).

## Known Bugs / Blockers

*   *(Resolved)* **Schema Mismatch:** Fixed in Phase 2, Part 1. The dashboard now properly reads HuggingFace `.parquet` schemas.
*   *(Resolved)* **Multi-Episode Playback Compression:** Multi-episode `.parquet` datasets (like `file-000.parquet` with 190+ episodes) were previously being squashed into a single playback timeline, causing the robot to rapidly jump across completely different episodes in high speed. `app.py` has been updated to automatically filter to just `episode_index == 0`, ensuring clean, coherent metrics and playback.
