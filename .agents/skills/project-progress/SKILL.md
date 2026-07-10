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

**Status: Phase 4 is fully complete! All planned dashboard features have been implemented.**

*   *(Resolved)* **Schema Mismatch:** Fixed in Phase 2, Part 1. The dashboard now properly reads HuggingFace `.parquet` schemas.
*   *(Resolved)* **Multi-Episode Playback Compression:** Multi-episode `.parquet` datasets (like `file-000.parquet` with 190+ episodes) were previously being squashed into a single playback timeline, causing the robot to rapidly jump across completely different episodes in high speed. `app.py` has been updated to automatically filter to just `episode_index == 0`, ensuring clean, coherent metrics and playback.

## Phase 5 (COMPLETE — Reverted Scoring to Hardware-Grounded Metrics)

The Phase 2 pivot to joint-position-only biomechanical scoring (smoothness/symmetry/periodicity/ROM/jumping/transitions) was masking the metrics the team actually wants graded. Now that our own CycloneDDS ingestion pipeline (`live_subscriber.py`, `mock_publisher.py`, `synchronizer.py`) reliably produces full hardware telemetry (torque, voltage/current, IMU), the classifier has been switched back to the four Phase-1 metrics that were already implemented in `metrics.py`/`stability.py` but never wired into the dashboard:

*   **Control Precision:** RMSE between commanded (`q_cmd`) and actual (`q`) joint positions — `compute_control_precision`.
*   **Dynamic Stability:** Mean variance of IMU pitch/roll during the run — `compute_imu_variance`.
*   **Energy Cost of Transport (CoT):** Motor power (voltage × current) integrated over distance traveled — `compute_cost_of_transport`.
*   **System Latency:** Cross-correlation lag between command and actuation — `compute_control_latency`.

**Correction (same day):** initially removed the 10 biomechanical metrics entirely from the API response and dashboard. Per follow-up direction, they've been restored as **display-only** metrics — `extract_metrics_from_dataframe` in `src/web/app.py` now computes and returns all 14 metrics (4 hardware + 10 biomechanical), but `baselines.py` still only defines thresholds/weights for the 4 hardware metrics, and `rules.py`'s `classify()` only iterates over keys present in `baselines.py` — so the biomechanical metrics are present in the response for reference (and still drive the per-task show/hide UI + 3D playback charts) but structurally cannot affect the weighted-sum score. Task-specific weight profiles in `baselines.py` (`TASK_WEIGHTS`) are defined only for the 4 hardware metrics (e.g. walking weights Dynamic Stability heaviest at 35%; reaching/manipulation zero out Cost of Transport and weight Control Precision + Latency at 45% each). The frontend (`index.html`, `script.js`) shows the 4 scoring metric cards always, plus a "Biomechanical Metrics (informational — not part of the score)" divider above the 10 old cards, which still show/hide per selected task as before. `tests/test_classification.py` and `tests/test_integration.py` were updated to match (14 expected metric keys).

> [!WARNING]
> **Known limitation reintroduced:** Public HuggingFace/LeRobot datasets that only contain joint positions (no torque/voltage/IMU) will score near-zero on Dynamic Stability and Cost of Transport for the actual score, since those columns are absent — the biomechanical metrics will still populate normally for those datasets since they only need `q`/`q_cmd`. This was an accepted tradeoff — the four metrics above are what the team wants measured for scoring, and are expected to be populated from our own telemetry pipeline going forward, not public datasets.

## Phase 6 (COMPLETE — Log Integration & Multi-Model Evaluation)

Addresses the two headline V1 problems: (a) log format compatibility could not be verified until a model was run on the physical robot, and (b) no way to check that metrics/scoring behave consistently across different models.

*   **Part 1: Log Integration (database + upfront validation):** **[DONE]**
    *   `src/storage/database.py` — SQLite-backed `BenchmarkDB` (stdlib `sqlite3`, no new dependencies; data lives in `data/benchmark.db` + `data/logs/`, already gitignored; `G1_BENCH_DATA_DIR` env var overrides the location). Tables: `logs` (registered files + validation reports) and `evaluations` (batched results). Tests inject an isolated instance via `src.storage.database._db_instance`.
    *   `src/ingestion/log_validator.py` — `validate_log()` exercises the real pipeline stages against an uploaded log (schema detection → normalisation → required columns → tick monotonicity/jitter → joint-array consistency/NaN rate → scoring-telemetry coverage → an actual end-to-end dry run of the extractors) and returns a structured pass/warn/fail report with overall status `valid`/`warnings`/`invalid`.
    *   `SchemaMapper.detect_format()` added (returns `native`/`hf_subarrays`/`hf_body`/`hf_flat`/`unknown`), mirroring `normalise()`'s branch order.
    *   API: `POST /api/logs` (register + validate), `GET /api/logs`, `GET /api/logs/{id}`, `DELETE /api/logs/{id}`.
*   **Part 2: Multi-Model Evaluation (parallel scoring + calibration):** **[DONE]**
    *   `src/classification/evaluator.py` — `run_parallel_evaluations()` scores several registered logs concurrently (ThreadPoolExecutor); `build_calibration_summary()` compares models per scoring metric: raw values, normalized sub-scores, mean/std/min/max, best model, task weight, and the current Class 1 ideal / Class 3 acceptable bounds for reference.
    *   `RuleBasedClassifier.score_breakdown()` added (per-metric value, normalized score, weight); `classify()` refactored to use it with identical numeric behaviour.
    *   API: `POST /api/evaluate` (`{log_ids, task}` → per-model results + calibration summary, persisted as one batch), `GET /api/evaluations?task=`.
*   **Part 3: Scoring-integrity fix:** **[DONE]** `extract_metrics_from_dataframe` moved to `src/features/extractor.py` (shared by upload endpoint, validator, and evaluator) with **availability gating**: scoring metrics whose telemetry columns are absent now return `None` (classifier skips them and renormalises weights) instead of `0.0` — previously a missing-voltage log scored a false-*perfect* Cost of Transport, since all four scoring metrics are lower-is-better. `metric_availability()` reports computability per metric.
*   **Part 4: Frontend:** **[DONE]** Two new dashboard sections in `index.html`/`library.js`/`style.css`: "Log Library — Physical Robot Log Integration" (register form, log table with validation badges, expandable per-check validation report) and "Multi-Model Evaluation" (select logs → run parallel evaluation → ranking cards + metric-comparison/calibration table).
*   **Part 5: Tests:** **[DONE]** `tests/test_log_integration.py` (11 tests): registration/validation of valid, kinematics-only (warnings), unknown-schema (invalid), and unreadable logs; log CRUD; parallel evaluation of three quality-differentiated baseline models; calibration structure; degraded models scoring measurably worse; batch persistence. Full suite: 51 passing (the 1 pre-existing `test_schema_mapper.py` Format-C failure is unrelated and predates this work).

## Immediate Next Steps

1. Recalibrate the numeric ideal/acceptable bounds in `baselines.py` (currently engineering placeholders) using the new multi-model calibration summaries from real G1 hardware runs.
2. *(Partially resolved by Phase 6)* Joint-position-only datasets are now flagged at registration (`warnings` status naming exactly which scoring metrics are unavailable) and unavailable metrics no longer poison the score — decide whether to additionally surface a "partial score" label on the main upload dashboard.
3. Consider a bulk re-validation endpoint so already-registered logs can be re-checked after pipeline/schema-mapper changes.
