---
name: project-progress
description: Tracks the current implementation progress of the Unitree G1-Edu Benchmarking pipeline project.
---

# Project Implementation Progress

**ATTENTION AI AGENTS**:
1. Use this document to understand the current state of the project and align on immediate next steps. Do not duplicate completed work.
2. **CRITICAL RULE**: You MUST update this document to reflect any newly completed work or changed priorities **every time before pushing code or finalizing your tasks** — this applies in particular to any change to `src/classification/clustering.py`. Keep the status accurate.

Based on `Context.txt`, here is the current global status of the implementation:

* **Part 1: Project Objective:** Understood and ongoing.
* **Part 2: Telemetry Ingestion & Storage:** **[PARTIAL]** `live_subscriber.py`, `exporter.py`, and `data_models.py` implement live CycloneDDS capture (`rt/lowstate` / `rt/highstate`) and DataFrame/Parquet export. `dds_client.py` and `parser.py` are still stub files (comments only) — no MCAP/HDF5 post-run log parsing yet.
* **Part 3: Data Processing & Synchronization (Post-Run):** **[NOT STARTED]** `filter.py` (zero-phase Butterworth/`filtfilt`) and `sync.py` (timestamp alignment between LowState/HighState) are stub files — no implementation yet.
* **Part 4: Feature Engineering (Key Metrics):** **[NOT STARTED]** `metrics.py` and `stability.py` are stub files. RMSE, IMU variance, Cost of Transport, control latency, and torque-spike detection are not yet implemented.
* **Part 5: Model Training Data & Baselines:** Not applicable to code yet — depends on sourced human/RL baseline datasets.
* **Part 6: Classification Architecture:**
  * Secondary / Exploratory (clustering): **[DONE]** `clustering.py` implements `PolicyClusterer` — K-Means/DBSCAN, silhouette scoring, and a `find_optimal_k` sweep helper.
  * Primary (rule-based scoring): **[NOT STARTED]** `rules.py` is a stub file.

## Immediate Next Steps

**Status: The clustering (exploratory) path in Part 6 is done, but it has nothing real to consume yet — `PolicyClusterer` expects feature columns (`rmse`, `imu_variance`, `cost_of_transport`, `control_latency`, `torque_spikes`) that no module currently produces, since Parts 3 and 4 aren't implemented.**

Priority order:
1. **Part 3: Data Processing** — implement `filter.py` (zero-phase filtering) and `sync.py` (tick-based LowState/HighState alignment).
2. **Part 4: Feature Engineering** — implement `metrics.py` and `stability.py` to actually produce the feature columns `clustering.py` (and eventually `rules.py`) consume.
3. **Part 6 (Primary): Rule-based classification** — implement `rules.py` once Part 4's features exist.

**Test coverage note:** `tests/test_classification.py`, `tests/test_features.py`, and `tests/test_processing.py` are currently empty stub files — there are no tests yet for clustering, feature engineering, or processing/sync/filter logic.
