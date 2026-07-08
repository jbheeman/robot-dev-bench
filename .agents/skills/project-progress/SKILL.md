---
name: project-progress
description: Tracks the current implementation progress of the Unitree G1-Edu Benchmarking pipeline project.
---

# Project Implementation Progress

**ATTENTION AI AGENTS**: 
1. Use this document to understand the current state of the project and align on the immediate next steps. Do not duplicate completed work.
2. **CRITICAL RULE**: You MUST update this document to reflect any newly completed work or changed priorities **every time before pushing code or finalizing your tasks**. Keep the status accurate.
Based on the `project-context` skill, here is the current global status of the implementation:

*   **Part 1: Project Objective:** Understood and ongoing.
*   **Part 2: Telemetry Ingestion & Storage:** **[DONE]** `live_subscriber.py`, `exporter.py`, and `data_models.py` are implemented for capturing hardware data via CycloneDDS and exporting it to Pandas/Parquet.
*   **Part 3: Data Processing & Synchronization (Post-Run):** **[DONE]** Zero-phase digital filtering (`filter.py`) and timestamp alignment/merging (`synchronizer.py`) are implemented and validated.
*   **Part 4: Feature Engineering (Key Metrics):** **[DONE]** Metric extractors for control precision (RMSE), Cost of Transport (CoT), control latency, hardware stress, and stability (IMU and CoM variance) are implemented in `metrics.py` and `stability.py` and covered by unit tests in `test_features.py`.
*   **Part 5: Model Training Data & Baselines:** **[DONE]** Defined static baseline bounds for optimal RL/humanoid target limits and updated Class 2 and 3 with empirical data from the Humanoid Everyday Dataset (`baselines.py`).
*   **Part 6: Classification Architecture:**
    *   Primary (rule-based scoring): **[DONE]** Implemented the supervised rule-based scorer in `rules.py` and integrated it.
    *   Secondary (Exploratory clustering): **[DONE, NOT INTEGRATED]** `clustering.py` implements `PolicyClusterer` (K-Means/DBSCAN, silhouette scoring, `find_optimal_k` sweep) for grouping policy runs without labelled tiers. It is not yet wired into `rules.py` or the `src/web` pipeline/tests — it currently only runs standalone.
*   **Part 7: Web Infrastructure & Integration:** **[DONE]** Built a premium modern local dashboard (`src/web`) with a FastAPI backend. The `/api/upload` endpoint now accepts `.parquet` telemetry logs, parses them with `pd.read_parquet`, runs the full feature extraction pipeline (`metrics.py`, `stability.py`), and classifies the result with the real rule-based engine. Mock data generation has been fully removed.

## Immediate Next Steps

**Status: The pipeline was implemented as COMPLETE (Parts 1–8), but the branch that did this work is currently mid-merge into `Alvin` with unresolved conflicts — see Known Bugs / Blockers. Resolve those before trusting any status below at the file level.**

*   **Part 8: End-to-End Testing:** **[DONE, per source branch]** A synthetic `.parquet` generator (`scripts/generate_test_parquet.py`) and a full integration test suite (`tests/test_integration.py`) were implemented. All 10 integration tests reportedly pass, validating the complete user journey from file upload through real metric extraction to classification output — re-verify after the merge conflict below is resolved, since some of the files these tests cover are currently unparseable.

Optional future work:
*   **Optional Enhancement:** Add a data conversion utility (`scripts/convert_to_parquet.py`) that transforms raw MCAP or ROS2 bag files into the expected Parquet schema.
*   **Integrate clustering:** Wire `PolicyClusterer` (`clustering.py`) into the rule-based pipeline or web dashboard as an exploratory/secondary view, and add test coverage for it.

## Known Bugs / Blockers

*   **Active unresolved git merge on branch `Alvin`** (merging in `d8d255e...`): as of last check, `src/features/stability.py`, `src/processing/filter.py`, `tests/test_classification.py`, `tests/test_features.py`, and `tests/test_processing.py` still contain literal `<<<<<<< HEAD` / `=======` / `>>>>>>>` conflict markers in the working tree, making them invalid Python. `src/classification/rules.py` and `src/features/metrics.py` were conflicted but have since been resolved. Do not treat Parts 3, 4, 6, or 8 as verified-working until `git status` shows no `UU` entries and the affected files import cleanly.
