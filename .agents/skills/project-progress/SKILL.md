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
*   **Part 6: Classification Architecture:** **[DONE]** Implemented the supervised rule-based scorer in `rules.py` and integrated it.
*   **Part 7: Web Infrastructure & Integration:** **[DONE]** Built a premium modern local dashboard (`src/web`) with a FastAPI backend. The `/api/upload` endpoint now accepts `.parquet` telemetry logs, parses them with `pd.read_parquet`, runs the full feature extraction pipeline (`metrics.py`, `stability.py`), and classifies the result with the real rule-based engine. Mock data generation has been fully removed.

## Immediate Next Steps

**Status: The pipeline is COMPLETE. All Parts 1–7 are implemented and validated.**

*   **Part 8: End-to-End Testing:** **[DONE]** A synthetic `.parquet` generator (`scripts/generate_test_parquet.py`) and a full integration test suite (`tests/test_integration.py`) were implemented. All 10 integration tests pass, validating the complete user journey from file upload through real metric extraction to classification output.

Optional future work:
*   **Optional Enhancement:** Add a data conversion utility (`scripts/convert_to_parquet.py`) that transforms raw MCAP or ROS2 bag files into the expected Parquet schema.

## Known Bugs / Blockers

*(No known bugs currently. Add items here if any issues are blocking progress.)*
