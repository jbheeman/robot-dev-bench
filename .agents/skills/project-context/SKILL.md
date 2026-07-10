---
name: project-context
description: Architectural context, project objectives, and technical details for the Unitree G1-Edu Benchmarking & Classification Pipeline.
---

# CONTEXT DOCUMENT: Unitree G1-Edu Benchmarking & Classification Pipeline

## 1. Project Objective

Develop a post-run, digital benchmarking and classification tool for the Unitree G1-Edu humanoid robot. The system ingests telemetry logs, processes the time-series data, extracts performance metrics, and categorizes the robot's control policy into specific performance tiers (e.g., Class 1: Superhuman/Industrial, Class 2: Research, Class 3: Experimental).

## 2. Telemetry Ingestion & Storage

* **Middleware:** Data is pulled via CycloneDDS (Unitree SDK2).
* **Data Streams:**
  * `rt/lowstate` (Proprioceptive/Hardware): Joint position $q$, velocity $\dot{q}$, motor torque $\tau$, IMU (RPY, linear acceleration), and hardware ticks.
  * `rt/highstate` (State Estimator/Locomotion): Base velocity, odometry, foot contact states.

* **Storage Format:** Logs are containerized exclusively using Parquet (`.parquet`) for highly optimized, column-oriented post-run parsing and Sim-to-Real comparison in Pandas/ML pipelines.

## 3. Data Processing & Synchronization (Post-Run)

* **Non-Causal Filtering:** Utilize zero-phase digital filtering (e.g., forward-backward Butterworth / `filtfilt`) to eliminate high-frequency noise in velocity and torque estimates without introducing temporal lag.
* **Timestamp Alignment:** Handle packet loss via interpolation. Synchronize hardware state logs (`LowState`) with commanded state logs (`LowCmd`) using hardware tick timestamps to establish a clean ground-truth array.
* **Sim-to-Real Alignment:** The pipeline must identically ingest logs from physical hardware and physics simulators (Isaac Sim / Isaac Lab) to quantify the Sim-to-Real gap.

## 4. Feature Engineering (Key Metrics)

Metrics must be clinically-grounded and dynamically tailored to the specific evaluation task. We do not use a "one-size-fits-all" approach (e.g., penalizing asymmetry during a one-handed manipulation task).

* **Task-Agnostic Core Metrics:**
  * **Smoothness:** Log Dimensionless Jerk (LDLJ) and Spectral Arc Length (SPARC) to quantify trajectory fluidity.
  * **Range of Motion (ROM) Utilisation:** Angular displacement to measure dynamic capability usage.
* **Walking / Locomotion Metrics:**
  * **Symmetry Index:** Variance comparison between left/right kinematic chains.
  * **Periodicity:** Autocorrelation to measure gait cycle regularity.
* **Manipulation / Reaching Metrics:**
  * **End-Effector Precision:** Deviation and shaking at the wrist.
  * **Settling Time:** Time required to stabilize after reaching a target.
* **Jumping Metrics:**
  * **Flight Time:** Duration of zero ground contact.
  * **Peak Z-Axis Acceleration:** Explosive power measurement.
  * **Landing Jerk:** Impact absorption and joint stress upon landing.
* **Transitions (Standing Up / Sitting Down):**
  * **CoM Oscillation:** Postural stability and wobble during weight shifting.
  * **Transition Time:** Speed and fluidity of the phase change.

## 5. Model Training Data & Baselines

* **Human/Baseline Data:** Sourced from open-source humanoid datasets (e.g., Humanoid Everyday Dataset, Humanoid-Bench, Unitree Hugging Face repos, and retargeted CMU ACCAD motions).
* **Superhuman Data (Class 1 Tier):** Because human teleoperation is biologically limited, Class 1 baseline data is generated synthetically. Reinforcement Learning (RL) policies trained in Isaac Sim/Isaac Lab over millions of iterations produce the mathematically optimized trajectories and telemetry required to define the highest tier.

## 6. Classification Architecture

* **Primary Approach:** Supervised Rule-Based Scoring. Aggregate the engineered features (RMSE, CoT, latency, stability variance) against deterministic industry thresholds (referencing frameworks like Fraunhofer IPA).
* **Secondary Approach (Exploratory):** Unsupervised clustering (K-Means or DBSCAN) to group control policies dynamically based on telemetry features, useful when comparing varying RL models.

## 7. Dashboard Enhancements (Phase 4 Roadmap)

To maximize analytical capabilities for researchers, the following extensions are planned for the web dashboard:
* **Interactive Time-Series Charts (Telemetry Deep-Dive):** Plot raw telemetry (e.g., CoM oscillation, joint velocities) over time, synced to the 3D playback timeline to pinpoint failure modes.
* **Anomaly Markers on the Timeline:** Visually inject red markers on the 3D scrubber at exact timestamps of critical events (e.g., peak Z-acceleration, maximum landing jerk).
* **Side-by-Side Policy Comparison:** Enable A/B testing by uploading two `.parquet` files simultaneously to compare scores, deltas, and dual 3D playbacks.
* **Export to PDF / CSV Report:** Provide a 1-click export of the classification score, metrics table, and metadata for standardized Hugging Face model cards or research papers.
