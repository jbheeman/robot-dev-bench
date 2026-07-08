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

* **Storage Format:** Logs should be containerized using MCAP, HDF5, or Parquet for efficient post-run parsing and Sim-to-Real comparison.

## 3. Data Processing & Synchronization (Post-Run)

* **Non-Causal Filtering:** Utilize zero-phase digital filtering (e.g., forward-backward Butterworth / `filtfilt`) to eliminate high-frequency noise in velocity and torque estimates without introducing temporal lag.
* **Timestamp Alignment:** Handle packet loss via interpolation. Synchronize hardware state logs (`LowState`) with commanded state logs (`LowCmd`) using hardware tick timestamps to establish a clean ground-truth array.
* **Sim-to-Real Alignment:** The pipeline must identically ingest logs from physical hardware and physics simulators (Isaac Sim / Isaac Lab) to quantify the Sim-to-Real gap.

## 4. Feature Engineering (Key Metrics)

* **Control Precision:** Root Mean Square Error (RMSE) between commanded target joint positions and actual encoder readings.
* **Dynamic Stability:** Variance in IMU roll and pitch, and Center of Mass (CoM) stability during gait cycles.
* **Cost of Transport (CoT):** Integrated power consumption (voltage $\times$ current) over trajectory distance.
* **Control Latency:** Temporal delay between policy output (command) and mechanical actuation.
* **Hardware Stress:** Detection of extreme torque $\tau$ spikes.

## 5. Model Training Data & Baselines

* **Human/Baseline Data:** Sourced from open-source humanoid datasets (e.g., Humanoid Everyday Dataset, Humanoid-Bench, Unitree Hugging Face repos, and retargeted CMU ACCAD motions).
* **Superhuman Data (Class 1 Tier):** Because human teleoperation is biologically limited, Class 1 baseline data is generated synthetically. Reinforcement Learning (RL) policies trained in Isaac Sim/Isaac Lab over millions of iterations produce the mathematically optimized trajectories and telemetry required to define the highest tier.

## 6. Classification Architecture

* **Primary Approach:** Supervised Rule-Based Scoring. Aggregate the engineered features (RMSE, CoT, latency, stability variance) against deterministic industry thresholds (referencing frameworks like Fraunhofer IPA).
* **Secondary Approach (Exploratory):** Unsupervised clustering (K-Means or DBSCAN) to group control policies dynamically based on telemetry features, useful when comparing varying RL models.
