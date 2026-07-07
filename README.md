# Unitree G1-Edu Benchmarking & Classification Pipeline

This repository contains a post-run, digital benchmarking and classification tool for the Unitree G1-Edu humanoid robot.

## Objective
To ingest telemetry logs (MCAP, HDF5, Parquet), process time-series data, extract performance metrics, and categorize the robot's control policy into performance tiers (Superhuman/Industrial, Research, Experimental).

## Setup
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt` (or via `pip install -e .` if using pyproject.toml)

## Data Structure
- Place raw logs in `data/raw/`
- Place simulator baseline data in `data/sim/`

*Note: The `data/` directory is ignored by git.*
