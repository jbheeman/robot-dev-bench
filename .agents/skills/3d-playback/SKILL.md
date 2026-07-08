---
name: 3d-playback
description: Plan and architectural guidelines for implementing a 3D WebGL robot movement playback viewer.
---

# 3D Movement Playback Implementation Plan

> [!CAUTION]
> **DO NOT IMPLEMENT THIS FEATURE** unless the user explicitly requests you to activate this skill and build the 3D viewer. This document is strictly for future reference and architectural alignment.

## Overview
It is possible to reconstruct and visualize the 3D movement of the Unitree G1-Edu robot entirely from the telemetry log file. Since the log file (e.g., Parquet) contains the base position (`odometry`), base orientation (`rpy`), and every individual joint angle (`q`), we have all the kinematic data required to render the robot's exact pose at any given timestamp.

## Implementation Architecture

If activated by the user, the implementation will require a **Moderate to High** level of effort and the following components:

### 1. Prerequisites (Source from GitHub)
Before any coding can begin, the AI must source the physical description files for the Unitree G1-Edu from the Unitree GitHub repository:
- **URDF File:** The XML file that describes the robot's kinematic tree (how joints and links are connected).
- **3D Meshes:** The visual files (e.g., `.stl`, `.obj`, or `.dae`) referenced by the URDF that represent the actual geometry of the robot links.

### 2. Frontend Integration (WebGL)
- **Rendering Engine:** Integrate a WebGL library like **Three.js** into the web dashboard.
- **URDF Loader:** Use a JavaScript package (like `urdf-loaders` for Three.js) to parse the URDF file and generate the 3D skeleton in the browser.
- **Scene Setup:** Configure the camera, lighting, and a floor grid to properly visualize the robot in 3D space.

### 3. Backend Streaming
- **Data Delivery:** The FastAPI backend must expose an endpoint or WebSocket to send the time-series kinematics data (`q`, `odometry`, `rpy`) to the frontend. 
- **Downsampling:** To prevent browser lag, the backend may need to downsample the telemetry (e.g., from 1000Hz down to 30Hz or 60Hz) before sending it to the client.

### 4. Animation Loop
- **Playback Logic:** The frontend will use a JavaScript animation loop (`requestAnimationFrame`) to iterate through the telemetry frames.
- **Pose Updates:** For each frame, the UI will update the respective joint angles and base transforms on the loaded 3D model, effectively acting as a video player for the logged run.
