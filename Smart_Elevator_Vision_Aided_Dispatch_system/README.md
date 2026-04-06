# 🏢 Vision-Aided Smart Elevator Dispatch System (VAD)

> **Optimising vertical transport efficiency via real-time Computer Vision and intelligent capacity management.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-ultralytics-00BFFF)](https://github.com/ultralytics/yolov5)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Traditional elevator dispatch systems are **blind** — they treat every button press as equally valid, regardless of whether anyone is still waiting. This leads to two costly inefficiencies:

| Problem | Impact |
|---|---|
| **Ghost Stops** — elevator serves an empty floor | Wasted energy + longer trips for existing passengers |
| **Capacity Violations** — full elevator stops to pick up more | Delays everyone |

The **VAD System** replaces blind logic with a *context-aware pipeline*:

```
Camera Feed ──► YOLOv5 + Centroid Tracker ──► Ghost Call Filter
                                           ──► Capacity Management Module
                                           ──► Dynamic Priority Scoring ──► Dispatch
```

**Projected improvements (simulation):**

| Metric | Traditional | VAD |
|---|---|---|
| Ghost Stop Reduction | 0 % | **85 – 95 %** |
| Average Trip Time | baseline | **8 – 15 % faster** |
| Energy Consumption | baseline | **5 – 10 % lower** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1 – Data Acquisition                         │
│  IP Cameras (hall + cabin)  ·  Call Buttons  ·  ECU │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  LAYER 2 – Core Processing (The Brain)              │
│                                                     │
│  ┌─────────────┐  ┌──────┐  ┌──────┐  ┌─────────┐  │
│  │ VPM         │  │ GCF  │  │ CMM  │  │   SDE   │  │
│  │ YOLOv5      │─►│Ghost │─►│Capac │─►│Priority │  │
│  │ + Centroid  │  │Filter│  │Mgmt  │  │ P_i     │  │
│  └─────────────┘  └──────┘  └──────┘  └─────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  LAYER 3 – Control & Actuation                      │
│  Elevator ECU  ·  Floor "FULL" displays             │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 Core Algorithms

### 1. Vision Pipeline — `VisionProcessingModule`
- **YOLOv5s** (loaded via `torch.hub`) detects people in the camera frame.
- A configurable **Region of Interest (ROI)** focuses detection on the waiting area only.
- **CentroidTracker** assigns each person a unique persistent ID across frames, producing the validated count **C_i**.

### 2. Ghost Call Filter — `GhostCallFilter`
```
If  B_i = True  AND  C_i = 0  for  Δt > T_ghost  →  B_i := False
```
Cancels calls where the passenger left before the elevator arrived.  
`T_ghost` defaults to **90 frames ≈ 3 seconds** at 30 fps (tunable).

### 3. Capacity Management Module — `CapacityManagementModule`
```
If  C_elevator ≥ C_max  →  ignore external pickup calls
                         →  send "FULL" signal to requesting floor
```

### 4. Dynamic Priority Score — `SmartDispatchEngine`
```
P_i = W1·C_i  +  W2·(1/D_i)  +  W3·A_i

C_i  = validated people count at floor i   (W1 = 10.0 — throughput)
D_i  = distance from elevator to floor i   (W2 =  1.0 — efficiency)
A_i  = wait time in seconds                (W3 =  0.1 — anti-starvation)
```

### 5. Multi-Elevator Group Control (Extension)
```
S_{i,E} = W_cost · (P_i / T_ETA(i,E)) · W_DIR · W_CANCEL
```
Assigns the best elevator to each call, penalising direction reversal (`W_DIR = 0.1`).

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/<your-username>/vad-elevator-system.git
cd vad-elevator-system
```

### 2. Install dependencies
```bash
# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All other packages
pip install -r requirements.txt
```
> YOLOv5 weights (~14 MB) are downloaded **automatically** on the first run via `torch.hub`.

### 3. Run

```bash
# Text simulation — no camera or GPU required
python run_vad.py --mode sim

# Custom simulation
python run_vad.py --mode sim --floors 15 --steps 100 --ghost 0.4

# Live webcam
python run_vad.py --mode live

# Live with custom ROI (x1 y1 x2 y2)
python run_vad.py --mode live --roi 100 50 540 430 --capacity 15
```

### 4. Run unit tests
```bash
python test_vad.py
```

---

## 📁 Repository Structure

```
vad-elevator-system/
│
├── vad_main.py          # All core modules (VPM · GCF · CMM · SDE)
├── run_vad.py           # CLI entry point
├── test_vad.py          # Unit tests (12 tests, no camera needed)
│
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--mode` | `sim` | `sim` = text simulation, `live` = webcam |
| `--floors` | `10` | Number of building floors |
| `--steps` | `50` | Simulation ticks |
| `--ghost` | `0.3` | Ghost call probability [0–1] |
| `--camera` | `0` | Camera index |
| `--floor` | `0` | Floor being monitored |
| `--roi` | None | `x1 y1 x2 y2` pixel bounding box |
| `--capacity` | `12` | Max elevator capacity (C_max) |
| `--w1` | `10.0` | Priority weight: people count |
| `--w2` | `1.0` | Priority weight: distance |
| `--w3` | `0.1` | Priority weight: wait age |

---

## 🖥️ Hardware Used (Original Report)
- NVIDIA RTX 3060 (inference)
- YOLOv5s — 7.2 M parameters, ~30 FPS real-time

---

## 📄 License
MIT © 2025 Hossen Md Jisan
