# GNN-CFD-Coupling

This repository explores the use of Graph Neural Networks (GNNs) as surrogate models for solving Computational Fluid Dynamics (CFD) problems. Traditional numerical solvers generate simulation data, which is then transformed into graph-based datasets for GNN training.

The aim is to combine the physical accuracy of CFD with the speed and flexibility of deep learning, enabling efficient flow prediction in scientific and engineering contexts.

---

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Framework: PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-red?logo=pytorch)
![Graph Library](https://img.shields.io/badge/Graph%20Library-Torch%20Geometric-blue)

---

## Table of Contents

- [Overview](#overview)
- [Modules](#modules)
- [File Structure](#file-structure)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project demonstrates how GNNs, particularly Graph Convolutional Networks (GCNs), can approximate CFD solutions over structured domains. Each physical problem is solved using conventional numerical methods, and then converted into graph data for supervised training. 

Applications include:
- Accelerated flow prediction
- Surrogate modeling
- Physics-informed machine learning
- Educational demonstration of graph-based learning on PDEs

---

## Modules

### `basic_flows/`  
A collection of introductory CFD problems coupled with GNNs:
- Lid-driven cavity flow (Navier–Stokes)
- 2D pipe potential flow (Laplace)

Each case includes:
- Traditional solver implementation
- Graph construction from numerical grids
- GCN training using PyTorch Geometric
- Visual output and error analysis

For full implementation details, environment setup, and results:
> **See `basic_flows/README.md`**

---

## File Structure

```bash
GNN-CFD-Coupling/
├── basic_flows/
│   ├── cavityFlow.py
│   ├── pipeFlow.py
│   ├── requirements.txt
│   ├── README.md
│   └── results/
│       ├── cavity_flow_21x21.png
│       ├── ...
│       └── pipe_flow_comparison_30x15.png
├── README.md           # ← (You are here)
└── LICENSE
