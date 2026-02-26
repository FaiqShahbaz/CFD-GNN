# CFD-GNN

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![framework](https://img.shields.io/badge/built%20with-PyTorch-orange)
![graph](https://img.shields.io/badge/graph-Torch%20Geometric-blue)
![domain](https://img.shields.io/badge/domain-CFD--GNN-black)

A structured collection of **Graph Neural Network (GNN)** implementations for learning solutions of **Computational Fluid Dynamics (CFD)** systems.

This repository investigates graph-based surrogate modeling of partial differential equations (PDEs), where high-fidelity numerical simulations are used to train neural networks defined on mesh connectivity graphs.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Current Projects](#current-projects)
- [Methodological Framework](#methodological-framework)
- [Goals](#goals-of-this-repository)
- [Author](#author)

---

## Overview

Many CFD problems are governed by nonlinear PDEs of the form

$begin:math:display$
\\mathcal\{N\}\(u\) \= 0\,
$end:math:display$

where $begin:math:text$ u $end:math:text$ denotes the state variables (e.g., velocity and pressure), and  
$begin:math:text$ \\mathcal\{N\} $end:math:text$ represents a nonlinear differential operator derived from conservation laws.

For example, incompressible Navier–Stokes equations are written as

$begin:math:display$
\\frac\{\\partial \\mathbf\{u\}\}\{\\partial t\}
\+ \(\\mathbf\{u\} \\cdot \\nabla\)\\mathbf\{u\}
\= \-\\nabla p
\+ \\nu \\nabla\^2 \\mathbf\{u\}\,
$end:math:display$

$begin:math:display$
\\nabla \\cdot \\mathbf\{u\} \= 0\.
$end:math:display$

Classical solvers approximate these equations using mesh-based discretization (finite difference, finite volume, or finite element methods).

In this repository, simulation outputs are converted into graph representations:

- **Nodes** → mesh points or cell centers  
- **Edges** → mesh connectivity  
- **Node features** → geometric and physical variables  
- **Edge features** → relative spatial relationships  

A GNN model $begin:math:text$ u\_\\theta $end:math:text$ is then trained to approximate the numerical solution field:

$begin:math:display$
u\_\\theta \\approx u\_\{\\text\{CFD\}\}\.
$end:math:display$

The learning problem is formulated as supervised regression over mesh graphs.

---

## Repository Structure

```
CFD-GNN/
│
├── projects/
│   ├── basic_flows/            # Structured grid CFD + GNN
│   └── airfoil-gnn-openfoam/   # OpenFOAM + unstructured mesh GNN
│
├── README.md
└── LICENSE
```

Each project directory contains:

- Source code
- Data preprocessing scripts
- Model architectures
- Training and evaluation pipelines
- Generated results and visualizations
- Independent documentation

---

## Current Projects

### 1️⃣ Basic Flows

Structured-grid CFD systems coupled with graph neural networks.

**Implemented systems:**
- Lid-driven cavity flow (2D incompressible Navier–Stokes)
- 2D potential pipe flow (Laplace equation)

**Pipeline:**
1. Solve PDE using finite difference methods  
2. Construct graph from structured grid  
3. Train GNN using PyTorch Geometric  
4. Compare predicted and numerical fields  

This module provides a controlled environment for evaluating graph-based approximations of PDE solutions.

📂 `projects/basic_flows/`

---

### 2️⃣ Airfoil Flow with OpenFOAM + GNN

A full CFD-to-GNN workflow using OpenFOAM simulations on unstructured meshes.

**Workflow:**
1. Generate parameterized airfoil cases (OpenFOAM)
2. Run simulations and export VTK/VTU outputs
3. Convert mesh data into graph objects
4. Normalize and split dataset
5. Train a MeshGraphNet-style architecture
6. Evaluate pressure and velocity field predictions  
   (optionally lift and drag coefficients)

**Key characteristics:**
- Unstructured mesh handling  
- Edge-conditioned message passing  
- Global pooling for aerodynamic quantities  
- Modular training and evaluation scripts  

This project demonstrates surrogate modeling on realistic aerodynamic geometries.

📂 `projects/airfoil-gnn-openfoam/`

---

## Methodological Framework

Across projects, the workflow follows:

1. **Numerical Simulation**  
   Generate reference solutions using classical discretization methods.

2. **Graph Construction**  
   Encode mesh topology and geometric relationships as graph structures.

3. **Supervised Training**  
   Minimize regression loss between predicted and numerical fields.

4. **Evaluation**  
   Quantify performance using:
   - Mean Squared Error (MSE)
   - Relative L2 error
   - Field visualizations

The approach aligns with graph-based surrogate modeling in scientific machine learning.

---

## Goals of This Repository

- Develop reproducible CFD–GNN pipelines  
- Benchmark graph neural networks against classical solvers  
- Explore data-driven surrogate modeling of fluid systems  
- Provide a structured research portfolio in Scientific Machine Learning  

Planned extensions include:

- Physics-informed regularization terms  
- Multi-condition generalization  
- 3D flow configurations  
- Hybrid solver–GNN coupling strategies  

---

## Author

Faiq Shahbaz  
GitHub: https://github.com/FaiqShahbaz