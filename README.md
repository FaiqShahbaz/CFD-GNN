# CFD-GNN

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![framework](https://img.shields.io/badge/built%20with-PyTorch-orange)
![graph](https://img.shields.io/badge/graph-Torch%20Geometric-blue)
![domain](https://img.shields.io/badge/domain-CFD--GNN-black)

A structured collection of **Graph Neural Network (GNN)** implementations for learning solutions of **Computational Fluid Dynamics (CFD)** systems.

This repository explores graph-based surrogate modeling of partial differential equations (PDEs), where numerical simulations are used to train neural networks defined over mesh connectivity graphs.

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

$$
\mathcal{N}(u) = 0,
$$

where $u$ denotes the state variables (e.g., velocity and pressure), and  
$\mathcal{N}$ represents a nonlinear differential operator derived from conservation laws.

For example, the incompressible Navier–Stokes equations are written as

$$
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u} \cdot \nabla)\mathbf{u}
= -\nabla p
+ \nu \nabla^2 \mathbf{u},
$$

$$
\nabla \cdot \mathbf{u} = 0.
$$

Classical solvers approximate these equations using mesh-based discretization methods such as finite difference, finite volume, or finite element techniques.

In this repository, simulation outputs are converted into graph representations:

- **Nodes** → mesh points or cell centers  
- **Edges** → mesh connectivity  
- **Node features** → geometric and physical variables  
- **Edge features** → spatial relationships  

A GNN model $u_\theta$ is trained to approximate the numerical solution:

$$
u_\theta \approx u_{\text{CFD}}.
$$

The learning problem is formulated as supervised regression over mesh graphs.

---

## Repository Structure

```
CFD-GNN/
│
├── projects/
│   ├── basic_flows/
│   │   ├── cavityFlow.py
│   │   ├── pipeFlow.py
│   │   └── results/
│   │
│   └── airfoil-gnn-openfoam/
│       ├── cfd/                  # OpenFOAM base case
│       ├── cfd_scripts/          # Case generation & execution
│       ├── gnn/                  # Graph construction & models
│       │   ├── models/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── ...
│       └── results/              # Visualization outputs
│
├── README.md
└── LICENSE
```

Each project directory contains source code, preprocessing scripts, model implementations, and evaluation pipelines.

---

## Current Projects

### 1️⃣ Basic Flows

Structured-grid CFD examples coupled with graph neural networks.

This project was developed as an introductory step to gain practical understanding of GNNs and how they can be applied to PDE-based problems. It focuses on building intuition around graph construction, training workflows, and model evaluation in a controlled setting.

**Implemented systems:**
- Lid-driven cavity flow (2D incompressible Navier–Stokes)
- 2D potential pipe flow (Laplace equation)

**Workflow:**
1. Solve the PDE using a classical numerical method  
2. Construct a graph from the structured grid  
3. Train a GNN using PyTorch Geometric  
4. Compare predicted and numerical solutions  

📂 `projects/basic_flows/`

---

### 2️⃣ Airfoil Flow with OpenFOAM + GNN

A full CFD-to-GNN workflow built around OpenFOAM simulations on unstructured meshes.

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

This project extends the foundational work into more realistic aerodynamic configurations.

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

---

## Goals of This Repository

- Develop reproducible CFD–GNN pipelines  
- Benchmark graph neural networks against classical solvers  
- Explore data-driven surrogate modeling of fluid systems  
- Build a structured research portfolio in Scientific Machine Learning  

Planned extensions include:

- Physics-informed regularization terms  
- Multi-condition generalization  
- 3D flow configurations  
- Hybrid solver–GNN coupling strategies  

---

## Author

Faiq Shahbaz  
GitHub: https://github.com/FaiqShahbaz