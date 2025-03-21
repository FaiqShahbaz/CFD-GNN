# Basic Flows: CFD + Graph Neural Networks

This folder contains two classic flow problems solved using traditional CFD solvers and enhanced with Graph Neural Networks (GNNs) for fast surrogate modeling.

---

## 📌 Problems Included

### 1. Lid-Driven Cavity Flow (`cavityFlow.py`)
- Solves the **Navier–Stokes equations** using finite difference methods.
- Computes the velocity field in a 2D square cavity where the top lid moves and induces flow.
- Converts the velocity field into a graph, where:
  - Nodes = grid points (with normalized spatial coordinates)
  - Node labels = velocity magnitudes
  - Edges = adjacency based on grid neighbors
- Trains a GCN (Graph Convolutional Network) to predict the velocity magnitude.
- Visualizes **actual vs predicted vs error**.

### 2. 2D Pipe Potential Flow (`pipeFlow.py`)
- Solves the **Laplace equation** for a 2D pipe with specified inlet/outlet conditions.
- Simulates potential flow between two plates.
- Builds a graph where:
  - Nodes = grid points with (x, y)
  - Node labels = potential value
  - Edges = horizontal/vertical grid connections
- Trains a GCN to predict the potential field.
- Outputs a comparison of actual vs predicted potential and absolute error.

---

## 🧪 Environment Setup

To create and activate a clean environment using the provided `requirements.txt`:

### ▶️ Using `virtualenv` (Python built-in):
```bash
# Create virtual environment
python -m venv gnn-cfd-env

# Activate (Windows)
gnn-cfd-env\Scripts\activate

# Activate (Linux/macOS)
source gnn-cfd-env/bin/activate

# Install dependencies
pip install -r ../requirements.txt
