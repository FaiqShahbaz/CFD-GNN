# SUMMARY
# The code simulates fluid flow in a lid-driven cavity using the Navier–Stokes equations to generate accurate velocity fields. 
# It then converts the grid data into a graph, where each node (grid point) holds spatial coordinates and a velocity magnitude. 
# A Graph Neural Network with multiple GCN layers is trained on this graph to learn and predict the velocity magnitude quickly
# acting as a surrogate model. Finally, the code visualizes and saves comparisons between the simulation results and the GNN predictions.

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Numerical Solver (Navier-Stokes)
def solve_navier_stokes(nx, ny, nt=300, nit=30, lid_velocity=1.0, dt=0.001, rho=1.0, nu=0.1):
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    u, v, p = np.zeros((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx))

    for t in range(nt):
        un, vn, pn = u.copy(), v.copy(), p.copy()

        # Pressure update loop
        for _ in range(nit):
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                              (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                             (2 * (dx**2 + dy**2))
                             - rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                             ((1 / dt) * ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) +
                                          (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy))))
            # Apply boundary conditions for pressure
            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[:, 0] = p[:, 1]
            p[-1, :] = 0

        # Update u velocity
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        # Update v velocity
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        # Set boundary conditions for velocities
        u[0, :], u[:, 0], u[:, -1] = 0, 0, 0
        u[-1, :] = lid_velocity  # Moving lid at the top
        v[0, :], v[-1, :], v[:, 0], v[:, -1] = 0, 0, 0, 0

    return u, v

# Create Graph Data
def create_graph_data(u, v, nx, ny):
    velocity = np.sqrt(u**2 + v**2)
    nodes = torch.tensor([[x / nx, y / ny] for y in range(ny) for x in range(nx)], dtype=torch.float)
    labels = torch.tensor(velocity.flatten(), dtype=torch.float).unsqueeze(1)

    edges = []
    for y in range(ny):
        for x in range(nx):
            idx = y * nx + x
            if x < nx - 1:
                edges += [[idx, idx + 1], [idx + 1, idx]]
            if y < ny - 1:
                edges += [[idx, idx + nx], [idx + nx, idx]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=nodes, edge_index=edge_index, y=labels)

# GNN Model
class CavityGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# Training and Visualization
def run_and_visualize(nx, ny):
    print(f"Running simulation for grid size: {nx}x{ny}")
    # Dynamically adjust dt and nit based on grid resolution.
    # For grid 21x21, dt=0.001 and nit=30; for finer grids, reduce dt and increase nit.
    dt = 0.001 * (21 / nx)  # e.g., for nx=61, dt becomes ~0.000344.
    nit_adjusted = int(30 * (nx / 21))
    print(f"Using dt = {dt} and nit = {nit_adjusted}")

    u, v = solve_navier_stokes(nx, ny, dt=dt, nit=nit_adjusted)

    # Check for numerical issues
    if np.isnan(u).any() or np.isnan(v).any():
        print("Warning: Simulation produced NaN values!")

    data = create_graph_data(u, v, nx, ny)
    model = CavityGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    actual = data.y.numpy()
    predicted = pred.detach().numpy()
    error = np.abs(actual - predicted)

    plt.figure(figsize=(15, 4))
    titles = ['Actual Velocity Magnitude', 'GNN Predicted Velocity Magnitude', 'Prediction Error']
    fields = [actual, predicted, error]
    cmaps = ['viridis', 'viridis', 'plasma']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(fields[i].reshape(ny, nx), cmap=cmaps[i], origin='lower')
        plt.colorbar(label=titles[i])
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(titles[i])

    plt.tight_layout()
    plt.savefig(f'cavity_flow_{nx}x{ny}.png')
    plt.show()

# Run simulations for different grid sizes
for nx, ny in [(21, 21), (41, 41), (61, 61)]:
    run_and_visualize(nx, ny)

