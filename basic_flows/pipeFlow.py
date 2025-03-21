#SUMMARY
#This code simulates potential flow in a 2D pipe using the Laplace equation with finite differences,
#  creates a graph-based dataset from the simulation results, and trains a Graph Neural Network (GNN) 
# to predict the potential field. It then combines three plots—showing the actual numerical potential,
#  the GNN’s predicted potential, and the absolute error—into a single figure for easy comparison. 
# The final figure is saved as a PNG file, making it convenient for later review.


import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

# Numerical Solver for Laplace Equation (Finite Difference Method)
def solve_laplace(nx, ny, max_iter=1000, tol=1e-4):
    phi = np.zeros((ny, nx))
    phi[:, 0] = 1  # inlet potential (left boundary)
    phi[:, -1] = 0  # outlet potential (right boundary)

    for iteration in range(max_iter):
        phi_old = phi.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                phi[i, j] = 0.25 * (phi_old[i+1, j] + phi_old[i-1, j] +
                                    phi_old[i, j+1] + phi_old[i, j-1])
        # Neumann boundary conditions on top and bottom walls
        phi[0, 1:-1] = phi[1, 1:-1]
        phi[-1, 1:-1] = phi[-2, 1:-1]

        if np.max(np.abs(phi - phi_old)) < tol:
            break

    return phi

# Create dataset from numerical solver: 2D pipe grid
def create_2d_pipe_grid(nx=10, ny=5):
    nodes, edges, labels = [], [], []
    phi_actual = solve_laplace(nx, ny)

    for y in range(ny):
        for x in range(nx):
            nodes.append([x, y])
            labels.append(phi_actual[y, x])

    nodes = torch.tensor(nodes, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

    for y in range(ny):
        for x in range(nx):
            idx = y * nx + x
            if x < nx - 1:
                edges += [[idx, idx + 1], [idx + 1, idx]]
            if y < ny - 1:
                edges += [[idx, idx + nx], [idx + nx, idx]]
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=nodes, edge_index=edges, y=labels)

# GNN Model Architecture
class CFDGNN(torch.nn.Module):
    def __init__(self):
        super(CFDGNN, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# Function to run the model, combine plots and save the figure
def run_and_plot(nx, ny):
    # Create the dataset from numerical simulation
    data = create_2d_pipe_grid(nx, ny)
    actual = data.y.numpy()

    # Initialize and train the GNN model
    model = CFDGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(data).numpy()
    error = np.abs(actual - predictions)

    # Create a figure with three subplots
    plt.figure(figsize=(18, 5))

    # Actual potential subplot
    plt.subplot(1, 3, 1)
    plt.imshow(actual.reshape(ny, nx), cmap='viridis', origin='lower')
    plt.colorbar(label='Potential φ')
    plt.title(f'Actual Numerical Potential (Grid {nx}x{ny})')
    plt.xlabel('Pipe Length')
    plt.ylabel('Pipe Height')

    # Predicted potential subplot
    plt.subplot(1, 3, 2)
    plt.imshow(predictions.reshape(ny, nx), cmap='viridis', origin='lower')
    plt.colorbar(label='Potential φ')
    plt.title(f'GNN Predicted Potential (Grid {nx}x{ny})')
    plt.xlabel('Pipe Length')
    plt.ylabel('Pipe Height')

    # Error subplot
    plt.subplot(1, 3, 3)
    plt.imshow(error.reshape(ny, nx), cmap='plasma', origin='lower')
    plt.colorbar(label='Absolute Error |Actual - Predicted|')
    plt.title('Prediction Error')
    plt.xlabel('Pipe Length')
    plt.ylabel('Pipe Height')

    plt.tight_layout()
    # Save the combined figure
    plt.savefig(f'pipe_flow_comparison_{nx}x{ny}.png')
    plt.show()

# Test different grid sizes
for nx, ny in [(10, 5), (20, 10), (30, 15)]:
    run_and_plot(nx, ny)
