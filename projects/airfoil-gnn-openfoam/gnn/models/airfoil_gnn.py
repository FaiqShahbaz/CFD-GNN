#!/usr/bin/env python3
"""
MeshGraphNet-style Graph Neural Network for airfoil flow-field prediction.

This model predicts:
  - Node-level quantities: pressure (p), velocity components (u, v)
  - Optional global quantities: lift (CL) and drag (CD)

Expected graph format (from vtu_to_graph.py):
  Node features (10D):
    [x, y, inlet, outlet, wall, interior, U_inf_x, U_inf_y, log_Re, alpha_rad]

  Edge features (4D):
    [dx, dy, distance, normalized_distance]

Typical usage:
  from models.airfoil_gnn import AirfoilGNN
  model = AirfoilGNN()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GINConv, EdgeConv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import numpy as np
from typing import Dict, Any, Optional, Union


class MLP(nn.Module):
    """Multi-Layer Perceptron with BatchNorm and Dropout"""
    def __init__(self, in_dim, out_dim, hidden_dims=[128, 64],
                 dropout=0.1, use_bn=True, activation='relu'):
        super().__init__()

        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i+1]))

            # Add BatchNorm and Activation (except for last layer)
            if i < len(dims) - 2:
                if use_bn:
                    layers.append(BatchNorm1d(dims[i+1]))

                if activation == 'relu':
                    layers.append(ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'swish':
                    layers.append(nn.SiLU())

                if dropout > 0:
                    layers.append(Dropout(dropout))

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EdgeProcessor(MessagePassing):
    """Custom Edge-based Message Passing Layer"""
    def __init__(self, node_dim, edge_dim, hidden_dim, aggr='add'):
        super().__init__(aggr=aggr)

        # Edge network processes both node and edge features
        self.edge_mlp = MLP(2 * node_dim + edge_dim, hidden_dim,
                           hidden_dims=[hidden_dim, hidden_dim//2])

        # Node update network
        self.node_mlp = MLP(node_dim + hidden_dim, node_dim,
                           hidden_dims=[hidden_dim, hidden_dim//2])

        # Edge update network
        self.edge_update_mlp = MLP(edge_dim + hidden_dim, edge_dim,
                                  hidden_dims=[hidden_dim//2, hidden_dim//4])

    def forward(self, x, edge_index, edge_attr):

        # 1. Update edge features first based on current node features
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_messages = self.edge_mlp(edge_input)
        edge_attr_updated = self.edge_update_mlp(
            torch.cat([edge_attr, edge_messages], dim=1)
        )

        # 2. Then propagate messages using updated edge features
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated)

        # 3. Update node features
        x_updated = self.node_mlp(torch.cat([x, out], dim=1))

        return x_updated, edge_attr_updated

    def message(self, x_i, x_j, edge_attr):
        # Create messages from neighboring nodes and edges
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(msg_input)


class GNNProcessor(nn.Module):
    """Multi-layer GNN Processor (MeshGraphNet style)"""
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=6,
                 layer_type='custom'):
        super().__init__()

        self.num_layers = num_layers
        self.layer_type = layer_type

        if layer_type == 'custom':
            self.layers = nn.ModuleList([
                EdgeProcessor(node_dim, edge_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        elif layer_type == 'gin':
            self.layers = nn.ModuleList([
                GINConv(MLP(node_dim, node_dim, [hidden_dim]))
                for _ in range(num_layers)
            ])
        elif layer_type == 'edge_conv':
            self.layers = nn.ModuleList([
                EdgeConv(MLP(2 * node_dim, node_dim, [hidden_dim]))
                for _ in range(num_layers)
            ])

        # Residual connections
        self.use_residual = True

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            x_in = x

            if self.layer_type == 'custom':
                x, edge_attr = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

            # Residual connection
            if self.use_residual and i > 0:
                x = x + x_in

        return x, edge_attr


class AirfoilGNN(nn.Module):
    """
    MeshGraphNet-style GNN for Airfoil Flow Prediction

    Architecture:
    Input Graph -> Node/Edge Encoders -> GNN Processor -> Decoders -> Outputs
    """
    def __init__(self,
                 node_input_dim=10,     # [x, y, inlet, outlet, wall, interior, U_inf_x, U_inf_y, log_Re, alpha_rad]
                 edge_input_dim=4,      # [dx, dy, distance, normalized_distance]
                 hidden_dim=128,
                 num_processor_layers=6,
                 output_node_dim=3,     # [p, u, v]
                 output_global_dim=2,   # [CL, CD]
                 predict_global=True,
                 layer_type='custom',
                 dropout=0.1,
                 pool_dropout=0.1):

        super().__init__()

        self.predict_global = predict_global
        self.hidden_dim = hidden_dim
        self.pool_dropout = pool_dropout

        # === ENCODERS ===
        self.node_encoder = MLP(
            node_input_dim, hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim//2],
            dropout=dropout
        )

        self.edge_encoder = MLP(
            edge_input_dim, hidden_dim//2,
            hidden_dims=[hidden_dim//2, hidden_dim//4],
            dropout=dropout
        )

        # === PROCESSOR (Core GNN) ===
        self.processor = GNNProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim//2,
            hidden_dim=hidden_dim,
            num_layers=num_processor_layers,
            layer_type=layer_type
        )

        # === DECODERS ===
        # Node-level decoder (pressure, velocity)
        self.node_decoder = MLP(
            hidden_dim, output_node_dim,
            hidden_dims=[hidden_dim//2, hidden_dim//4],
            dropout=dropout
        )

        # Global decoder (lift/drag coefficients)
        if predict_global:
            self.global_pooling_dropout = Dropout(self.pool_dropout)
            self.global_layer_norm = nn.LayerNorm(hidden_dim * 2)
            self.global_decoder = MLP(
                hidden_dim * 2,  # mean + max pooling
                output_global_dim,
                hidden_dims=[hidden_dim, hidden_dim//2],
                dropout=dropout
            )

    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [N, 10]
                - edge_index: edge connectivity [2, E]
                - edge_attr: edge features [E, 4]
                - batch: batch assignment (for batched graphs)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'node_pred': Node-level predictions [N, 3] for [pressure, u_velocity, v_velocity]
                - 'global_pred': Global predictions [B, 2] for [CL, CD] (if predict_global=True)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)

        # === ENCODING ===
        x_encoded = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)

        # === PROCESSING ===
        x_processed, edge_attr_processed = self.processor(
            x_encoded, edge_index, edge_attr_encoded
        )

        # === DECODING ===
        node_pred = self.node_decoder(x_processed)

        outputs = {'node_pred': node_pred}

        if self.predict_global:
            if batch is not None:
                x_mean = global_mean_pool(x_processed, batch)
                x_max = global_max_pool(x_processed, batch)
            else:
                x_mean = x_processed.mean(dim=0, keepdim=True)
                x_max = x_processed.max(dim=0, keepdim=True)[0]

            global_features = torch.cat([x_mean, x_max], dim=1)

            if self.training:
                global_features = self.global_pooling_dropout(global_features)

            global_features = self.global_layer_norm(global_features)
            global_pred = self.global_decoder(global_features)
            outputs['global_pred'] = global_pred

        return outputs


# === LOSS FUNCTIONS ===
class AirfoilLoss(nn.Module):
    """Custom loss function for airfoil flow prediction with flexible target handling"""
    def __init__(self, node_weight=1.0, global_weight=0.1,
                 pressure_weight=1.0, velocity_weight=1.0):
        super().__init__()
        self.node_weight = node_weight
        self.global_weight = global_weight
        self.pressure_weight = pressure_weight
        self.velocity_weight = velocity_weight

    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss with flexible target handling

        Args:
            pred: Dictionary with 'node_pred' [N, 3] and optionally 'global_pred' [B, 2]
            target: Dictionary with 'node_target' [N, 3] and optionally 'global_target' [B, 2]

        Returns:
            Dict[str, torch.Tensor]: Dictionary of losses
        """
        losses = {}
        total_loss = 0

        if 'node_pred' in pred and 'node_target' in target:
            node_pred = pred['node_pred']
            node_target = target['node_target']

            p_loss = F.mse_loss(node_pred[:, 0], node_target[:, 0])
            uv_loss = F.mse_loss(node_pred[:, 1:], node_target[:, 1:])

            node_loss = self.pressure_weight * p_loss + self.velocity_weight * uv_loss
            losses['node_loss'] = node_loss
            losses['pressure_loss'] = p_loss
            losses['velocity_loss'] = uv_loss

            total_loss += self.node_weight * node_loss

        if ('global_pred' in pred and 'global_target' in target and
            target['global_target'] is not None and
            target['global_target'].numel() > 0):

            global_loss = F.mse_loss(pred['global_pred'], target['global_target'])
            losses['global_loss'] = global_loss
            total_loss += self.global_weight * global_loss

        losses['total_loss'] = total_loss
        return losses


# === MODEL INITIALIZATION HELPER ===
def create_airfoil_model(config: Optional[Dict[str, Any]] = None) -> AirfoilGNN:
    """
    Create AirfoilGNN model with sensible defaults

    Args:
        config: Optional configuration dictionary to override defaults

    Returns:
        AirfoilGNN: Initialized model
    """

    default_config = {
        'node_input_dim': 10,
        'edge_input_dim': 4,
        'hidden_dim': 128,
        'num_processor_layers': 6,
        'output_node_dim': 3,
        'output_global_dim': 2,
        'predict_global': True,
        'layer_type': 'custom',
        'dropout': 0.1,
        'pool_dropout': 0.1
    }

    if config:
        default_config.update(config)

    model = AirfoilGNN(**default_config)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)

    return model


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    model = create_airfoil_model()

    from torch_geometric.data import Data

    num_nodes = 1000
    num_edges = 3000

    x = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    with torch.no_grad():
        output = model(data)

    print("Model created successfully!")
    print(f"Node predictions shape: {output['node_pred'].shape}")
    if 'global_pred' in output:
        print(f"Global predictions shape: {output['global_pred'].shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")