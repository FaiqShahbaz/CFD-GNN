#!/usr/bin/env python3
"""
Evaluate a trained AirfoilGNN on the test split and generate field plots.

Expected layout (relative to this script location):
  <project_root>/data/normalized_graphs/   (graphs + split.json)
  <project_root>/evaluation_results/      (outputs)

Typical usage:
  python evaluate.py --model-path checkpoints/best_model.pt --data-dir data/normalized_graphs
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader as GeomDataLoader

from train import AirfoilDataset, assign_node_targets   # <-- rename if your file isn't train.py
from models.airfoil_gnn import AirfoilGNN


def create_airfoil_model(config):
    return AirfoilGNN(
        node_input_dim=config['node_input_dim'],
        edge_input_dim=config['edge_input_dim'],
        hidden_dim=config['hidden_dim'],
        num_processor_layers=config['num_processor_layers'],
        output_node_dim=config.get('output_node_dim', 3),
        output_global_dim=config.get('output_global_dim', 2),
        predict_global=config.get('predict_global', False),
        layer_type=config.get('layer_type', 'custom'),
        dropout=config.get('dropout', 0.1),
    )


class ModelEvaluator:
    def __init__(self, model_path: str, data_dir: str, output_dir: str = "evaluation_results"):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_loader = None
        self.has_global_features = False

    def run_evaluation(self, model_config_override: Optional[Dict] = None, batch_size: int = 32):
        print("Starting model evaluation...")
        self.load_model(model_config_override)
        self.load_test_data(batch_size)
        predictions = self.predict_all_samples()
        np.savez_compressed(self.output_dir / "predictions.npz", **predictions)
        self.visualize_field_predictions(predictions)
        print(f"Evaluation complete. Results saved to: {self.output_dir}")

    def load_model(self, config_override: Optional[Dict] = None):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        split_path = self.data_dir / "split.json"
        sample_data = AirfoilDataset(self.data_dir, split_path, "test")[0]

        model_config = {
            'node_input_dim': sample_data.x.size(1),
            'edge_input_dim': sample_data.edge_attr.size(1),
            'hidden_dim': 128,
            'num_processor_layers': 4,
            'output_node_dim': 3,
            'output_global_dim': 2,
            'predict_global': sample_data.y.size(1) >= 5,
            'layer_type': 'custom',
            'dropout': 0.1
        }
        if config_override:
            model_config.update(config_override)

        self.has_global_features = model_config["predict_global"]
        self.model = create_airfoil_model(model_config).to(self.device)

        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def load_test_data(self, batch_size: int = 32):
        split_path = self.data_dir / "split.json"
        dataset = AirfoilDataset(self.data_dir, split_path, "test")
        self.test_loader = GeomDataLoader(dataset, batch_size=batch_size, shuffle=False)

    def predict_all_samples(self) -> Dict[str, np.ndarray]:
        predictions = {
            'pressure_pred': [], 'pressure_true': [],
            'velocity_pred': [], 'velocity_true': [],
            'node_coords': [], 'node_counts': []
        }

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(self.device)
                assign_node_targets(batch)

                output = self.model(batch)
                node_pred = output['node_pred'] if isinstance(output, dict) else output

                predictions['pressure_pred'].append(node_pred[:, 0].cpu().numpy())
                predictions['pressure_true'].append(batch.pressure.cpu().numpy().flatten())
                predictions['velocity_pred'].append(node_pred[:, 1:3].cpu().numpy())
                predictions['velocity_true'].append(batch.velocity.cpu().numpy())
                predictions['node_coords'].append(batch.x[:, :2].cpu().numpy())
                predictions['node_counts'].append(batch.batch.bincount().cpu().numpy())

        predictions['pressure_pred'] = np.concatenate(predictions['pressure_pred'], axis=0)
        predictions['pressure_true'] = np.concatenate(predictions['pressure_true'], axis=0)
        predictions['velocity_pred'] = np.concatenate(predictions['velocity_pred'], axis=0)
        predictions['velocity_true'] = np.concatenate(predictions['velocity_true'], axis=0)
        predictions['node_coords'] = np.concatenate(predictions['node_coords'], axis=0)
        predictions['node_counts'] = np.concatenate(predictions['node_counts'], axis=0)
        return predictions

    def visualize_field_predictions(self, predictions: Dict[str, np.ndarray], max_samples: int = 5):
        print("Creating node-level field visualizations...")

        node_coords = predictions['node_coords']
        pressure_pred = predictions['pressure_pred']
        pressure_true = predictions['pressure_true']
        velocity_pred = predictions['velocity_pred']
        velocity_true = predictions['velocity_true']

        node_counts = predictions['node_counts']
        offsets = np.concatenate([[0], np.cumsum(node_counts)])
        total_samples = min(len(node_counts), max_samples)

        for i in range(total_samples):
            start, end = offsets[i], offsets[i + 1]
            coords = node_coords[start:end]
            x, y = coords[:, 0], coords[:, 1]

            u_true, v_true = velocity_true[start:end, 0], velocity_true[start:end, 1]
            u_pred, v_pred = velocity_pred[start:end, 0], velocity_pred[start:end, 1]

            vmag_true = np.linalg.norm(np.stack([u_true, v_true], axis=1), axis=1)
            vmag_pred = np.linalg.norm(np.stack([u_pred, v_pred], axis=1), axis=1)

            p_error = np.abs(pressure_pred[start:end] - pressure_true[start:end])
            vmax_p = np.max(np.abs(np.concatenate([pressure_true[start:end], pressure_pred[start:end]])))
            vmax_e = np.max(p_error)

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axs[0].scatter(x, y, c=pressure_true[start:end], cmap='viridis', s=10, vmin=-vmax_p, vmax=vmax_p)
            axs[0].set_title(f"True Pressure (Sample {i})")
            axs[0].set_aspect('equal')
            plt.colorbar(im0, ax=axs[0])

            im1 = axs[1].scatter(x, y, c=pressure_pred[start:end], cmap='viridis', s=10, vmin=-vmax_p, vmax=vmax_p)
            axs[1].set_title(f"Predicted Pressure (Sample {i})")
            axs[1].set_aspect('equal')
            plt.colorbar(im1, ax=axs[1])

            im2 = axs[2].scatter(x, y, c=p_error, cmap='hot', s=10, vmin=0, vmax=vmax_e)
            axs[2].set_title(f"Absolute Error |P_pred - P_true| (Sample {i})")
            axs[2].set_aspect('equal')
            plt.colorbar(im2, ax=axs[2])

            plt.suptitle(f"Pressure Comparison - Sample {i}")
            plt.subplots_adjust(wspace=0.3)
            plt.savefig(self.output_dir / f"pressure_comparison_sample_{i}.png", dpi=300)
            plt.close()

            vmag_error = np.abs(vmag_pred - vmag_true)
            vmax_vm = np.max(np.abs(np.concatenate([vmag_true, vmag_pred])))
            vmax_ve = np.max(vmag_error)

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            im3 = axs[0].scatter(x, y, c=vmag_true, cmap='coolwarm', s=10, vmin=0, vmax=vmax_vm)
            axs[0].set_title(f"True Velocity Magnitude (Sample {i})")
            axs[0].set_aspect('equal')
            plt.colorbar(im3, ax=axs[0])

            im4 = axs[1].scatter(x, y, c=vmag_pred, cmap='coolwarm', s=10, vmin=0, vmax=vmax_vm)
            axs[1].set_title(f"Predicted Velocity Magnitude (Sample {i})")
            axs[1].set_aspect('equal')
            plt.colorbar(im4, ax=axs[1])

            im5 = axs[2].scatter(x, y, c=vmag_error, cmap='hot', s=10, vmin=0, vmax=vmax_ve)
            axs[2].set_title(f"Absolute Error |Vmag_pred - Vmag_true| (Sample {i})")
            axs[2].set_aspect('equal')
            plt.colorbar(im5, ax=axs[2])

            plt.suptitle(f"Velocity Magnitude Comparison - Sample {i}")
            plt.subplots_adjust(wspace=0.3)
            plt.savefig(self.output_dir / f"velocity_magnitude_comparison_sample_{i}.png", dpi=300)
            plt.close()

            with open(self.output_dir / f"sample_{i}_meta.txt", 'w') as f:
                wind_velocity = np.mean(vmag_true)
                angle_of_attack = np.arctan2(np.mean(v_true), np.mean(u_true)) * 180 / np.pi
                f.write(f"Sample {i} info:\n")
                f.write(f"Mean wind velocity magnitude: {wind_velocity:.4f} m/s\n")
                f.write(f"Estimated angle of attack: {angle_of_attack:.2f} degrees\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='normalized_graphs')
    parser.add_argument('--output-dir', type=str, default='evaluation_results')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path, args.data_dir, args.output_dir)
    evaluator.run_evaluation(model_config_override={'num_processor_layers': 4}, batch_size=args.batch_size)


if __name__ == "__main__":
    import argparse
    main()