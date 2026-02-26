#!/usr/bin/env python3
"""
Visualize AirfoilGNN predictions on the test split.

This script loads a trained model checkpoint, runs inference on the test dataset,
and generates:
  - Node scatter plots (true/pred/error)
  - Interpolated contour plots (true/pred/error)
  - Error histograms and correlation plots
  - Summary metrics saved to JSON + bar chart

Typical usage (run from inside the gnn/ folder):
  python visualize_results.py --model-path ../checkpoints/best_model.pt --data-dir normalized_graphs --output-dir evaluation_results
  python visualize_results.py --model-path ../checkpoints/best_model.pt --data-dir normalized_graphs --batch-size 64
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader as GeomDataLoader
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from train import AirfoilDataset, assign_node_targets
from models.airfoil_gnn import AirfoilGNN


def create_airfoil_model(config: Dict) -> AirfoilGNN:
    return AirfoilGNN(
        node_input_dim=config["node_input_dim"],
        edge_input_dim=config["edge_input_dim"],
        hidden_dim=config["hidden_dim"],
        num_processor_layers=config["num_processor_layers"],
        output_node_dim=config.get("output_node_dim", 3),
        output_global_dim=config.get("output_global_dim", 2),
        predict_global=config.get("predict_global", False),
        layer_type=config.get("layer_type", "custom"),
        dropout=config.get("dropout", 0.1),
    )


class ModelEvaluator:
    def __init__(self, model_path: str, data_dir: str, output_dir: str = "evaluation_results"):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self._make_dirs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_loader = None

    def _make_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["contour_plots", "node_scatter", "error_analysis", "statistics"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def run_evaluation(self, config_override: Optional[Dict] = None, batch_size: int = 32):
        self.load_model(config_override)
        self.load_test_data(batch_size)
        preds = self.predict_all()
        np.savez_compressed(self.output_dir / "predictions.npz", **preds)
        self._plot_node_scatter(preds)
        self._plot_contours(preds)
        self._plot_error_analysis(preds)
        self._plot_statistics(preds)
        print(f"Results saved to {self.output_dir}")

    def load_model(self, config_override: Optional[Dict] = None):
        ckpt = torch.load(self.model_path, map_location=self.device)
        sample = AirfoilDataset(self.data_dir, self.data_dir / "split.json", "test")[0]
        cfg = {
            "node_input_dim": sample.x.size(1),
            "edge_input_dim": sample.edge_attr.size(1),
            "hidden_dim": 128,
            "num_processor_layers": 4,
            "output_node_dim": 3,
            "output_global_dim": 2,
            "predict_global": sample.y.size(1) >= 5,
            "layer_type": "custom",
            "dropout": 0.1,
        }
        if config_override:
            cfg.update(config_override)

        self.model = create_airfoil_model(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def load_test_data(self, batch_size: int = 32):
        ds = AirfoilDataset(self.data_dir, self.data_dir / "split.json", "test")
        self.test_loader = GeomDataLoader(ds, batch_size=batch_size, shuffle=False)

    def predict_all(self) -> Dict[str, np.ndarray]:
        accum = {k: [] for k in ["pressure_pred", "pressure_true", "velocity_pred", "velocity_true", "coords", "counts"]}
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Predicting"):
                batch = batch.to(self.device)
                assign_node_targets(batch)
                out = self.model(batch)
                pred = out["node_pred"] if isinstance(out, dict) else out

                p = pred[:, 0].cpu().numpy()
                v = pred[:, 1:3].cpu().numpy()

                accum["pressure_pred"].append(p)
                accum["pressure_true"].append(batch.pressure.cpu().numpy().flatten())
                accum["velocity_pred"].append(v)
                accum["velocity_true"].append(batch.velocity.cpu().numpy())

                coords = batch.x[:, :2].cpu().numpy()
                counts = batch.batch.bincount().cpu().numpy()
                accum["coords"].append(coords)
                accum["counts"].append(counts)

        for k in accum:
            accum[k] = np.concatenate(accum[k], axis=0)
        return accum

    def _grid(self, coords, vals, res=200):
        x, y = coords[:, 0], coords[:, 1]
        pad = 0.05
        xm, xM = x.min(), x.max()
        ym, yM = y.min(), y.max()
        xi = np.linspace(xm - pad * (xM - xm), xM + pad * (xM - xm), res)
        yi = np.linspace(ym - pad * (yM - ym), yM + pad * (yM - ym), res)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), vals, (X, Y), method="linear", fill_value=np.nan)
        return X, Y, Z

    def _save_contour(self, ax, coords, vals, title, cmap, vmin=None, vmax=None):
        X, Y, Z = self._grid(coords, vals)
        cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(title)
        return cf

    def _plot_node_scatter(self, preds):
        coords, counts = preds["coords"], preds["counts"]
        offs = np.r_[0, np.cumsum(counts)]
        base = self.output_dir / "node_scatter"

        for i in range(min(len(counts), 3)):
            s, e = offs[i], offs[i + 1]
            c = coords[s:e]
            p_t = preds["pressure_true"][s:e]
            p_p = preds["pressure_pred"][s:e]
            v_t = np.linalg.norm(preds["velocity_true"][s:e], axis=1)
            v_p = np.linalg.norm(preds["velocity_pred"][s:e], axis=1)
            e_p = np.abs(p_p - p_t)
            e_v = np.abs(v_p - v_t)
            e_pn = e_p / (np.max(e_p) + 1e-8)
            e_vn = e_v / (np.max(e_v) + 1e-8)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            items = [
                (p_t, "True Pressure", "RdYlBu_r", "Pressure"),
                (p_p, "Predicted Pressure", "RdYlBu_r", "Pressure"),
                (e_pn, "Pressure Error (norm)", "hot", "Normalized Error"),
                (v_t, "True Velocity Mag", "viridis", "Velocity"),
                (v_p, "Pred Velocity Mag", "viridis", "Velocity"),
                (e_vn, "Velocity Error (norm)", "hot", "Normalized Error"),
            ]

            for ax, (data, title, cmap, label) in zip(axes.flatten(), items):
                scat = ax.scatter(
                    c[:, 0],
                    c[:, 1],
                    c=data,
                    s=5,
                    cmap=cmap,
                    vmin=0 if "Error" in title else None,
                    vmax=1 if "Error" in title else None,
                )
                ax.set_title(title)
                ax.axis("off")
                cb = plt.colorbar(scat, ax=ax)
                cb.set_label(label)

            plt.suptitle(f"Node Scatter Sample {i}", fontsize=16)
            plt.tight_layout()
            fig.savefig(base / f"node_scatter_{i}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _plot_contours(self, preds):
        coords, counts = preds["coords"], preds["counts"]
        offs = np.r_[0, np.cumsum(counts)]
        base = self.output_dir / "contour_plots"

        for i in range(min(len(counts), 3)):
            s, e = offs[i], offs[i + 1]
            c = coords[s:e]
            p_t = preds["pressure_true"][s:e]
            p_p = preds["pressure_pred"][s:e]
            v_t = np.linalg.norm(preds["velocity_true"][s:e], axis=1)
            v_p = np.linalg.norm(preds["velocity_pred"][s:e], axis=1)
            e_p = np.abs(p_p - p_t)
            e_v = np.abs(v_p - v_t)
            e_pn = e_p / (np.max(e_p) + 1e-8)
            e_vn = e_v / (np.max(e_v) + 1e-8)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            self._save_contour(axes[0, 0], c, p_t, "True Pressure", "RdYlBu_r")
            self._save_contour(axes[0, 1], c, p_p, "Predicted Pressure", "RdYlBu_r")
            self._save_contour(axes[0, 2], c, e_pn, "Pressure Error (norm)", "hot", 0, 1)
            self._save_contour(axes[1, 0], c, v_t, "True Velocity Mag", "viridis")
            self._save_contour(axes[1, 1], c, v_p, "Predicted Velocity Mag", "viridis")
            self._save_contour(axes[1, 2], c, e_vn, "Velocity Error (norm)", "hot", 0, 1)

            for ax in axes.flatten():
                ax.axis("off")

            plt.suptitle(f"Interpolated Contours Sample {i}", fontsize=16)
            plt.tight_layout()
            fig.savefig(base / f"contour_plot_{i}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _plot_error_analysis(self, preds):
        p_t, p_p = preds["pressure_true"], preds["pressure_pred"]
        v_t, v_p = preds["velocity_true"], preds["velocity_pred"]
        err_p = p_p - p_t
        rel_p = err_p / (np.abs(p_t) + 1e-8) * 100
        mag_t = np.linalg.norm(v_t, axis=1)
        mag_p = np.linalg.norm(v_p, axis=1)
        err_v = mag_p - mag_t
        rel_v = err_v / (mag_t + 1e-8) * 100

        base = self.output_dir / "error_analysis"
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].hist(err_p, bins=50, density=True)
        axes[0, 0].set_title("Pressure Absolute Error")
        axes[0, 0].set_xlabel("Error")
        axes[0, 1].hist(rel_p, bins=50, density=True)
        axes[0, 1].set_title("Pressure Relative Error (%)")
        axes[0, 1].set_xlabel("Rel Error (%)")
        axes[1, 0].hist(err_v, bins=50, density=True)
        axes[1, 0].set_title("Velocity Absolute Error")
        axes[1, 0].set_xlabel("Error")
        axes[1, 1].hist(rel_v, bins=50, density=True)
        axes[1, 1].set_title("Velocity Relative Error (%)")
        axes[1, 1].set_xlabel("Rel Error (%)")
        plt.tight_layout()
        fig.savefig(base / "hist_errors.png", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(p_t, p_p, s=1)
        axes[0].plot([p_t.min(), p_t.max()], [p_t.min(), p_t.max()], "r--")
        axes[0].set_title("Pressure True vs Pred")
        axes[0].set_xlabel("True")
        axes[0].set_ylabel("Pred")
        axes[1].scatter(mag_t, mag_p, s=1)
        axes[1].plot([mag_t.min(), mag_t.max()], [mag_t.min(), mag_t.max()], "r--")
        axes[1].set_title("Velocity True vs Pred")
        axes[1].set_xlabel("True")
        axes[1].set_ylabel("Pred")
        plt.tight_layout()
        fig.savefig(base / "corr_errors.png", dpi=300)
        plt.close(fig)

    def _plot_statistics(self, preds):
        p_t, p_p = preds["pressure_true"], preds["pressure_pred"]
        v_t, v_p = preds["velocity_true"], preds["velocity_pred"]
        mag_t = np.linalg.norm(v_t, axis=1)
        mag_p = np.linalg.norm(v_p, axis=1)

        metrics = {
            "MSE_P": mean_squared_error(p_t, p_p),
            "RMSE_P": np.sqrt(mean_squared_error(p_t, p_p)),
            "MAE_P": mean_absolute_error(p_t, p_p),
            "R2_P": r2_score(p_t, p_p),
            "MSE_V": mean_squared_error(mag_t, mag_p),
            "R2_V": r2_score(mag_t, mag_p),
        }

        stat_dir = self.output_dir / "statistics"
        stat_dir.mkdir(exist_ok=True)

        with open(stat_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(metrics.keys(), metrics.values())
        ax.set_title("Overall Metrics")
        ax.set_ylabel("Value")
        ax.set_xticklabels(metrics.keys(), rotation=45, ha="right")

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + max(metrics.values()) * 0.01,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        fig.savefig(stat_dir / "metrics.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="normalized_graphs")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path, args.data_dir, args.output_dir)
    evaluator.run_evaluation(batch_size=args.batch_size)