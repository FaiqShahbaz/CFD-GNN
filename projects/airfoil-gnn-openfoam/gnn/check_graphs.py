#!/usr/bin/env python3
"""
Quick sanity checks + visualizations for generated graph samples.

Checks:
  - graph.pt exists and loads
  - x, y, edge_index, edge_attr dimensions look correct
  - metadata.json exists

Examples:
  python check_graphs.py --graphs-root graphs --n-random 3 --save-plots
  python check_graphs.py --graphs-root graphs --n-random 1 --show
  python check_graphs.py --graphs-root normalized_graphs --n-random 0
"""

import os
import json
import torch
import random
from glob import glob
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def summarize_metadata(metadata_path: Path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    return {
        "case": meta.get("case_name"),
        "U": meta.get("U"),
        "alpha": meta.get("alpha_deg"),
        "Re": meta.get("Re"),
        "nodes": meta.get("n_nodes"),
        "edges": meta.get("n_edges"),
        "x_dim": meta.get("node_feature_dim"),
        "y_dim": meta.get("target_dim"),
        "edge_dim": meta.get("edge_feature_dim"),
    }


def load_graph(graph_path: Path):
    return torch.load(graph_path, map_location="cpu", weights_only=False)


def load_and_check_graph(graph_path: Path):
    data = load_graph(graph_path)

    checks = {
        "has_x": hasattr(data, "x") and data.x is not None and data.x.shape[0] > 0,
        "has_y": hasattr(data, "y") and data.y is not None and data.y.shape[1] == 3,
        "has_edges": hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.shape[1] > 0,
        "edge_attr_dim": hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[1] == 4,
        "pos_match": hasattr(data, "pos") and data.pos is not None and data.pos.shape[0] == data.x.shape[0],
    }

    return data, checks


def plot_graph(data, title="Graph", mode="structure", save_path: Path | None = None, show: bool = False):
    pos = data.pos.cpu().numpy()
    edge_index = data.edge_index.cpu()

    plt.figure(figsize=(8, 6))

    if mode == "structure":
        for i, j in edge_index.T.tolist():
            plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="gray", linewidth=0.2)
        plt.scatter(pos[:, 0], pos[:, 1], s=1)
        plt.title(f"{title} - Graph Structure")

    elif mode == "pressure":
        pressure = data.y[:, 0].cpu().numpy()
        plt.scatter(pos[:, 0], pos[:, 1], c=pressure, cmap="coolwarm", s=5)
        plt.colorbar(label="Normalized Pressure")
        plt.title(f"{title} - Pressure Field")

    elif mode == "velocity":
        u = data.y[:, 1].cpu().numpy()
        v = data.y[:, 2].cpu().numpy()
        plt.quiver(pos[:, 0], pos[:, 1], u, v, scale=50, width=0.001)
        plt.title(f"{title} - Velocity Field")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def verify_all_graphs(graphs_root: str, n_random: int = 3, save_plots: bool = False, show: bool = False):
    graphs_root = graphs_root.rstrip("/")

    graph_dirs = sorted(glob(os.path.join(graphs_root, "U*_A*")))
    print(f"Found {len(graph_dirs)} graph folders under '{graphs_root}'.")

    summaries = []
    failures = []

    for d in graph_dirs:
        dpath = Path(d)
        metadata_path = dpath / "metadata.json"
        graph_path = dpath / "graph.pt"

        if not metadata_path.exists() or not graph_path.exists():
            failures.append((str(dpath), "missing files"))
            continue

        summary = summarize_metadata(metadata_path)
        _, checks = load_and_check_graph(graph_path)

        if not all(checks.values()):
            failures.append((str(dpath), "failed checks", checks))

        summaries.append(summary)

    print("\n=== Summary Report ===")
    for s in summaries:
        print(
            f"{s['case']}: U={s['U']}, alpha={s['alpha']}, Re={int(s['Re']) if s['Re'] else 'NA'}, "
            f"Nodes={s['nodes']}, Edges={s['edges']}, x_dim={s['x_dim']}, y_dim={s['y_dim']}, edge_dim={s['edge_dim']}"
        )

    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(f)
    else:
        print("\nAll graphs passed basic checks.")

    if n_random > 0 and len(graph_dirs) > 0:
        print("\nPlotting random samples...")
        sampled_dirs = random.sample(graph_dirs, min(n_random, len(graph_dirs)))

        for d in sampled_dirs:
            dpath = Path(d)
            case_name = dpath.name
            data = load_graph(dpath / "graph.pt")

            out_dir = Path("plots") / case_name if save_plots else None

            plot_graph(
                data,
                title=case_name,
                mode="structure",
                save_path=(out_dir / "structure.png") if out_dir else None,
                show=show,
            )
            plot_graph(
                data,
                title=case_name,
                mode="pressure",
                save_path=(out_dir / "pressure.png") if out_dir else None,
                show=show,
            )
            plot_graph(
                data,
                title=case_name,
                mode="velocity",
                save_path=(out_dir / "velocity.png") if out_dir else None,
                show=show,
            )


def main():
    parser = argparse.ArgumentParser(description="Check and visualize generated graphs.")
    parser.add_argument("--graphs-root", type=str, default="graphs", help="Folder containing U*_A*/graph.pt folders")
    parser.add_argument("--n-random", type=int, default=3, help="Number of random cases to plot")
    parser.add_argument("--save-plots", action="store_true", help="Save plots into ./plots/<case>/")
    parser.add_argument("--show", action="store_true", help="Display plots interactively (may not work headless)")
    args = parser.parse_args()

    verify_all_graphs(args.graphs_root, args.n_random, args.save_plots, args.show)


if __name__ == "__main__":
    main()