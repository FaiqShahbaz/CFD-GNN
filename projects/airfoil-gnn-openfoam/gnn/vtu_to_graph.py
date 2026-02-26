#!/usr/bin/env python3
"""
Convert OpenFOAM VTU outputs to PyTorch Geometric graph objects.

Default repo layout (relative to this script location):
  <project_root>/data/vtu        (input)
  <project_root>/data/graphs     (output)

Typical usage:
  python vtu_to_graph.py --batch_process
  python vtu_to_graph.py --batch_process --edge_cutoff 0.2 --use_cells --bidirectional
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import glob
import meshio
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def project_root_from_this_file() -> Path:
    # expected: <project_root>/gnn/vtu_to_graph.py
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root_from_this_file()

    parser = argparse.ArgumentParser(description="Convert VTU files to PyTorch Geometric graphs")
    parser.add_argument(
        "--vtu_dir",
        default=str(root / "data" / "vtu"),
        help="Directory containing VTU files (default: <project_root>/data/vtu)",
    )
    parser.add_argument(
        "--out_dir",
        default=str(root / "data" / "graphs"),
        help="Output directory for graph.pt + metadata.json (default: <project_root>/data/graphs)",
    )
    parser.add_argument("--chord_length", type=float, default=1.0, help="Chord length for normalization")
    parser.add_argument("--reference_density", type=float, default=1.0, help="Reference density")
    parser.add_argument("--reference_viscosity", type=float, default=1.5e-5, help="Reference viscosity")
    parser.add_argument("--edge_cutoff", type=float, default=0.1, help="Maximum edge distance (normalized)")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--batch_process", action="store_true", help="Process all VTU files in vtu_dir")
    parser.add_argument("--vtu_pattern", default="**/internal.vtu", help="Pattern to find VTU files")
    parser.add_argument("--use_cells", action="store_true", help="Use cell centers as nodes instead of points")
    parser.add_argument("--bidirectional", action="store_true", help="Add reverse edges for bidirectional graph")
    return parser.parse_args()


def load_vtu_data(vtu_path: str, use_cells: bool = False):
    try:
        mesh = meshio.read(vtu_path)

        if use_cells:
            cells = None
            if "triangle" in mesh.cells_dict:
                cells = mesh.cells_dict["triangle"]
            elif "quad" in mesh.cells_dict:
                cells = mesh.cells_dict["quad"]
            elif len(mesh.cells) > 0:
                cells = mesh.cells[0].data

            if cells is not None:
                points = np.mean(mesh.points[cells], axis=1)[:, :2]
                point_data = {}
                if mesh.cell_data:
                    for key, data_list in mesh.cell_data.items():
                        if len(data_list) > 0:
                            point_data[key] = data_list[0]
            else:
                logger.warning("No cells found, falling back to points")
                points = mesh.points[:, :2]
                point_data = mesh.point_data if mesh.point_data else {}
        else:
            points = mesh.points[:, :2]
            point_data = mesh.point_data if mesh.point_data else {}

        cells = None
        if "triangle" in mesh.cells_dict:
            cells = mesh.cells_dict["triangle"]
        elif "quad" in mesh.cells_dict:
            cells = mesh.cells_dict["quad"]
        elif len(mesh.cells) > 0:
            cells = mesh.cells[0].data

        cell_data = mesh.cell_data if mesh.cell_data else {}

        return points, cells, point_data, cell_data

    except Exception as e:
        logger.error(f"Error loading VTU file {vtu_path}: {e}")
        return None, None, None, None


def detect_boundary_conditions(points: np.ndarray, cells, tolerance: float = 1e-6) -> np.ndarray:
    n_points = points.shape[0]
    bc_tags = np.zeros((n_points, 4))

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    inlet_mask = np.abs(points[:, 0] - x_min) < tolerance
    outlet_mask = np.abs(points[:, 0] - x_max) < tolerance
    wall_mask = (np.abs(points[:, 1] - y_min) < tolerance) | (np.abs(points[:, 1] - y_max) < tolerance)

    inlet_mask = inlet_mask & ~wall_mask
    outlet_mask = outlet_mask & ~wall_mask

    middle_x_min = x_min + 0.1 * (x_max - x_min)
    middle_x_max = x_max - 0.1 * (x_max - x_min)
    airfoil_mask = (
        (np.abs(points[:, 1]) < tolerance)
        & (points[:, 0] >= middle_x_min)
        & (points[:, 0] <= middle_x_max)
    )
    wall_mask = wall_mask | airfoil_mask

    interior_mask = ~(inlet_mask | outlet_mask | wall_mask)

    bc_tags[inlet_mask, 0] = 1.0
    bc_tags[outlet_mask, 1] = 1.0
    bc_tags[wall_mask, 2] = 1.0
    bc_tags[interior_mask, 3] = 1.0

    return bc_tags


def build_connectivity(points: np.ndarray, edge_cutoff: float = 0.1, bidirectional: bool = False):
    logger.info("Building graph connectivity...")

    tree = cKDTree(points)
    edge_list = []
    edge_distances = []
    edge_set = set()

    for i, point in enumerate(points):
        neighbors = tree.query_ball_point(point, edge_cutoff)
        for j in neighbors:
            if i != j and (i, j) not in edge_set:
                dist = np.linalg.norm(points[i] - points[j])
                edge_list.append([i, j])
                edge_distances.append(dist)
                edge_set.add((i, j))

                if bidirectional and (j, i) not in edge_set:
                    edge_list.append([j, i])
                    edge_distances.append(dist)
                    edge_set.add((j, i))

    if not edge_list:
        logger.warning("No edges found! Consider increasing edge_cutoff")
        edge_list = [[i, i + 1] for i in range(len(points) - 1)]
        edge_distances = [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]

        if bidirectional:
            reverse_edges = [[i + 1, i] for i in range(len(points) - 1)]
            reverse_distances = [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
            edge_list.extend(reverse_edges)
            edge_distances.extend(reverse_distances)

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    return edge_index, np.array(edge_distances)


def extract_node_features(
    points: np.ndarray,
    bc_tags: np.ndarray,
    U_inf_vec: np.ndarray,
    logRe: float,
    alpha: float,
    chord_length: float = 1.0,
) -> np.ndarray:
    coords_norm = points / chord_length
    n_points = points.shape[0]

    features = np.hstack(
        [
            coords_norm,
            bc_tags,
            np.tile(U_inf_vec, (n_points, 1)),
            np.full((n_points, 1), logRe),
            np.full((n_points, 1), alpha),
        ]
    )
    return features


def extract_edge_attributes(points: np.ndarray, edge_index: torch.Tensor, edge_distances: np.ndarray) -> np.ndarray:
    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]

    src_pos = points[src_nodes]
    tgt_pos = points[tgt_nodes]
    relative_pos = tgt_pos - src_pos

    distance_norm = edge_distances / (np.max(edge_distances) + 1e-8)

    edge_attr = np.column_stack([relative_pos[:, 0], relative_pos[:, 1], edge_distances, distance_norm])
    return edge_attr


def parse_case_parameters(case_name: str) -> Tuple[float, float]:
    try:
        parts = case_name.split("_")
        U = float(parts[0][1:])
        alpha_deg = float(parts[1][1:])
        return U, alpha_deg
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse case parameters from {case_name}: {e}")
        return 10.0, 0.0


def build_graph(vtu_path: str, out_dir: str, args: argparse.Namespace) -> bool:
    case_name = Path(vtu_path).parent.name
    logger.info(f"Processing case: {case_name}")

    U, alpha_deg = parse_case_parameters(case_name)
    alpha = math.radians(alpha_deg)

    Re = args.reference_density * U * args.chord_length / args.reference_viscosity
    logRe = float(np.log(Re))

    points, cells, point_data, cell_data = load_vtu_data(vtu_path, args.use_cells)
    if points is None:
        logger.error(f"Failed to load {vtu_path}")
        return False

    logger.info(f"Loaded mesh with {len(points)} {'cells' if args.use_cells else 'points'}")

    bc_tags = detect_boundary_conditions(points, cells)
    edge_index, edge_distances = build_connectivity(points, args.edge_cutoff, args.bidirectional)

    U_vec = np.array([np.cos(alpha), np.sin(alpha)], dtype=float)

    node_features = extract_node_features(points, bc_tags, U_vec, logRe, alpha, args.chord_length)
    edge_attr = extract_edge_attributes(points, edge_index, edge_distances)

    # Targets: normalized pressure and velocity components (p, u, v)
    if "p" in point_data and "U" in point_data:
        p_norm = point_data["p"] / (args.reference_density * U**2)
        U_norm = point_data["U"][:, :2] / U
        targets = np.column_stack([p_norm, U_norm[:, 0], U_norm[:, 1]])
    else:
        logger.warning(f"No solution data found in {vtu_path}")
        targets = np.zeros((len(points), 3), dtype=float)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(targets, dtype=torch.float32),
        pos=torch.tensor(points, dtype=torch.float32),
        case_params=torch.tensor([U, alpha_deg, logRe], dtype=torch.float32),
    )
    data.u_inf = float(U)
    data.alpha = float(alpha_deg)

    case_output_dir = Path(out_dir) / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = case_output_dir / "graph.pt"
    if graph_path.exists():
        logger.warning(f"Overwriting existing file: {graph_path}")
    torch.save(data, str(graph_path))

    metadata: Dict[str, Any] = {
        "case_name": case_name,
        "U": float(U),
        "alpha_deg": float(alpha_deg),
        "alpha_rad": float(alpha),
        "Re": float(Re),
        "logRe": float(logRe),
        "n_nodes": int(data.x.shape[0]),
        "n_edges": int(data.edge_index.shape[1]),
        "node_feature_dim": int(data.x.shape[1]),
        "edge_feature_dim": int(data.edge_attr.shape[1]),
        "target_dim": int(data.y.shape[1]),
        "processing_config": {
            "use_cells": bool(args.use_cells),
            "bidirectional": bool(args.bidirectional),
            "edge_cutoff": float(args.edge_cutoff),
            "chord_length": float(args.chord_length),
            "reference_density": float(args.reference_density),
            "reference_viscosity": float(args.reference_viscosity),
        },
    }

    metadata_path = case_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(f"Saved graph to {graph_path}")
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info(f"Graph stats - Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    return True


def process_single_case(args_tuple) -> bool:
    vtu_path, out_dir, args = args_tuple
    try:
        return build_graph(vtu_path, out_dir, args)
    except Exception as e:
        logger.error(f"Error processing {vtu_path}: {e}")
        return False


def batch_process(vtu_dir: str, out_dir: str, args: argparse.Namespace) -> None:
    vtu_pattern = str(Path(vtu_dir) / args.vtu_pattern)
    vtu_files = glob.glob(vtu_pattern, recursive=True)

    if not vtu_files:
        logger.error(f"No VTU files found matching pattern: {vtu_pattern}")
        return

    logger.info(f"Found {len(vtu_files)} VTU files to process")

    process_args = [(vtu_path, out_dir, args) for vtu_path in vtu_files]

    successful = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(process_single_case, arg) for arg in process_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting VTU files"):
            if future.result():
                successful += 1

    logger.info(f"Successfully processed {successful}/{len(vtu_files)} files")


def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.batch_process:
        batch_process(args.vtu_dir, args.out_dir, args)
    else:
        # In single-file mode, --vtu_dir is treated as a file path (preserve your original behavior)
        vtu_path = args.vtu_dir
        build_graph(vtu_path, args.out_dir, args)


if __name__ == "__main__":
    main()