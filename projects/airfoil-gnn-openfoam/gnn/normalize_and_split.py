#!/usr/bin/env python3
"""
Normalize graphs and create train/val/test split.

Default repo layout (relative to this script location):
  <project_root>/data/graphs             (input)
  <project_root>/data/normalized_graphs  (output)

Outputs written to out_dir:
  - split.json
  - norm_stats.json
  - normalized graph.pt files (mirroring input structure if out_dir differs)
"""

import os
import json
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Minimal addition: compute project root for safe defaults ---
def _project_root() -> Path:
    # expected location: <project_root>/gnn/<this_file>.py
    return Path(__file__).resolve().parents[1]

class DatasetNormalizer:
    """Enhanced dataset normalizer for GNN airfoil flow prediction."""
    
    def __init__(self, graph_dir: str = 'graphs', out_dir: Optional[str] = None, seed: int = 42):
        self.graph_dir = Path(graph_dir)
        self.out_dir = Path(out_dir) if out_dir else self.graph_dir
        self.seed = seed
        self.set_seed()
        
        # Create output directory if it doesn't exist
        if self.out_dir != self.graph_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.out_dir}")
        
    def set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        import random
        random.seed(self.seed)
        
    def find_graph_files(self) -> List[Path]:
        """Find all graph.pt files with improved pattern matching."""
        patterns = [
            'U*_A*/graph.pt',  # Original pattern
            '**/graph.pt',     # Recursive search
            'graph_*.pt'       # Alternative naming
        ]
        
        graph_paths = []
        for pattern in patterns:
            found = list(self.graph_dir.glob(pattern))
            graph_paths.extend(found)
            
        # Remove duplicates and sort
        graph_paths = sorted(list(set(graph_paths)))
        
        if len(graph_paths) == 0:
            raise FileNotFoundError(f"No graph files found in {self.graph_dir}")
            
        logger.info(f"Found {len(graph_paths)} graph files")
        return graph_paths
        
    def create_stratified_split(self, graph_paths: List[Path], 
                              train_ratio: float = 0.7, 
                              val_ratio: float = 0.15) -> Dict[str, List[Path]]:
        """Create stratified split based on flow conditions if possible."""
        test_ratio = 1.0 - train_ratio - val_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Try to extract flow conditions for stratification
        try:
            conditions = []
            for path in graph_paths:
                # Extract conditions from path (e.g., U5.0_A10.0)
                parts = path.parent.name.split('_')
                if len(parts) >= 2:
                    u_val = float(parts[0][1:])  # Remove 'U' prefix
                    a_val = float(parts[1][1:])  # Remove 'A' prefix
                    conditions.append((u_val, a_val))
                else:
                    conditions.append((0, 0))  # Default values
                    
            # Group by conditions for stratified split
            condition_groups = {}
            for i, cond in enumerate(conditions):
                if cond not in condition_groups:
                    condition_groups[cond] = []
                condition_groups[cond].append(graph_paths[i])
                
            # Perform stratified split with sorted conditions for reproducibility
            train_paths, val_paths, test_paths = [], [], []
            
            for cond in sorted(condition_groups):
                paths = condition_groups[cond]
                np.random.shuffle(paths)
                n = len(paths)
                
                # For very small groups, just add to training
                if n == 1:
                    train_paths.extend(paths)
                elif n == 2:
                    train_paths.extend(paths[:1])
                    test_paths.extend(paths[1:])
                else:
                    # Normal split for groups with 3+ samples
                    n_train = max(1, int(train_ratio * n))
                    n_val = max(1, int(val_ratio * n))
                    n_test = max(1, n - n_train - n_val)
                    
                    # Adjust if total exceeds available samples
                    total_assigned = n_train + n_val + n_test
                    if total_assigned > n:
                        # Reduce validation first, then test
                        if n_val > 1 and total_assigned - n >= 1:
                            n_val -= min(n_val - 1, total_assigned - n)
                        if n_test > 1 and n_train + n_val + n_test > n:
                            n_test = n - n_train - n_val
                    
                    train_paths.extend(paths[:n_train])
                    val_paths.extend(paths[n_train:n_train + n_val])
                    test_paths.extend(paths[n_train + n_val:n_train + n_val + n_test])
                
            logger.info(f"Stratified split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
            
            # If stratified split resulted in empty val/test sets, fall back to simple split
            if len(val_paths) == 0 or len(test_paths) == 0:
                logger.warning("Stratified split resulted in empty val/test sets, using simple random split")
                raise Exception("Fallback to random split")
            
        except Exception as e:
            logger.warning(f"Stratified split failed ({e}), using random split")
            # Fallback to random split
            np.random.shuffle(graph_paths)
            n = len(graph_paths)
            
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            # Ensure at least 1 sample in each split if possible
            if n_val == 0 and n > 2:
                n_val = 1
                n_train -= 1
            if n - n_train - n_val == 0 and n > 1:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 0:
                    n_val -= 1
            
            train_paths = graph_paths[:n_train]
            val_paths = graph_paths[n_train:n_train + n_val]
            test_paths = graph_paths[n_train + n_val:]
            
            logger.info(f"Random split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
            
        return {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }
    
    def compute_robust_stats(self, paths: List[Path]) -> Dict[str, Dict[str, List[float]]]:
        """Compute normalization statistics with robust handling."""
        logger.info("Computing normalization statistics...")
        
        # Initialize collectors
        x_collector = []
        edge_collector = []
        y_collector = []
        
        # Collect data with error handling
        valid_paths = []
        for path in tqdm(paths, desc="Loading training data"):
            try:
                data = torch.load(path, map_location='cpu', weights_only=False)
                
                # Validate data structure
                if not all(hasattr(data, attr) for attr in ['x', 'edge_attr', 'y']):
                    logger.warning(f"Missing attributes in {path}, skipping")
                    continue
                    
                # Check for NaN/Inf values
                if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                    logger.warning(f"NaN/Inf in node features for {path}, skipping")
                    continue
                    
                if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
                    logger.warning(f"NaN/Inf in edge features for {path}, skipping")
                    continue
                    
                if torch.isnan(data.y).any() or torch.isinf(data.y).any():
                    logger.warning(f"NaN/Inf in targets for {path}, skipping")
                    continue
                
                x_collector.append(data.x.numpy())
                edge_collector.append(data.edge_attr.numpy())
                y_collector.append(data.y.numpy())
                valid_paths.append(path)
                
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue
        
        if len(valid_paths) == 0:
            raise ValueError("No valid training files found")
            
        logger.info(f"Using {len(valid_paths)}/{len(paths)} files for normalization")
        
        # Stack arrays
        x_all = np.vstack(x_collector)
        edge_all = np.vstack(edge_collector)
        y_all = np.vstack(y_collector)
        
        # Compute robust statistics (using axis=0 for feature-wise stats)
        def robust_stats(arr):
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            
            # Handle zero std (add small epsilon to avoid division by zero)
            std = np.where(std < 1e-8, 1.0, std)
            
            return mean.tolist(), std.tolist()
        
        x_mean, x_std = robust_stats(x_all)
        edge_mean, edge_std = robust_stats(edge_all)
        y_mean, y_std = robust_stats(y_all)
        
        stats = {
            'x': {'mean': x_mean, 'std': x_std},
            'edge_attr': {'mean': edge_mean, 'std': edge_std},
            'y': {'mean': y_mean, 'std': y_std}
        }
        
        # Log statistics summary
        logger.info(f"Node features: shape={x_all.shape}, mean_range=[{np.min(x_mean):.3f}, {np.max(x_mean):.3f}]")
        logger.info(f"Edge features: shape={edge_all.shape}, mean_range=[{np.min(edge_mean):.3f}, {np.max(edge_mean):.3f}]")
        logger.info(f"Targets: shape={y_all.shape}, mean_range=[{np.min(y_mean):.3f}, {np.max(y_mean):.3f}]")
        
        return stats
    
    def normalize_tensor(self, tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        """Normalize tensor with proper broadcasting."""
        mean_tensor = torch.from_numpy(mean).float()
        std_tensor = torch.from_numpy(std).float()
        
        # Ensure proper broadcasting
        if tensor.dim() == 2 and mean_tensor.dim() == 1:
            mean_tensor = mean_tensor.unsqueeze(0)
            std_tensor = std_tensor.unsqueeze(0)
            
        return (tensor - mean_tensor) / std_tensor
    
    def normalize_graphs(self, graph_paths: List[Path], stats: Dict) -> None:
        """Normalize all graphs with improved error handling and validation."""
        logger.info("Normalizing graphs...")
        
        # Convert stats to numpy arrays for efficiency
        x_mean = np.array(stats['x']['mean'])
        x_std = np.array(stats['x']['std'])
        edge_mean = np.array(stats['edge_attr']['mean'])
        edge_std = np.array(stats['edge_attr']['std'])
        y_mean = np.array(stats['y']['mean'])
        y_std = np.array(stats['y']['std'])
        
        failed_files = []
        
        for path in tqdm(graph_paths, desc="Normalizing"):
            try:
                # Load data
                data = torch.load(path, map_location='cpu', weights_only=False)
                
                # Normalize with proper error handling
                data.x = self.normalize_tensor(data.x, x_mean, x_std)
                data.edge_attr = self.normalize_tensor(data.edge_attr, edge_mean, edge_std)
                data.y = self.normalize_tensor(data.y, y_mean, y_std)
                
                # Validate normalization results
                if not torch.isfinite(data.x).all():
                    logger.error(f"Infinite or NaN in normalized node features: {path}")
                    failed_files.append(path)
                    continue
                
                if not torch.isfinite(data.edge_attr).all():
                    logger.error(f"Infinite or NaN in normalized edge features: {path}")
                    failed_files.append(path)
                    continue
                    
                if not torch.isfinite(data.y).all():
                    logger.error(f"Infinite or NaN in normalized targets: {path}")
                    failed_files.append(path)
                    continue
                
                # Determine output path
                if self.out_dir != self.graph_dir:
                    # Create subdirectory structure in output directory
                    rel_path = path.relative_to(self.graph_dir)
                    out_path = self.out_dir / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_path = path
                
                # Save normalized data
                torch.save(data, out_path)
                
            except Exception as e:
                logger.error(f"Failed to normalize {path}: {e}")
                failed_files.append(path)
                
        if failed_files:
            logger.warning(f"Failed to normalize {len(failed_files)} files")
            
    def save_metadata(self, split: Dict[str, List[Path]], stats: Dict) -> None:
        """Save split and normalization metadata with enhanced information."""
        # Convert paths to relative strings
        split_rel = {}
        for split_name, paths in split.items():
            if self.out_dir != self.graph_dir:
                # Use relative paths from output directory
                split_rel[split_name] = [str(path.relative_to(self.graph_dir)) for path in paths]
            else:
                split_rel[split_name] = [str(path.relative_to(self.graph_dir)) for path in paths]
        
        # Enhanced metadata with feature dimensions and summary stats
        total_files = sum(len(paths) for paths in split.values())
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_files': total_files,
            'split_ratios': {k: len(v) / total_files for k, v in split.items()},
            'seed': self.seed,
            'feature_dims': {
                'x': len(stats['x']['mean']),
                'edge_attr': len(stats['edge_attr']['mean']),
                'y': len(stats['y']['mean'])
            },
            'normalization_ranges': {
                'x': {
                    'mean_min': float(np.min(stats['x']['mean'])),
                    'mean_max': float(np.max(stats['x']['mean'])),
                    'std_min': float(np.min(stats['x']['std'])),
                    'std_max': float(np.max(stats['x']['std']))
                },
                'edge_attr': {
                    'mean_min': float(np.min(stats['edge_attr']['mean'])),
                    'mean_max': float(np.max(stats['edge_attr']['mean'])),
                    'std_min': float(np.min(stats['edge_attr']['std'])),
                    'std_max': float(np.max(stats['edge_attr']['std']))
                },
                'y': {
                    'mean_min': float(np.min(stats['y']['mean'])),
                    'mean_max': float(np.max(stats['y']['mean'])),
                    'std_min': float(np.min(stats['y']['std'])),
                    'std_max': float(np.max(stats['y']['std']))
                }
            },
            'directories': {
                'input_dir': str(self.graph_dir),
                'output_dir': str(self.out_dir)
            }
        }
        
        # Save split information
        split_file = self.out_dir / 'split.json'
        with open(split_file, 'w') as f:
            json.dump({'split': split_rel, 'metadata': metadata}, f, indent=2)
        
        # Save normalization stats
        stats_file = self.out_dir / 'norm_stats.json'
        stats_with_meta = {
            'stats': stats,
            'metadata': metadata
        }
        with open(stats_file, 'w') as f:
            json.dump(stats_with_meta, f, indent=2)
            
        logger.info(f"Saved metadata to {split_file} and {stats_file}")
    
    def run(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
        """Run the complete normalization pipeline."""
        logger.info("Starting Stage 4: Normalization & Dataset Split")
        
        # Find graph files
        graph_paths = self.find_graph_files()
        
        # Create split
        split = self.create_stratified_split(graph_paths, train_ratio, val_ratio)
        logger.info(f"Dataset split: Train={len(split['train'])}, Val={len(split['val'])}, Test={len(split['test'])}")
        
        # Compute normalization stats on training set only
        stats = self.compute_robust_stats(split['train'])
        
        # Normalize all graphs
        all_paths = split['train'] + split['val'] + split['test']
        self.normalize_graphs(all_paths, stats)
        
        # Save metadata
        self.save_metadata(split, stats)
        
        logger.info("✅ Stage 4 completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("STAGE 4 SUMMARY")
        print("="*50)
        print(f"Graph directory: {self.graph_dir}")
        if self.out_dir != self.graph_dir:
            print(f"Output directory: {self.out_dir}")
        print(f"Total files: {len(all_paths)}")
        print(f"Train: {len(split['train'])} ({len(split['train'])/len(all_paths)*100:.1f}%)")
        print(f"Val: {len(split['val'])} ({len(split['val'])/len(all_paths)*100:.1f}%)")
        print(f"Test: {len(split['test'])} ({len(split['test'])/len(all_paths)*100:.1f}%)")
        print(f"Node features: {len(stats['x']['mean'])} dimensions")
        print(f"Edge features: {len(stats['edge_attr']['mean'])} dimensions")
        print(f"Target features: {len(stats['y']['mean'])} dimensions")
        print("="*50)

def main():
    """Main function with command line interface."""
    root = _project_root()

    parser = argparse.ArgumentParser(description='Stage 4: Normalize and split GNN airfoil dataset')

    # --- Minimal change: defaults aligned with new repo layout ---
    parser.add_argument('--graph_dir', type=str, default=str(root / 'data' / 'graphs'),
                       help='Directory containing graph files')
    parser.add_argument('--out_dir', type=str, default=str(root / 'data' / 'normalized_graphs'),
                       help='Output directory for normalized graphs (default: <project_root>/data/normalized_graphs)')

    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    
    # Run normalization
    normalizer = DatasetNormalizer(args.graph_dir, args.out_dir, args.seed)
    normalizer.run(args.train_ratio, args.val_ratio)

if __name__ == "__main__":
    main()