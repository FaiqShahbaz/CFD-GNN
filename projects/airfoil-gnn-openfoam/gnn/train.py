#!/usr/bin/env python3
"""
Train GNN model for airfoil flow-field prediction (pressure + velocity, optional CL/CD).

Default repo layout (relative to this script location):
  <project_root>/data/normalized_graphs   (input: normalized graph.pt files + split.json)
  <project_root>/data/training_output     (output: logs, plots, final model)
  <project_root>/data/checkpoints         (output: checkpoints)

Typical usage:
  python training.py
  python training.py --epochs 200 --batch-size 8
  python training.py --resume <project_root>/data/checkpoints/best_model.pt
"""

import matplotlib
matplotlib.use('Agg')

import os
import json
import argparse
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeomDataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Assuming these are from previous stages
# from models.airfoil_gnn import AirfoilGNN, AirfoilLoss

# project root for safe defaults 
def _project_root() -> Path:
    # expected location: <project_root>/gnn/training.py
    return Path(__file__).resolve().parents[1]

class Config:
    """Configuration class for training parameters"""
    def __init__(self):
        # Model parameters
        self.hidden_dim = 128
        self.num_layers = 4
        self.dropout = 0.1
        self.use_global_pred = True

        # Training parameters
        self.learning_rate = 1e-3
        self.batch_size = 16
        self.epochs = 100
        self.weight_decay = 1e-5

        # Loss weights
        self.pressure_weight = 1.0
        self.velocity_weight = 1.0
        self.cl_weight = 0.1
        self.cd_weight = 0.1

        # Paths (minimal change: match repo layout)
        root = _project_root()
        self.data_dir = str(root / "data" / "normalized_graphs")
        self.output_dir = str(root / "data" / "training_output")
        self.checkpoint_dir = str(root / "data" / "checkpoints")

        # Training settings
        self.save_every = 10
        self.early_stopping_patience = 20
        self.log_every = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class AirfoilLoss(nn.Module):
    """Custom loss function for airfoil flow prediction"""

    def __init__(self, pressure_weight=1.0, velocity_weight=1.0, cl_weight=0.1, cd_weight=0.1):
        super().__init__()
        self.pressure_weight = pressure_weight
        self.velocity_weight = velocity_weight
        self.cl_weight = cl_weight
        self.cd_weight = cd_weight

    def forward(self, pred_pressure, pred_velocity, pred_global, batch):
        # Node-level losses
        pressure_loss = F.mse_loss(pred_pressure, batch.pressure)
        velocity_loss = F.mse_loss(pred_velocity, batch.velocity)

        node_loss = (self.pressure_weight * pressure_loss +
                    self.velocity_weight * velocity_loss)

        # Global losses
        global_loss = 0.0
        if pred_global is not None and hasattr(batch, 'cl') and hasattr(batch, 'cd'):
            cl_loss = F.mse_loss(pred_global[:, 0], batch.cl)
            cd_loss = F.mse_loss(pred_global[:, 1], batch.cd)
            global_loss = self.cl_weight * cl_loss + self.cd_weight * cd_loss

        return node_loss + global_loss

class AirfoilDataset:
    """PyTorch Geometric dataset for airfoil flow data"""

    def __init__(self, data_dir, split_file, split_type="train"):
        self.data_dir = Path(data_dir)
        self.split_type = split_type

        # Load split information
        with open(split_file, 'r') as f:
            splits = json.load(f)

        self.sample_ids = splits["split"][split_type]
        self.graphs = []

        # Load all graphs for this split
        print(f"Loading {split_type} graphs...")
        for sample_id in tqdm(self.sample_ids):
            graph_path = self.data_dir / sample_id
            if graph_path.exists():
                graph = torch.load(graph_path)
                self.graphs.append(graph)
            else:
                print(f"Warning: Graph {sample_id} not found")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def assign_node_targets(batch):
    """Assign node-level targets from batch.y and global targets if available"""
    if not hasattr(batch, 'pressure'):
        # Assume y contains [pressure, velocity_x, velocity_y, ...]
        batch.pressure = batch.y[:, 0:1]  # First column is pressure
        batch.velocity = batch.y[:, 1:3]  # Next 2 columns are velocity

        # Check if global features are available
        if batch.y.size(1) >= 5:  # [p, vx, vy, cl, cd]
            # Extract global features per graph
            batch_size = batch.batch.max().item() + 1
            batch.cl = torch.zeros(batch_size, device=batch.y.device)
            batch.cd = torch.zeros(batch_size, device=batch.y.device)

            for i in range(batch_size):
                mask = batch.batch == i
                if mask.sum() > 0:
                    # Take the first occurrence of cl/cd for each graph
                    batch.cl[i] = batch.y[mask, 3][0]
                    batch.cd[i] = batch.y[mask, 4][0]

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    split_file = os.path.join(config.data_dir, "split.json")

    # Create datasets
    train_dataset = AirfoilDataset(config.data_dir, split_file, "train")
    val_dataset = AirfoilDataset(config.data_dir, split_file, "val")
    test_dataset = AirfoilDataset(config.data_dir, split_file, "test")

    # Create dataloaders
    train_loader = GeomDataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = GeomDataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = GeomDataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader

def detect_global_features(data_loader):
    """Detect if dataset has global features by checking a sample batch"""
    sample_batch = next(iter(data_loader))
    assign_node_targets(sample_batch)

    # Check if batch has cl and cd attributes after assignment
    has_global = (hasattr(sample_batch, 'cl') and hasattr(sample_batch, 'cd') and
                  sample_batch.y.size(1) >= 5)

    return has_global

class MetricsTracker:
    """Track and log training metrics"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # minimal safety

        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_pressure_mae': [],
            'val_pressure_mae': [],
            'train_velocity_mae': [],
            'val_velocity_mae': [],
            'train_cl_error': [],
            'val_cl_error': [],
            'train_cd_error': [],
            'val_cd_error': [],
            'learning_rate': []
        }

        # Setup CSV logging
        self.csv_file = self.output_dir / "training_log.csv"
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def update(self, epoch, train_metrics, val_metrics, lr):
        """Update metrics for current epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['learning_rate'].append(lr)

        # Training metrics
        self.metrics['train_loss'].append(train_metrics['loss'])
        self.metrics['train_pressure_mae'].append(train_metrics['pressure_mae'])
        self.metrics['train_velocity_mae'].append(train_metrics['velocity_mae'])
        self.metrics['train_cl_error'].append(train_metrics['cl_error'])
        self.metrics['train_cd_error'].append(train_metrics['cd_error'])

        # Validation metrics
        self.metrics['val_loss'].append(val_metrics['loss'])
        self.metrics['val_pressure_mae'].append(val_metrics['pressure_mae'])
        self.metrics['val_velocity_mae'].append(val_metrics['velocity_mae'])
        self.metrics['val_cl_error'].append(val_metrics['cl_error'])
        self.metrics['val_cd_error'].append(val_metrics['cd_error'])

        # Log to console
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val P-MAE: {val_metrics['pressure_mae']:.6f} | "
            f"Val V-MAE: {val_metrics['velocity_mae']:.6f}"
        )

        # Save to CSV
        self.save_csv()

    def save_csv(self):
        """Save metrics to CSV file"""
        if len(self.metrics['epoch']) == 1:
            # Write header for first epoch
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.metrics.keys())

        # Append latest metrics
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [self.metrics[key][-1] for key in self.metrics.keys()]
            writer.writerow(row)

    def plot_metrics(self, has_global_features=True):
        """Plot training curves"""
        fig_size = (15, 10) if has_global_features else (10, 6)
        n_rows = 2 if has_global_features else 1
        n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Loss curves
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Pressure MAE
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['train_pressure_mae'], label='Train')
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['val_pressure_mae'], label='Val')
        axes[0, 1].set_title('Pressure MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Velocity MAE
        axes[0, 2].plot(self.metrics['epoch'], self.metrics['train_velocity_mae'], label='Train')
        axes[0, 2].plot(self.metrics['epoch'], self.metrics['val_velocity_mae'], label='Val')
        axes[0, 2].set_title('Velocity MAE')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        if has_global_features:
            # CL Error
            axes[1, 0].plot(self.metrics['epoch'], self.metrics['train_cl_error'], label='Train')
            axes[1, 0].plot(self.metrics['epoch'], self.metrics['val_cl_error'], label='Val')
            axes[1, 0].set_title('CL Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # CD Error
            axes[1, 1].plot(self.metrics['epoch'], self.metrics['train_cd_error'], label='Train')
            axes[1, 1].plot(self.metrics['epoch'], self.metrics['val_cd_error'], label='Val')
            axes[1, 1].set_title('CD Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            # Learning Rate
            axes[1, 2].plot(self.metrics['epoch'], self.metrics['learning_rate'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

def compute_metrics(model, data_loader, criterion, device, has_global_features=True):
    """Compute metrics on a dataset"""
    model.eval()
    total_loss = 0
    pressure_mae = 0
    velocity_mae = 0
    cl_error = 0
    cd_error = 0
    num_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            assign_node_targets(batch)

            # Forward pass
            output = model(batch)

            # Handle different model output formats
            if isinstance(output, dict):
                pred_pressure = output['node_pred'][:, 0:1]
                pred_velocity = output['node_pred'][:, 1:3]
                pred_global = output.get('global_pred', None)
            elif isinstance(output, tuple):
                if len(output) == 2:
                    node_pred, pred_global = output
                    pred_pressure = node_pred[:, 0:1]
                    pred_velocity = node_pred[:, 1:3]
                else:
                    # Legacy format: (pressure, velocity, global)
                    pred_pressure, pred_velocity, pred_global = output
            else:
                # Single tensor output
                pred_pressure = output[:, 0:1]
                pred_velocity = output[:, 1:3]
                pred_global = None

            # Compute loss using custom loss function
            loss = criterion(pred_pressure, pred_velocity, pred_global, batch)
            total_loss += loss.item()

            # Compute MAE for pressure and velocity
            pressure_mae += F.l1_loss(pred_pressure, batch.pressure).item()
            velocity_mae += F.l1_loss(pred_velocity, batch.velocity).item()

            # Compute CL/CD errors if available
            if (pred_global is not None and has_global_features and
                hasattr(batch, 'cl') and hasattr(batch, 'cd')):
                pred_cl, pred_cd = pred_global[:, 0], pred_global[:, 1]
                cl_error += F.l1_loss(pred_cl, batch.cl).item()
                cd_error += F.l1_loss(pred_cd, batch.cd).item()

            num_samples += 1

    return {
        'loss': total_loss / num_samples,
        'pressure_mae': pressure_mae / num_samples,
        'velocity_mae': velocity_mae / num_samples,
        'cl_error': cl_error / num_samples if has_global_features else 0.0,
        'cd_error': cd_error / num_samples if has_global_features else 0.0
    }

def train_epoch(model, train_loader, optimizer, criterion, device, has_global_features=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        assign_node_targets(batch)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(batch)

        # Handle different model output formats
        if isinstance(output, dict):
            pred_pressure = output['node_pred'][:, 0:1]
            pred_velocity = output['node_pred'][:, 1:3]
            pred_global = output.get('global_pred', None)
        elif isinstance(output, tuple):
            if len(output) == 2:
                node_pred, pred_global = output
                pred_pressure = node_pred[:, 0:1]
                pred_velocity = node_pred[:, 1:3]
            else:
                # Legacy format: (pressure, velocity, global)
                pred_pressure, pred_velocity, pred_global = output
        else:
            # Single tensor output
            pred_pressure = output[:, 0:1]
            pred_velocity = output[:, 1:3]
            pred_global = None

        # Compute loss using custom loss function
        loss = criterion(pred_pressure, pred_velocity, pred_global, batch)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(train_loader)

class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']

def main():
    """Main training function"""
    # Parse arguments
    root = _project_root()

    parser = argparse.ArgumentParser(description='Train GNN for airfoil flow prediction')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    # --- Minimal change: defaults aligned with new repo layout ---
    parser.add_argument('--data-dir', type=str, default=str(root / 'data' / 'normalized_graphs'))
    parser.add_argument('--output-dir', type=str, default=str(root / 'data' / 'training_output'))

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--num-layers', type=int, help='Number of message passing layers')
    parser.add_argument('--weight-decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--early-stopping-patience', type=int, help='Early stopping patience')

    args = parser.parse_args()

    # Initialize config
    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience

    # Create output directories (minimal safety)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Get sample batch to determine input dimensions and global features
    sample_batch = next(iter(train_loader))
    node_features = sample_batch.x.size(1)
    edge_features = sample_batch.edge_attr.size(1) if sample_batch.edge_attr is not None else 0

    # Check if dataset has global features (CL/CD)
    has_global_features = detect_global_features(train_loader)
    print(f"Dataset has global features (CL/CD): {has_global_features}")

    # Initialize model
    print("Initializing model...")
    try:
        from models.airfoil_gnn import AirfoilGNN
    except ImportError:
        print("Warning: AirfoilGNN not found. Please implement or import the model.")
        return

    model = AirfoilGNN(
        node_input_dim=node_features,
        edge_input_dim=edge_features,
        hidden_dim=config.hidden_dim,
        num_processor_layers=config.num_layers,
        dropout=config.dropout,
        predict_global=config.use_global_pred and has_global_features
    )
    model = model.to(device)

    # Initialize loss function
    criterion = AirfoilLoss(
        pressure_weight=config.pressure_weight,
        velocity_weight=config.velocity_weight,
        cl_weight=config.cl_weight if has_global_features else 0.0,
        cd_weight=config.cd_weight if has_global_features else 0.0
    )

    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Initialize metrics tracker and early stopping
    metrics_tracker = MetricsTracker(config.output_dir)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}")

    print("Starting training...")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, has_global_features)

        # Evaluate on train and validation sets
        train_metrics = compute_metrics(model, train_loader, criterion, device, has_global_features)
        val_metrics = compute_metrics(model, val_loader, criterion, device, has_global_features)

        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Update metrics
        metrics_tracker.update(epoch, train_metrics, val_metrics, current_lr)

        # Save checkpoint if best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['loss'], checkpoint_path)
            print(f"Saved best model with val_loss: {best_val_loss:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['loss'], checkpoint_path)

        # Check early stopping
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Plot metrics periodically
        if (epoch + 1) % config.log_every == 0:
            metrics_tracker.plot_metrics(has_global_features)

    # Save last epoch checkpoint
    last_checkpoint_path = Path(config.checkpoint_dir) / "last_epoch.pt"
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['loss'], last_checkpoint_path)

    print("Training completed!")

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics = compute_metrics(model, test_loader, criterion, device, has_global_features)
    metrics_tracker.logger.info(f"Test Results:")
    metrics_tracker.logger.info(f"  Loss: {test_metrics['loss']:.6f}")
    metrics_tracker.logger.info(f"  Pressure MAE: {test_metrics['pressure_mae']:.6f}")
    metrics_tracker.logger.info(f"  Velocity MAE: {test_metrics['velocity_mae']:.6f}")
    if has_global_features:
        metrics_tracker.logger.info(f"  CL Error: {test_metrics['cl_error']:.6f}")
        metrics_tracker.logger.info(f"  CD Error: {test_metrics['cd_error']:.6f}")

    # Save final model
    final_model_path = Path(config.output_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)

    # Final metrics plot
    metrics_tracker.plot_metrics(has_global_features)

    print(f"Results saved to: {config.output_dir}")

if __name__ == "__main__":
    main()