#!/usr/bin/env python3
"""
Training configuration presets.

These are example dictionaries you can import in your training scripts, or export
to JSON for reproducible CLI runs.

Examples:
  python config.py        # writes config_pv_only.json and config_full.json
"""

import json

# Pressure + velocity only (no CL/CD)
config_pressure_velocity = {
    "hidden_dim": 128,
    "num_layers": 4,
    "dropout": 0.1,
    "use_global_pred": False,

    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 100,
    "weight_decay": 1e-5,

    "pressure_weight": 1.0,
    "velocity_weight": 1.0,
    "cl_weight": 0.0,
    "cd_weight": 0.0,

    "data_dir": "normalized_graphs",
    "output_dir": "training_output_pv_only",
    "checkpoint_dir": "checkpoints_pv_only",

    "save_every": 10,
    "early_stopping_patience": 20,
    "log_every": 1,
}

# Pressure + velocity + CL/CD (requires dataset labels for CL/CD)
config_full_model = {
    "hidden_dim": 128,
    "num_layers": 4,
    "dropout": 0.1,
    "use_global_pred": True,

    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 100,
    "weight_decay": 1e-5,

    "pressure_weight": 1.0,
    "velocity_weight": 1.0,
    "cl_weight": 0.1,
    "cd_weight": 0.1,

    "data_dir": "normalized_graphs",
    "output_dir": "training_output_full",
    "checkpoint_dir": "checkpoints_full",

    "save_every": 10,
    "early_stopping_patience": 20,
    "log_every": 1,
}

if __name__ == "__main__":
    with open("config_pv_only.json", "w") as f:
        json.dump(config_pressure_velocity, f, indent=2)

    with open("config_full.json", "w") as f:
        json.dump(config_full_model, f, indent=2)

    print("Configuration files created:")
    print("- config_pv_only.json")
    print("- config_full.json")