from pathlib import Path
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
import torch

class CheckpointManager:
    """A simple checkpoint manager that saves to a structured directory."""

    def __init__(self, base_dir, exp_name, run_id):
        self.base_dir = Path(base_dir) / exp_name / run_id

    def get_path(self, type='checkpoints'):
        path = self.base_dir  / type
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, model, model_name, epoch=None):
        filename = f"{model_name}_final.pth" if epoch is None else f"{model_name}_epoch_{epoch}.pth"
        save_path = self.get_path('checkpoints') / filename
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

    def load(self, model, model_name, device, epoch=None):
        filename = f"{model_name}_final.pth" if epoch is None else f"{model_name}_epoch_{epoch}.pth"
        load_path = self.get_path('checkpoints') / filename
        if not load_path.exists(): raise FileNotFoundError(f"Checkpoint {load_path} not found.")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded model checkpoint from {load_path}")
        return model

def save_config_to_yaml(config, log_dir):
    """Saves the configuration Box object to a YAML file."""
    config_path = Path(log_dir) / f'{config.experiment.name}_{config.experiment.run}.yml'
    with open(config_path, 'w') as f:
        # Convert Box object to a standard dict for clean YAML output
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")

def is_cluster():
    import socket, os
    hostname = socket.gethostname()
    return "mscluster" in hostname or "wits" in hostname or os.environ.get("IS_CLUSTER") == "1"

def tiny_subset(dataset: Dataset, num_items: int = 8) -> Subset:
    """Return a tiny subset of the dataset for quick overfitting checks."""
    indices = list(range(min(len(dataset), num_items)))
    return Subset(dataset, indices)
