# data_loader/utils.py

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from data_loader.feature_dataset import FeatureDataset
from data_loader.binary_dataset import BinaryClassDataset
from data_loader.multiclass_dataset import MultiClassDataset


def stratified_train_val_test_indices(labels, val_frac, test_frac, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        val_idx.extend(idx_c[:n_val].tolist())
        test_idx.extend(idx_c[n_val:n_val+n_test].tolist())
        train_idx.extend(idx_c[n_val+n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def make_datasets(cfg, which="all", batch_size=64, train_transform=None):
    """
    Build train/val/test datasets and return DataLoaders.
    
    Args:
        cfg: Configuration dict
        which: "all", "train", "val", or "test" (defaults to "all")
        batch_size: Batch size for DataLoaders
        train_transform: Optional transform to apply to training data only
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = cfg.get("data", {})
    task_cfg = cfg.get("task", {})
    train_cfg = cfg.get("train", {})

    root = data_cfg.get("preprocessed_dir")
    if not root:
        raise ValueError("Config must define data.preprocessed_dir")

    # Split fracs (defaults if not provided)
    val_frac = float(data_cfg.get("val_frac", 0.1))
    test_frac = float(data_cfg.get("test_frac", 0.1))
    seed = int(cfg.get("augmentation", {}).get("seed", 0)) if "augmentation" in cfg else 0

    # Loader knobs - disabled multiprocessing for simplicity
    num_workers = 0
    pin_memory = bool(train_cfg.get("pin_memory", False))

    # Base dataset (raw feature items with full label space)
    base = FeatureDataset(root)

    # For stratified split, collect base labels
    base_labels = []
    for file_path, label_idx in base.samples:
        # label_idx is already an integer index from FeatureDataset
        base_labels.append(label_idx)

    # Build splits
    train_idx, val_idx, test_idx = stratified_train_val_test_indices(
        base_labels, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    
    # Apply transform only to training base dataset if provided
    base_train = FeatureDataset(root, transform=train_transform) if train_transform else base
    base_val = base
    base_test = base
    
    # Create subsets
    base_train_subset = Subset(base_train, train_idx)
    base_val_subset = Subset(base_val, val_idx)
    base_test_subset = Subset(base_test, test_idx)

    # Task selection
    task_type = str(task_cfg.get("type", "multiclass")).lower()

    if task_type == "multiclass":
        class_list = list(task_cfg.get("class_list", []))
        include_unknown = bool(task_cfg.get("include_unknown", True))
        include_background = bool(task_cfg.get("include_background", True))
        background_label = data_cfg.get("background_label", "_background_noise_")

        # Build simple target class list 
        target_classes = class_list.copy()
        if include_unknown:
            target_classes.append("unknown")
        if include_background:
            target_classes.append(background_label)

        # Wrapped datasets - simplified interface
        train_ds = MultiClassDataset(base_train_subset, target_classes)
        val_ds = MultiClassDataset(base_val_subset, target_classes)
        test_ds = MultiClassDataset(base_test_subset, target_classes)
    elif task_type == "binary":
        # Implement if you use BinaryClassDataset; placeholder for now
        raise NotImplementedError("Binary task path not implemented in make_datasets()")
    else:
        raise ValueError(f"Unknown task.type '{task_type}'")

    # DataLoaders - simple configuration
    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **dl_kwargs)

    return train_loader, val_loader, test_loader

def get_num_classes(cfg):
    class_list = list(cfg["task"]["class_list"])
    if bool(cfg["task"].get("include_unknown", False)):
        class_list.append("unknown")
    if bool(cfg["task"].get("include_background", False)):
        background_label = cfg["data"].get("background_label", "_background_noise_")
        class_list.append(background_label)
    return len(class_list)

class FeatureAugment:
    """Zero-padded time shift + optional Gaussian noise for (C, T) feature tensors (MFCC or log-mel)."""
    def __init__(self, hop_length_s: float, max_shift_ms: float = 100.0,
                 noise_prob: float = 0.15, noise_std_factor: float = 0.05, seed=None):
        import numpy as _np
        self.hop_length_s = float(hop_length_s)
        self.max_shift_ms = float(max_shift_ms)
        self.noise_prob = float(noise_prob)
        self.noise_std_factor = float(noise_std_factor)
        self.rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        self.max_shift_frames = int(round((self.max_shift_ms / 1000.0) / self.hop_length_s))

    def _shift_with_zeros(self, x: torch.Tensor, s: int) -> torch.Tensor:
        C, T = x.shape
        if s == 0:
            return x
        if abs(s) >= T:
            return torch.zeros_like(x)
        if s > 0:
            out = torch.zeros_like(x)
            out[:, s:] = x[:, :T - s]
            return out
        s = -s
        out = torch.zeros_like(x)
        out[:, :T - s] = x[:, s:]
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.ndim != 2:
            raise ValueError(f"FeatureAugment expects (C, T), got shape {tuple(x.shape)}")
        C, T = x.shape
        
        # Time shift augmentation
        if self.max_shift_frames > 0 and T > 1:
            s = int(self.rng.integers(-self.max_shift_frames, self.max_shift_frames + 1))
            if s != 0:
                x = self._shift_with_zeros(x, s)
        
        # Noise augmentation
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std_factor * x.std()
            x = x + noise
        
        # Ensure we return a contiguous tensor with its own storage for DataLoader batching
        return x.contiguous().clone()