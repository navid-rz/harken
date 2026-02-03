import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FeatureDataset(Dataset):
    """
    Generic dataset for preprocessed audio features (MFCC or log-mel spectrograms).
    Loads .npy files from a directory structure where each subfolder represents a label.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to preprocessed data directory (contains one subfolder per label)
            transform (callable, optional): Optional transform applied on a sample tensor shaped (C, T)
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []          # list of (file_path, label_idx)
        self.label_to_index = {}   # e.g., {"yes":0, "no":1, ...}
        self.index_to_label = {}   # reverse mapping
        self._prepare_dataset()

        # quick access to labels only (no I/O)
        self.labels = [lbl for _, lbl in self.samples]

    def _prepare_dataset(self):
        label_idx = 0
        # Stable ordering for reproducibility
        for label_name in sorted(os.listdir(self.root_dir)):
            label_path = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            if label_name not in self.label_to_index:
                self.label_to_index[label_name] = label_idx
                self.index_to_label[label_idx] = label_name
                label_idx += 1

            for fname in sorted(os.listdir(label_path)):
                if fname.lower().endswith(".npy"):
                    file_path = os.path.join(label_path, fname)
                    self.samples.append((file_path, self.label_to_index[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_idx = self.samples[idx]

        # Load features and ensure we have a clean, resizable tensor
        features = np.load(file_path, mmap_mode=None, allow_pickle=False)
        # Convert to tensor with own storage - no sharing with numpy
        x = torch.tensor(features, dtype=torch.float32)  # Creates new tensor with own storage
        x = x.transpose(0, 1).contiguous()  # (C, T)

        if self.transform is not None:
            x = self.transform(x)

        # Return label name instead of index for MultiClassDataset compatibility
        label_name = self.index_to_label[label_idx]
        return x, label_name


# Example quick check
if __name__ == "__main__":
    ds = FeatureDataset("data/preprocessed")
    print("Num samples:", len(ds))
    x0, y0 = ds[0]
    print("First item:", x0.shape, y0)           # expect (C, T), e.g., (16 or 28, ~60-100)

    # Simple split (your training code does stratified splits anyway)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

    bx, by = next(iter(train_loader))
    print("Batch X shape:", bx.shape)            # (B, C, T)
    print("Batch y shape:", by.shape)            # (B,)
