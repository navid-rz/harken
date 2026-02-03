import torch
from torch.utils.data import Dataset, Subset


class MultiClassDataset(Dataset):
    """
    Simple wrapper that filters a base dataset to only include specified classes.
    """
    def __init__(self, base_dataset, target_classes):
        """
        Args:
            base_dataset: A dataset (or Subset) with samples that return (features, label)
            target_classes: List of class names to include
        """
        self.base_dataset = base_dataset
        self.target_classes = target_classes
        
        # Handle Subset case
        if isinstance(base_dataset, Subset):
            original_dataset = base_dataset.dataset
            subset_indices = base_dataset.indices
            
            # Build index mapping for subset
            self.indices = []
            for subset_idx, original_idx in enumerate(subset_indices):
                file_path, label_idx = original_dataset.samples[original_idx]
                label_name = original_dataset.index_to_label[label_idx]
                if label_name in target_classes:
                    self.indices.append(subset_idx)
        else:
            # Build simple index mapping
            self.indices = []
            for i in range(len(base_dataset)):
                file_path, label_idx = base_dataset.samples[i]
                label_name = base_dataset.index_to_label[label_idx]
                if label_name in target_classes:
                    self.indices.append(i)
        
        # Create class-to-index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"MultiClassDataset: Found {len(self.indices)} samples "
              f"for {len(target_classes)} classes: {target_classes}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index in base dataset
        base_idx = self.indices[idx]
        
        # Get features and label from base dataset
        features, label_name = self.base_dataset[base_idx]
        
        # Convert label name to index for this subset
        label_idx = self.class_to_idx[label_name]
        
        return features, label_idx
    
    @property 
    def num_classes(self):
        return len(self.class_to_idx)
    
    @property
    def class_names(self):
        return self.target_classes

