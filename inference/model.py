"""
Model wrapper for keyword spotting inference.
"""
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from train.utils import load_state_dict_forgiving
from model.model import DilatedTCN


def get_mfcc_section(cfg: dict) -> dict:
    """
    Locate feature settings in config (MFCC or log-mel). Supports:
      cfg['features']
      cfg['data']['features']
      cfg['mfcc']  (backward compat)
      cfg['data']['mfcc']  (backward compat)
      cfg['features']['mfcc']
      cfg['audio']['mfcc']
    """
    return (
        cfg.get("features")
        or cfg.get("data", {}).get("features")
        or cfg.get("mfcc")
        or cfg.get("data", {}).get("mfcc")
        or cfg.get("features", {}).get("mfcc")
        or cfg.get("audio", {}).get("mfcc")
    )


def preprocess_wave(y: np.ndarray, sr: int, cfg_mfcc: dict, fixed_duration_s: float) -> torch.Tensor:
    """
    Preprocess audio waveform to MFCC features.
    
    Args:
        y: Audio waveform (1D numpy array)
        sr: Sample rate
        cfg_mfcc: MFCC configuration dict
        fixed_duration_s: Target duration in seconds
        
    Returns:
        MFCC features as (channels, time) tensor
    """
    # Pad / trim to fixed duration
    target_len = int(fixed_duration_s * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Support both n_mfcc and n_features naming
    n_mfcc = int(cfg_mfcc.get("n_mfcc", cfg_mfcc.get("n_features", 28)))
    frame_length_s = float(cfg_mfcc["frame_length_s"])
    hop_length_s = float(cfg_mfcc["hop_length_s"])
    n_fft = int(round(frame_length_s * sr))
    hop_length = int(round(hop_length_s * sr))

    # Librosa MFCC (match training assumptions)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True
    )  # shape (n_mfcc, T)

    # (C, T) float32 tensor
    feat = torch.from_numpy(mfcc).float()
    return feat


class KeywordModel:
    """
    Lightweight inference wrapper for keyword spotting models.
    
    Handles:
    - Config and weight loading
    - MFCC preprocessing
    - Model forward pass
    - Probability output formatting
    """
    
    def __init__(self, cfg_path: str, weights_path: str, device: torch.device):
        """
        Initialize keyword spotting model for inference.
        
        Args:
            cfg_path: Path to config YAML file
            weights_path: Path to model weights (.pt file)
            device: torch.device for inference
        """
        self.cfg = load_config(cfg_path)
        self.mfcc_cfg = get_mfcc_section(self.cfg)
        if self.mfcc_cfg is None:
            raise KeyError(
                "Could not find 'mfcc' section in config. "
                f"Top-level keys: {list(self.cfg.keys())}"
            )

        self.device = device
        self.task_type = self.cfg["task"]["type"]
        self.class_list = list(self.cfg["task"]["class_list"])
        if self.cfg["task"].get("include_unknown"):
            self.class_list.append("unknown")
        if self.cfg["task"].get("include_background"):
            self.class_list.append("background")
        
        self.sr = int(self.mfcc_cfg["sample_rate"])
        self.fixed_duration_s = float(self.mfcc_cfg["fixed_duration_s"])

        # Build and load model
        self.model = DilatedTCN.from_config(self.cfg)
        self.model = load_state_dict_forgiving(self.model, weights_path, device)
        self.model.to(device).eval()

    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio waveform to MFCC features.
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            MFCC features as (channels, time) tensor
        """
        return preprocess_wave(audio, self.sr, self.mfcc_cfg, self.fixed_duration_s)

    def prepare_input(self, feat: torch.Tensor) -> torch.Tensor:
        """Add batch dimension: (C, T) -> (1, C, T)"""
        return feat.unsqueeze(0)

    @torch.no_grad()
    def predict(self, feat: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Run inference and return top-k predictions.
        
        Args:
            feat: MFCC features (C, T)
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        x = self.prepare_input(feat).to(self.device)
        logits = self.model(x)
        
        if self.task_type == "multiclass":
            probs = F.softmax(logits, dim=1)[0]
            topk = min(top_k, probs.numel())
            vals, idx = torch.topk(probs, topk)
            results = [(self.class_list[i], float(vals[j])) for j, i in enumerate(idx)]
            return results
        else:
            prob = torch.sigmoid(logits)[0, 0].item()
            return [("keyword", prob), ("non_keyword", 1 - prob)]

    @torch.no_grad()
    def predict_full(self, feat: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Run inference and return all class probabilities in original order.
        
        Args:
            feat: MFCC features (C, T)
            
        Returns:
            List of (class_name, probability) tuples for all classes
        """
        x = self.prepare_input(feat).to(self.device)
        logits = self.model(x)
        
        if self.task_type == "multiclass":
            probs = F.softmax(logits, dim=1)[0]
            return [(cls, float(probs[i])) for i, cls in enumerate(self.class_list)]
        else:
            prob = torch.sigmoid(logits)[0, 0].item()
            return [("keyword", prob), ("non_keyword", 1 - prob)]

    @torch.no_grad()
    def predict_audio(self, audio: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        End-to-end prediction from raw audio waveform.
        
        Args:
            audio: Audio waveform (1D numpy array)
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        feat = self.preprocess(audio)
        return self.predict(feat, top_k)

    @torch.no_grad()
    def predict_audio_full(self, audio: np.ndarray) -> List[Tuple[str, float]]:
        """
        End-to-end prediction from raw audio waveform, returning all class probabilities.
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            List of (class_name, probability) tuples for all classes
        """
        feat = self.preprocess(audio)
        return self.predict_full(feat)

    @torch.no_grad()
    def evaluate_dataloader(self, dataloader: DataLoader, device: torch.device = None, 
                           threshold: float = 0.5, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance on a DataLoader of preprocessed features.
        
        This method is designed for efficient batch evaluation of validation/test sets,
        particularly useful for comparing original vs quantized model performance.
        
        Args:
            dataloader: DataLoader yielding (features, labels) batches
                       features: (batch_size, channels, time_steps) 
                       labels: (batch_size,) for multiclass or (batch_size, 1) for binary
            device: Device to run evaluation on (defaults to model's device)
            threshold: Classification threshold for binary tasks (ignored for multiclass)
            verbose: Whether to show progress bar during evaluation
            
        Returns:
            Dictionary containing metrics:
            - 'accuracy': Overall classification accuracy [0, 1]
            - 'precision': Macro-averaged precision [0, 1] 
            - 'recall': Macro-averaged recall [0, 1]
            - 'f1': Macro-averaged F1 score [0, 1]
            - 'total_samples': Number of samples evaluated
            - 'correct_samples': Number of correctly classified samples
            
        Note:
            - Uses preprocessed features directly (no audio preprocessing)
            - Efficient batch processing for faster evaluation than single-sample inference
            - Consistent with training evaluation methodology
        """
        if device is None:
            device = self.device
            
        # Move model to evaluation device temporarily if needed
        original_device = next(self.model.parameters()).device
        if device != original_device:
            self.model.to(device)
            
        # Initialize counters for metrics computation
        total_samples = 0
        correct_samples = 0
        
        # For multiclass: track per-class statistics for macro averaging
        if self.task_type == "multiclass":
            num_classes = len(self.class_list)
            # Track true positives, false positives, false negatives per class
            class_tp = torch.zeros(num_classes, device=device)
            class_fp = torch.zeros(num_classes, device=device) 
            class_fn = torch.zeros(num_classes, device=device)
        else:
            # For binary: track overall TP, FP, TN, FN
            tp = fp = tn = fn = 0
        
        # Setup progress bar if requested
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        try:
            # Process all batches
            for batch_features, batch_labels in iterator:
                # Move data to device
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass through model
                logits = self.model(batch_features)
                batch_size = batch_features.size(0)
                total_samples += batch_size
                
                if self.task_type == "multiclass":
                    # Multiclass classification: get predicted classes
                    predicted_classes = torch.argmax(logits, dim=1)  # (batch_size,)
                    true_classes = batch_labels.long()  # (batch_size,)
                    
                    # Count correct predictions for accuracy
                    correct_samples += (predicted_classes == true_classes).sum().item()
                    
                    # Update per-class statistics for precision/recall/F1
                    for class_idx in range(num_classes):
                        # True positives: predicted and actual both equal class_idx
                        tp_mask = (predicted_classes == class_idx) & (true_classes == class_idx)
                        class_tp[class_idx] += tp_mask.sum()
                        
                        # False positives: predicted class_idx but actual is different
                        fp_mask = (predicted_classes == class_idx) & (true_classes != class_idx)
                        class_fp[class_idx] += fp_mask.sum()
                        
                        # False negatives: actual class_idx but predicted different
                        fn_mask = (predicted_classes != class_idx) & (true_classes == class_idx)
                        class_fn[class_idx] += fn_mask.sum()
                        
                else:
                    # Binary classification: apply threshold to sigmoid output
                    probabilities = torch.sigmoid(logits).squeeze()  # (batch_size,)
                    predicted_binary = (probabilities > threshold).long()  # (batch_size,)
                    
                    # Handle different label formats (squeeze to ensure 1D)
                    if batch_labels.dim() > 1:
                        true_binary = batch_labels.squeeze().long()
                    else:
                        true_binary = batch_labels.long()
                    
                    # Count correct predictions for accuracy
                    correct_samples += (predicted_binary == true_binary).sum().item()
                    
                    # Update binary classification statistics
                    tp += ((predicted_binary == 1) & (true_binary == 1)).sum().item()
                    fp += ((predicted_binary == 1) & (true_binary == 0)).sum().item()
                    tn += ((predicted_binary == 0) & (true_binary == 0)).sum().item()
                    fn += ((predicted_binary == 0) & (true_binary == 1)).sum().item()
                    
        finally:
            # Restore model to original device
            if device != original_device:
                self.model.to(original_device)
        
        # Compute final metrics
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
        
        if self.task_type == "multiclass":
            # Compute per-class precision, recall, F1, then macro-average
            class_precisions = []
            class_recalls = []
            class_f1s = []
            
            for class_idx in range(num_classes):
                # Precision: TP / (TP + FP), handle division by zero
                if (class_tp[class_idx] + class_fp[class_idx]) > 0:
                    precision = class_tp[class_idx] / (class_tp[class_idx] + class_fp[class_idx])
                else:
                    precision = 0.0
                    
                # Recall: TP / (TP + FN), handle division by zero  
                if (class_tp[class_idx] + class_fn[class_idx]) > 0:
                    recall = class_tp[class_idx] / (class_tp[class_idx] + class_fn[class_idx])
                else:
                    recall = 0.0
                    
                # F1: harmonic mean of precision and recall
                if (precision + recall) > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                    
                class_precisions.append(float(precision))
                class_recalls.append(float(recall))
                class_f1s.append(float(f1))
            
            # Macro-average across all classes
            macro_precision = sum(class_precisions) / len(class_precisions)
            macro_recall = sum(class_recalls) / len(class_recalls)
            macro_f1 = sum(class_f1s) / len(class_f1s)
            
        else:
            # Binary classification metrics
            # Precision: TP / (TP + FP)
            macro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall: TP / (TP + FN)  
            macro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1: harmonic mean
            if (macro_precision + macro_recall) > 0:
                macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
            else:
                macro_f1 = 0.0
        
        # Return comprehensive metrics dictionary
        return {
            'accuracy': float(accuracy),
            'precision': float(macro_precision), 
            'recall': float(macro_recall),
            'f1': float(macro_f1),
            'total_samples': int(total_samples),
            'correct_samples': int(correct_samples)
        }
