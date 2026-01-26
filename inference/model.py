"""
Model wrapper for keyword spotting inference.
"""
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from typing import List, Tuple

from config import load_config
from train.utils import load_state_dict_forgiving
from model.model import DilatedTCN


def get_mfcc_section(cfg: dict) -> dict:
    """
    Locate MFCC settings in config. Supports:
      cfg['mfcc']
      cfg['data']['mfcc']
      cfg['features']['mfcc']
      cfg['audio']['mfcc']
    """
    return (
        cfg.get("mfcc")
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

    n_mfcc = int(cfg_mfcc["n_mfcc"])
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
