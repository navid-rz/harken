import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Optional, Literal


def extract_features(
    filepath_or_audio, 
    sr=16000, 
    n_features=16,
    frame_length=0.02, 
    hop_length=0.01, 
    center=True,
    feature_type: Literal["mfcc", "log-mel"] = "mfcc"
):
    """
    Extract audio features from a file or audio array - either MFCC or log-mel spectrogram.
    
    Args:
        filepath_or_audio (str or np.ndarray): Path to .wav file or audio data array.
        sr (int): Target sampling rate.
        n_features (int): Number of features (MFCCs or mel bands).
        frame_length (float): Window size in seconds.
        hop_length (float): Hop size in seconds.
        center (bool): Whether to center frames (librosa default).
        feature_type (str): Either "mfcc" or "log-mel".
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: waveform (1D), features (2D: time x n_features)
    """
    if isinstance(filepath_or_audio, str):
        # Load from file
        y, sr = librosa.load(filepath_or_audio, sr=sr)
    else:
        # Use provided audio array
        y = np.asarray(filepath_or_audio)
    
    n_fft = int(frame_length * sr)
    hop_samples = int(hop_length * sr)
    
    if feature_type == "mfcc":
        # Traditional MFCCs (can be negative due to DCT)
        features = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_features,
            n_fft=n_fft,
            hop_length=hop_samples,
            center=center
        )
    elif feature_type == "log-mel":
        # Log-mel spectrogram (all positive after proper scaling)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_features,
            n_fft=n_fft,
            hop_length=hop_samples,
            center=center
        )
        # Natural log (standard for ML, not dB scale)
        features = np.log(mel_spec + 1e-10)  # Add epsilon to avoid log(0)
        # Shift to [0, max] range for unsigned hardware compatibility
        features = features - features.min()
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Use 'mfcc' or 'log-mel'")
    
    return y, features.T


# Backward compatibility alias
def extract_mfcc(filepath, sr=16000, n_mfcc=16, frame_length=0.02, hop_length=0.01, center=True):
    """Legacy function for backward compatibility."""
    return extract_features(filepath, sr, n_mfcc, frame_length, hop_length, center, feature_type="mfcc")


def plot_features_and_waveform(y, features, sr=16000, hop_length=0.01, word: Optional[str] = None, feature_type: str = "mfcc"):
    """
    Plot waveform and features (MFCC or log-mel) side-by-side.
    
    Args:
        y (np.ndarray): Audio signal
        features (np.ndarray): Audio features (time x n_features)
        sr (int): Sampling rate
        hop_length (float): Hop length in seconds
        word (str, optional): Label for plot title
        feature_type (str): "mfcc" or "log-mel" for labeling
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot waveform
    t = np.linspace(0, len(y) / sr, num=len(y))
    axs[0].plot(t, y)
    axs[0].set_title(f"Waveform" + (f" — {word}" if word else ""))
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")

    # Plot features
    feature_label = "MFCC Coefficients" if feature_type == "mfcc" else "Mel Frequency Bands"
    im = axs[1].imshow(
        features.T, origin='lower', aspect='auto',
        extent=[0, features.shape[0]*hop_length, 0, features.shape[1]],
        cmap='viridis'
    )
    axs[1].set_title(f"{feature_type.upper()}" + (f" — {word}" if word else ""))
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel(feature_label)
    fig.colorbar(im, ax=axs[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()


# Backward compatibility alias
def plot_mfcc_and_waveform(y, mfcc, sr=16000, hop_length=0.01, word: Optional[str] = None):
    """Legacy function for backward compatibility."""
    return plot_features_and_waveform(y, mfcc, sr, hop_length, word, feature_type="mfcc")


# 🧪 Test script
if __name__ == "__main__":
    # Example file: download_gsc.py will place files under ./data/speech_commands_v0.02/
    sample_file = "./data/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"

    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
    else:
        word = os.path.basename(os.path.dirname(sample_file))
        
        # Test MFCC
        print("\n=== Testing MFCC ===")
        y, mfcc = extract_features(sample_file, feature_type="mfcc", n_features=16)
        print(f"MFCC shape: {mfcc.shape}")
        print(f"MFCC range: [{mfcc.min():.2f}, {mfcc.max():.2f}]")
        plot_features_and_waveform(y, mfcc, word=word, feature_type="mfcc")
        
        # Test log-mel
        print("\n=== Testing Log-Mel ===")
        y, logmel = extract_features(sample_file, feature_type="log-mel", n_features=40)
        print(f"Log-mel shape: {logmel.shape}")
        print(f"Log-mel range: [{logmel.min():.2f}, {logmel.max():.2f}]")
        plot_features_and_waveform(y, logmel, word=word, feature_type="log-mel")
