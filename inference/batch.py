"""
Batch inference utilities for processing multiple audio files.
"""
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
import librosa


class BatchInference:
    """Process multiple audio files with a keyword spotting model."""
    
    def __init__(self, model, verbose: bool = True):
        """
        Initialize batch inference.
        
        Args:
            model: KeywordModel instance
            verbose: Show progress bar
        """
        self.model = model
        self.verbose = verbose
    
    def process_directory(
        self,
        audio_dir: str,
        output_path: str,
        pattern: str = "*.wav",
        top_k: int = 3,
        output_format: str = "csv"
    ) -> List[Dict]:
        """
        Process all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            output_path: Path to save results
            pattern: File glob pattern (e.g., "*.wav", "*.mp3")
            top_k: Number of top predictions per file
            output_format: "csv" or "json"
            
        Returns:
            List of result dicts
        """
        audio_dir = Path(audio_dir)
        audio_files = sorted(audio_dir.glob(pattern))
        
        if not audio_files:
            raise ValueError(f"No files matching '{pattern}' found in {audio_dir}")
        
        results = []
        iterator = tqdm(audio_files, desc="Processing") if self.verbose else audio_files
        
        for audio_path in iterator:
            try:
                # Load audio
                y, sr = librosa.load(str(audio_path), sr=self.model.sr, mono=True)
                
                # Predict
                predictions = self.model.predict_audio(y, top_k=top_k)
                
                result = {
                    "file": audio_path.name,
                    "path": str(audio_path),
                    "duration_s": len(y) / sr,
                    "predictions": predictions
                }
                results.append(result)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {audio_path.name}: {e}")
                results.append({
                    "file": audio_path.name,
                    "path": str(audio_path),
                    "error": str(e)
                })
        
        # Save results
        self.save_results(results, output_path, output_format)
        return results
    
    def process_file_list(
        self,
        file_list: List[str],
        output_path: str,
        top_k: int = 3,
        output_format: str = "csv"
    ) -> List[Dict]:
        """
        Process a list of audio files.
        
        Args:
            file_list: List of audio file paths
            output_path: Path to save results
            top_k: Number of top predictions per file
            output_format: "csv" or "json"
            
        Returns:
            List of result dicts
        """
        results = []
        iterator = tqdm(file_list, desc="Processing") if self.verbose else file_list
        
        for audio_path in iterator:
            try:
                y, sr = librosa.load(audio_path, sr=self.model.sr, mono=True)
                predictions = self.model.predict_audio(y, top_k=top_k)
                
                result = {
                    "file": os.path.basename(audio_path),
                    "path": audio_path,
                    "duration_s": len(y) / sr,
                    "predictions": predictions
                }
                results.append(result)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {audio_path}: {e}")
                results.append({
                    "file": os.path.basename(audio_path),
                    "path": audio_path,
                    "error": str(e)
                })
        
        self.save_results(results, output_path, output_format)
        return results
    
    def save_results(self, results: List[Dict], output_path: str, format: str = "csv"):
        """Save results to file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        elif format == "csv":
            # Flatten predictions for CSV
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "path", "duration_s", "top1_class", "top1_prob", 
                                "top2_class", "top2_prob", "top3_class", "top3_prob"])
                
                for r in results:
                    if "error" in r:
                        continue
                    preds = r["predictions"]
                    row = [r["file"], r["path"], f"{r['duration_s']:.3f}"]
                    for i in range(3):
                        if i < len(preds):
                            row.extend([preds[i][0], f"{preds[i][1]:.4f}"])
                        else:
                            row.extend(["", ""])
                    writer.writerow(row)
        
        print(f"[OK] Results saved to {output_path}")


def filter_results(
    results: List[Dict],
    min_confidence: float = 0.5,
    target_classes: Optional[List[str]] = None
) -> List[Dict]:
    """
    Filter results by confidence and/or target classes.
    
    Args:
        results: List of prediction results
        min_confidence: Minimum confidence threshold
        target_classes: List of target class names (None = all classes)
        
    Returns:
        Filtered results
    """
    filtered = []
    for r in results:
        if "error" in r or "predictions" not in r:
            continue
        
        top_pred = r["predictions"][0]
        if top_pred[1] >= min_confidence:
            if target_classes is None or top_pred[0] in target_classes:
                filtered.append(r)
    
    return filtered
