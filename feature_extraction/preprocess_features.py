# scripts/preprocess_features.py

import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from feature_extraction.extract_features import extract_features
from config import load_config
import time


def preprocess_and_save_all_features(
    input_dir,
    output_dir,
    sr,
    n_features,
    frame_length_s,
    hop_length_s,
    fixed_duration_s,
    feature_type="mfcc",
    normalize=False,
    limit_per_class=None,
):
    hop_samples   = int(hop_length_s  * sr)
    frame_samples = int(frame_length_s * sr)
    fixed_len     = int(fixed_duration_s * sr)

    # Rough expected frames (librosa center=True by default pads at edges)
    approx_frames = int(np.floor((fixed_len - frame_samples) / hop_samples) + 1)

    print(f"[Features] Type: {feature_type}")
    print(f"[Features] Input raw dir: {input_dir}")
    print(f"[Features] Output preprocessed dir: {output_dir}")
    print(f"[Features] sr={sr}, n_features={n_features}, frame_len={frame_length_s}s, hop={hop_length_s}s, fixed_dur={fixed_duration_s}s")
    print(f"[Features] Normalize: {normalize}")
    print(f"[Features] ~Expected frames: {approx_frames}")

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    labels = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    labels.sort()

    # Count total files for progress tracking
    total_files = 0
    for label in labels:
        label_path = os.path.join(input_dir, label)
        wav_files = [f for f in os.listdir(label_path) if f.lower().endswith(".wav")]
        if limit_per_class is not None:
            wav_files = wav_files[:limit_per_class]
        total_files += len(wav_files)
    
    print(f"[Features] Total files to process: {total_files}")
    
    # Start timing
    start_time = time.time()
    processed = 0
    failed = 0

    # Overall progress bar
    with tqdm(total=total_files, desc="Overall Progress", unit="file") as pbar:
        for label in labels:
            label_path = os.path.join(input_dir, label)
            out_label_dir = os.path.join(output_dir, label)
            os.makedirs(out_label_dir, exist_ok=True)

            wav_files = [f for f in os.listdir(label_path) if f.lower().endswith(".wav")]
            wav_files.sort()
            
            # Limit files per class if specified
            if limit_per_class is not None:
                wav_files = wav_files[:limit_per_class]

            for fname in wav_files:
                in_path = os.path.join(label_path, fname)
                out_path = os.path.join(out_label_dir, fname.replace(".wav", ".npy"))

                try:
                    # Load and pad/truncate audio
                    y, _ = librosa.load(in_path, sr=sr)
                    if len(y) < fixed_len:
                        y = np.pad(y, (0, fixed_len - len(y)), mode="constant")
                    else:
                        y = y[:fixed_len]

                    # Normalize audio time samples if enabled (AGC behavior - normalize input signal)
                    if normalize:
                        sample_max = np.abs(y).max()
                        if sample_max > 0:
                            y = y / sample_max

                    # Extract features using unified function with the (potentially normalized) audio
                    _, features = extract_features(
                        y, sr=sr, n_features=n_features,  # Pass audio array instead of file path
                        frame_length=frame_length_s, hop_length=hop_length_s,
                        feature_type=feature_type
                    )
                    
                    # Save as (time, n_features)
                    np.save(out_path, features)
                    processed += 1
                except Exception as e:
                    failed += 1
                    tqdm.write(f"[WARN] Failed: {in_path} -> {e}")
                finally:
                    pbar.update(1)
                    # Update description with current label
                    pbar.set_postfix(label=label, failed=failed)
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n[Summary]")
    print(f"  Total processed: {processed}/{total_files}")
    print(f"  Failed: {failed}")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if processed > 0:
        print(f"  Avg time per file: {elapsed/processed:.3f}s")



def main():
    ap = argparse.ArgumentParser(description="Preprocess raw audio into feature numpy files using YAML config.")
    ap.add_argument("--config", type=str, default="config/base.yaml",
                    help="Path to YAML config file.")
    # Optional quick overrides (leave unset to use YAML)
    ap.add_argument("--raw-dir", type=str, default=None, help="Override data.raw_dir")
    ap.add_argument("--out-dir", type=str, default=None, help="Override data.preprocessed_dir")
    ap.add_argument("--limit", type=int, default=None, help="Process only N files per class (for testing)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Pull paths and feature params from config
    input_dir = args.raw_dir or cfg["data"].get("raw_dir", "data/speech_commands_v0.02")
    output_dir = args.out_dir or cfg["data"]["preprocessed_dir"]

    feat_cfg = cfg["data"].get("features", {})
    feature_type     = feat_cfg.get("type", "mfcc")
    sr               = int(feat_cfg.get("sample_rate",       16000))
    n_features       = int(feat_cfg.get("n_features",        16))
    frame_length_s   = float(feat_cfg.get("frame_length_s",  0.02))
    hop_length_s     = float(feat_cfg.get("hop_length_s",    0.01))
    fixed_duration_s = float(feat_cfg.get("fixed_duration_s", 1.0))
    normalize        = bool(feat_cfg.get("normalize",        False))

    preprocess_and_save_all_features(
        input_dir=input_dir,
        output_dir=output_dir,
        sr=sr,
        n_features=n_features,
        frame_length_s=frame_length_s,
        hop_length_s=hop_length_s,
        fixed_duration_s=fixed_duration_s,
        feature_type=feature_type,
        normalize=normalize,
        limit_per_class=args.limit,
    )

    print(f"[OK] Done! {feature_type.upper()} features saved in {output_dir}")


if __name__ == "__main__":
    main()
