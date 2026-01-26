# scripts/preprocess_features.py

import os
import argparse
from copy import deepcopy
import numpy as np
import librosa
from tqdm import tqdm
import yaml
from feature_extraction.extract_features import extract_features


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml_with_encodings(path):
    import yaml
    # Try UTF-8 first; then UTF-8 with BOM; then cp1252 as a last resort.
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    # If we get here, re-raise with a helpful error
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Could not decode {path} with utf-8/utf-8-sig/cp1252")

def load_config(path):
    cfg_task = _load_yaml_with_encodings(path)

    # Optional base.yaml merge (same pattern as train.py)
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml_with_encodings(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task


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

    # For normalization: collect global max if normalize=True
    global_max = None
    if normalize:
        print("[Features] Computing global max for normalization (pass 1)...")
        all_maxs = []
        for label in labels:
            label_path = os.path.join(input_dir, label)
            wav_files = [f for f in os.listdir(label_path) if f.lower().endswith(".wav")]
            for fname in tqdm(wav_files, desc=f"Scanning '{label}'", leave=False):
                try:
                    in_path = os.path.join(label_path, fname)
                    y, _ = librosa.load(in_path, sr=sr)
                    if len(y) < fixed_len:
                        y = np.pad(y, (0, fixed_len - len(y)), mode="constant")
                    else:
                        y = y[:fixed_len]
                    _, features = extract_features(
                        in_path, sr=sr, n_features=n_features,
                        frame_length=frame_length_s, hop_length=hop_length_s,
                        feature_type=feature_type
                    )
                    all_maxs.append(np.abs(features).max())
                except Exception as e:
                    pass
        global_max = np.max(all_maxs) if all_maxs else 1.0
        print(f"[Features] Global max: {global_max:.4f}")

    for label in labels:
        label_path = os.path.join(input_dir, label)
        out_label_dir = os.path.join(output_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)

        wav_files = [f for f in os.listdir(label_path) if f.lower().endswith(".wav")]
        wav_files.sort()

        for fname in tqdm(wav_files, desc=f"Processing '{label}'", leave=False):
            in_path = os.path.join(label_path, fname)
            out_path = os.path.join(out_label_dir, fname.replace(".wav", ".npy"))

            try:
                # Load and pad/truncate audio
                y, _ = librosa.load(in_path, sr=sr)
                if len(y) < fixed_len:
                    y = np.pad(y, (0, fixed_len - len(y)), mode="constant")
                else:
                    y = y[:fixed_len]

                # Extract features using unified function
                _, features = extract_features(
                    in_path, sr=sr, n_features=n_features,
                    frame_length=frame_length_s, hop_length=hop_length_s,
                    feature_type=feature_type
                )
                
                # Normalize if enabled
                if normalize and global_max is not None and global_max > 0:
                    features = features / global_max
                
                # Save as (time, n_features)
                np.save(out_path, features)
            except Exception as e:
                print(f"[WARN] Failed: {in_path} -> {e}")


def main():
    ap = argparse.ArgumentParser(description="Preprocess raw audio into feature numpy files using YAML config.")
    ap.add_argument("--config", type=str, default="config/base.yaml",
                    help="Path to YAML config file.")
    # Optional quick overrides (leave unset to use YAML)
    ap.add_argument("--raw-dir", type=str, default=None, help="Override data.raw_dir")
    ap.add_argument("--out-dir", type=str, default=None, help="Override data.preprocessed_dir")
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
    )

    print(f"[OK] Done! {feature_type.upper()} features saved in {output_dir}")


if __name__ == "__main__":
    main()
