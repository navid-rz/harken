import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


def analyze_feature_ranges(data_dir="data/preprocessed", feature_type="log-mel"):
    """
    Load all feature files (MFCC or log-mel) and analyze their min/max values per label.
    Plots histograms of min and max values for each word.
    
    Args:
        data_dir: Path to preprocessed feature data
        feature_type: Type of features ('mfcc', 'log-mel', or 'auto' for automatic detection)
    """
    # Auto-detect feature type from directory name if not specified
    if feature_type == "auto":
        if "log_mel" in data_dir.lower() or "logmel" in data_dir.lower():
            feature_type = "log-mel"
        else:
            feature_type = "mfcc"
    
    feature_name = feature_type.upper()
    print(f"Analyzing {feature_name} feature ranges in: {data_dir}\n")
    
    # Dictionary to store min/max values per label
    label_stats = defaultdict(lambda: {"mins": [], "maxs": [], "time_lengths": []})
    
    # Track time lengths globally
    all_time_lengths = []
    
    # Iterate through all label directories
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for label in labels:
        label_path = os.path.join(data_dir, label)
        npy_files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
        
        for fname in tqdm(npy_files, desc=f"Processing '{label}'", leave=False):
            file_path = os.path.join(label_path, fname)
            try:
                features = np.load(file_path)
                time_length = features.shape[0]  # First dimension is time
                label_stats[label]["mins"].append(features.min())
                label_stats[label]["maxs"].append(features.max())
                label_stats[label]["time_lengths"].append(time_length)
                all_time_lengths.append(time_length)
            except Exception as e:
                print(f"[WARN] Failed to load {file_path}: {e}")
    
    # Compute global statistics
    all_mins = []
    all_maxs = []
    for label in labels:
        all_mins.extend(label_stats[label]["mins"])
        all_maxs.extend(label_stats[label]["maxs"])
    
    print(f"\n=== Global {feature_name} Statistics ===")
    print(f"Total files analyzed: {len(all_mins)}")
    print(f"Global min across all files: {np.min(all_mins):.2f}")
    print(f"Global max across all files: {np.max(all_maxs):.2f}")
    print(f"Mean of mins: {np.mean(all_mins):.2f} ± {np.std(all_mins):.2f}")
    print(f"Mean of maxs: {np.mean(all_maxs):.2f} ± {np.std(all_maxs):.2f}")
    
    # Check time length consistency
    unique_lengths = set(all_time_lengths)
    print(f"\n=== Time Length Analysis ===")
    print(f"Unique time lengths found: {sorted(unique_lengths)}")
    if len(unique_lengths) == 1:
        print(f"✅ ALL FILES HAVE CONSISTENT TIME LENGTH: {list(unique_lengths)[0]} frames")
    else:
        print(f"❌ INCONSISTENT TIME LENGTHS - found {len(unique_lengths)} different lengths!")
        length_counts = {length: all_time_lengths.count(length) for length in unique_lengths}
        for length, count in sorted(length_counts.items()):
            print(f"  Length {length}: {count} files ({count/len(all_time_lengths)*100:.1f}%)")
    
    print(f"Min time length: {np.min(all_time_lengths)}")
    print(f"Max time length: {np.max(all_time_lengths)}")
    print(f"Mean time length: {np.mean(all_time_lengths):.1f} ± {np.std(all_time_lengths):.1f}")
    
    # Plot overall distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(all_mins, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title(f"Distribution of Minimum {feature_name} Values (All Files)")
    axes[0].set_xlabel(f"Min {feature_name} value")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(np.mean(all_mins), color='red', linestyle='--', label=f'Mean: {np.mean(all_mins):.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_maxs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title(f"Distribution of Maximum {feature_name} Values (All Files)")
    axes[1].set_xlabel(f"Max {feature_name} value")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(np.mean(all_maxs), color='blue', linestyle='--', label=f'Mean: {np.mean(all_maxs):.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_filename = f"plots/{feature_type.replace('-', '_')}_ranges_global.png"
    plt.savefig(output_filename, dpi=150)
    print(f"[OK] Plot saved to: {output_filename}")
    plt.show()
    
    return label_stats, all_mins, all_maxs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze feature value ranges across dataset (MFCC or log-mel)")
    parser.add_argument("--data-dir", type=str, default="data/preprocessed",
                        help="Path to preprocessed feature data")
    parser.add_argument("--feature-type", type=str, default="auto", choices=["auto", "mfcc", "log-mel"],
                        help="Feature type: 'mfcc', 'log-mel', or 'auto' for automatic detection (default: auto)")
    args = parser.parse_args()
    
    os.makedirs("plots", exist_ok=True)
    analyze_feature_ranges(args.data_dir, args.feature_type)
