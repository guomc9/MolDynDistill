import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse


def extract_metrics_from_checkpoints(checkpoints_dir):
    """
    Extract metrics (best_valid_mae, best_valid_force_mae, best_valid_energy_mae)
    from all checkpoint files in the given directory.

    Args:
        checkpoints_dir (str): Path to the directory containing checkpoint files.

    Returns:
        dict: A dictionary with keys 'idx', 'best_valid_mae', 'best_valid_force_mae', 
              and 'best_valid_energy_mae', containing corresponding metric values.
    """
    metrics = {
        "idx": [],
        "best_valid_mae": [],
        "best_valid_force_mae": [],
        "best_valid_energy_mae": []
    }

    # Iterate over all files in the directory
    for file_name in os.listdir(checkpoints_dir):
        if file_name.startswith("checkpoint_epoch_") and file_name.endswith(".pt"):
            # Extract epoch index (idx) from the file name
            idx = int(file_name.split("_")[-1].split(".")[0])
            file_path = os.path.join(checkpoints_dir, file_name)

            # Load the checkpoint file
            checkpoint = torch.load(file_path, map_location="cpu")
            
            # Extract metrics from the checkpoint
            best_valid_mae = checkpoint.get("best_valid_mae", None)
            best_valid_force_mae = checkpoint.get("best_valid_force_mae", None)
            best_valid_energy_mae = checkpoint.get("best_valid_energy_mae", None)

            # Only add metrics if all are available
            if best_valid_mae is not None and best_valid_force_mae is not None and best_valid_energy_mae is not None:
                metrics["idx"].append(idx)
                metrics["best_valid_mae"].append(best_valid_mae)
                metrics["best_valid_force_mae"].append(best_valid_force_mae)
                metrics["best_valid_energy_mae"].append(best_valid_energy_mae)
    
    # Sort metrics by idx
    sorted_indices = sorted(range(len(metrics["idx"])), key=lambda k: metrics["idx"][k])
    metrics = {key: [metrics[key][i] for i in sorted_indices] for key in metrics}

    return metrics


def find_checkpoints_dirs(base_dir):
    """
    Find all subdirectories under the given base directory.

    Args:
        base_dir (str): Path to the base directory.

    Returns:
        list: A list of checkpoint subdirectories.
    """
    checkpoints_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # Add directories containing checkpoint files
        if any(file.startswith("checkpoint_epoch_") and file.endswith(".pt") for file in files):
            checkpoints_dirs.append(root)
    return checkpoints_dirs


def plot_metrics_and_save(checkpoints_dirs, base_dir, y_min, y_max):
    """
    Plot metrics (best_valid_mae, best_valid_force_mae, best_valid_energy_mae) 
    from multiple checkpoint directories and save them as PNG files.

    Args:
        checkpoints_dirs (list): List of checkpoint directory paths.
        base_dir (str): Base directory where the plots will be saved.
        y_min (float): Minimum value of the y-axis.
        y_max (float): Maximum value of the y-axis.
    """
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "pink"]
    labels = ["best_valid_mae", "best_valid_force_mae", "best_valid_energy_mae"]
    png_filenames = ["best_valid_mae.png", "best_valid_force_mae.png", "best_valid_energy_mae.png"]

    # Initialize subplots for the three metrics
    for j, label in enumerate(labels):
        plt.figure(figsize=(10, 5))
        for i, checkpoints_dir in enumerate(checkpoints_dirs):
            # Extract metrics for the current directory
            metrics = extract_metrics_from_checkpoints(checkpoints_dir)

            # Plot each metric
            plt.plot(metrics["idx"], metrics[label], label=f"{os.path.basename(checkpoints_dir)}", color=colors[i % len(colors)])
        
        # Set plot title, labels, and legend
        plt.title(label)
        plt.xlabel("Epoch Index (idx)")
        plt.ylabel(label)
        plt.legend()

        # Set y-axis range and finer intervals
        if y_min is not None and y_max is not None:
            plt.gca().set_ylim(y_min, y_max)  # Set y-axis range
            interval = (y_max - y_min) / 5  # Set major tick interval dynamically
            plt.gca().yaxis.set_major_locator(MultipleLocator(interval))  # Major ticks
            plt.gca().yaxis.set_minor_locator(MultipleLocator(interval / 5))  # Minor ticks

        # Save the plot to base_dir
        save_path = os.path.join(base_dir, png_filenames[j])
        plt.savefig(save_path)
        print(f"Saved {label} plot to {save_path}")

        # Close the figure to free up memory
        plt.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot metrics from checkpoint directories and save them.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing checkpoint subdirectories.")
    parser.add_argument("--y_min", type=float, default=None, help="Minimum value of the y-axis.")
    parser.add_argument("--y_max", type=float, default=None, help="Maximum value of the y-axis.")
    args = parser.parse_args()

    # Find all checkpoint directories under the base directory
    checkpoints_dirs = find_checkpoints_dirs(args.base_dir)

    # Check if any checkpoint directories were found
    if not checkpoints_dirs:
        print(f"No checkpoint directories found in {args.base_dir}")
    else:
        print(f"Found {len(checkpoints_dirs)} checkpoint directories.")
        for dir_path in checkpoints_dirs:
            print(f"  - {dir_path}")

        # Plot metrics and save as PNG files
        plot_metrics_and_save(checkpoints_dirs, args.base_dir, args.y_min, args.y_max)