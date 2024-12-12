import torch
import os
import re
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_param_diff(file1, file2):
    params1 = torch.load(file1)['model_state_dict']
    params2 = torch.load(file2)['model_state_dict']
    
    total_diff = 0
    param_count = 0
    
    for key in params1.keys():
        if key in params2:
            diff = torch.abs(params1[key] - params2[key]).mean().item()
            total_diff += diff
            param_count += 1
    
    return total_diff / param_count if param_count > 0 else 0

def non_uniform_sampling(files, high_freq_count=50, low_freq_step=5):
    """
    Perform non-uniform sampling on the list of files.
    :param files: Sorted list of checkpoint files.
    :param high_freq_count: Number of files to sample with high frequency at the start.
    :param low_freq_step: Step size for low-frequency sampling in the remaining files.
    :return: Subset of files after non-uniform sampling.
    """
    if len(files) <= high_freq_count:
        return files  # If total files are fewer than high_freq_count, return all.
    
    # High-frequency sampling for the first `high_freq_count` files
    high_freq_files = files[:high_freq_count]
    
    # Low-frequency sampling for the remaining files
    low_freq_files = files[high_freq_count::low_freq_step]
    
    # Combine both
    return high_freq_files + low_freq_files

def plot_diff_line_chart(results, output_file):
    # Extract iteration numbers and parameter differences
    iter_pairs = [f"{res['iter1']}→{res['iter2']}" for res in results]
    param_diffs = [res['parameter_diff'] for res in results]
    
    # Dynamically set tick step for X-axis based on the number of data points
    num_points = len(iter_pairs)
    tick_step = max(1, num_points // 10)  # Show 10 ticks at most

    # Determine max Y value and create uneven ticks
    # max_y = max(param_diffs) * 1.1  # Add 10% margin for better visualization
    # min_y = min(param_diffs) * 0.9  # Add 10% margin for better visualization
    y_ticks = [1 / (10 ** i) for i in range(6)]  # Create 1/10 uneven ticks
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(iter_pairs)), param_diffs, marker='o', label='Parameter Difference', color='b')
    plt.xticks(
        ticks=range(0, num_points, tick_step),  # Only show ticks at intervals of `tick_step`
        labels=[iter_pairs[i] for i in range(0, num_points, tick_step)],  # Corresponding labels
        rotation=45,
        fontsize=8
    )
    # Set the Y-axis ticks for uneven spacing
    plt.yticks(ticks=y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
    # plt.ylim(min_y, max_y)  # Ensure Y-axis starts at 0 and ends at max_y
    plt.yscale('log')

    plt.xlabel('Iteration Pairs')
    plt.ylabel('Parameter Difference')
    plt.title('Parameter Differences Between Checkpoints')
    plt.grid(alpha=0.5)
    plt.legend()
    
    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

def filter_files_by_iteration(files, prefix, iter_start, iter_end):
    """
    Filter files based on the iteration range.
    :param files: List of checkpoint files.
    :param prefix: Prefix of the checkpoint files.
    :param iter_start: Start iteration (inclusive).
    :param iter_end: End iteration (inclusive).
    :return: Filtered list of files sorted by iteration number.
    """
    filtered_files = []
    iter_numbers = []
    
    for file in files:
        match = re.search(rf'{prefix}(\d+)\.pt', file)
        if match:
            iteration = int(match.group(1))
            if (iter_start is None or iteration >= iter_start) and (iter_end is None or iteration <= iter_end):
                filtered_files.append(file)
                iter_numbers.append(iteration)
    
    # Sort files by their extracted iteration numbers
    sorted_indices = np.argsort(iter_numbers)
    return [filtered_files[i] for i in sorted_indices], sorted(iter_numbers)

def main(args):
    files = [f for f in os.listdir(args.input_dir) if f.startswith(args.prefix) and f.endswith('.pt')]
    
    # Filter files based on iter_start and iter_end
    files, iter_numbers = filter_files_by_iteration(files, args.prefix, args.iter_start, args.iter_end)
    
    if not files:
        print("No files found in the specified iteration range.")
        return
    
    # Apply non-uniform sampling
    sampled_files = non_uniform_sampling(files, args.high_freq_count, args.low_freq_step)
    sampled_iters = [int(re.search(rf'{args.prefix}(\d+)\.pt', f).group(1)) for f in sampled_files]
    
    results = []
    
    for i in range(len(sampled_files) - 1):
        file1 = os.path.join(args.input_dir, sampled_files[i])
        file2 = os.path.join(args.input_dir, sampled_files[i + 1])
        
        iter1 = sampled_iters[i]
        iter2 = sampled_iters[i + 1]
        
        diff = get_param_diff(file1, file2)
        
        results.append({
            'iter1': iter1,
            'iter2': iter2,
            'parameter_diff': diff
        })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Generate and save the plot
    plot_output_file = os.path.splitext(args.output)[0] + '.png'
    plot_diff_line_chart(results, plot_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='.', help='Directory containing checkpoint files')
    parser.add_argument('-p', '--prefix', type=str, default='checkpoint_iters_', help='Prefix of checkpoint files')
    parser.add_argument('-o', '--output', type=str, default='parameter_differences.csv', help='Output CSV file path')
    parser.add_argument('--high_freq_count', type=int, default=50, help='Number of high-frequency samples at the start')
    parser.add_argument('--low_freq_step', type=int, default=5, help='Step size for low-frequency sampling')
    parser.add_argument('--iter_start', type=int, default=None, help='Start iteration (inclusive)')
    parser.add_argument('--iter_end', type=int, default=None, help='End iteration (inclusive)')
    args = parser.parse_args()
    main(args)