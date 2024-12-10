import torch
import argparse

def compare_datasets(path1, path2):
    data1 = torch.load(path1)
    data2 = torch.load(path2)
    
    print(f"Comparing {path1} and {path2}")
    print("\nDifference Statistics:")
    print("-" * 80)
    print(f"{'Key':<10} {'Shape':<15} {'Max Diff':>12} {'Min Diff':>12} {'Mean Diff':>12}")
    print("-" * 80)
    
    for key in data1.keys():
        if key not in data2:
            print(f"{key:<10} Not found in second file")
            continue
            
        tensor1 = data1[key]
        tensor2 = data2[key]
        
        if not isinstance(tensor1, torch.Tensor):
            continue
            
        if tensor1.shape != tensor2.shape:
            print(f"{key:<10} Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            continue
        
        diff = (tensor1 - tensor2).abs().float()
        
        print(f"{key:<10} {str(tensor1.shape):<15} "
              f"{diff.max().item():>12.10e} "
              f"{diff.min().item():>12.10e} "
              f"{diff.mean().item():>12.10e}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two PyTorch saved files')
    parser.add_argument('-x', '--path1', type=str, help='Path to first .pt file')
    parser.add_argument('-y', '--path2', type=str, help='Path to second .pt file')
    
    args = parser.parse_args()
    compare_datasets(args.path1, args.path2)