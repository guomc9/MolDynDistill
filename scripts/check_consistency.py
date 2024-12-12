import sys
sys.path.append('.')
import argparse
import matplotlib.pyplot as plt
from utils.dataset import get_dataset, split_dataset
import os

def main(args):
    dataset = get_dataset(dataset_name='MD17', root='data/MD17', name=args.molecular)

    train_dataset, valid_dataset, _ = split_dataset(
        dataset=dataset,
        train_size=args.train_size,
        valid_size=args.valid_size,
        seed=args.seed,
        shuffle=False
    )

    train_energies = [sample['energy'] for sample in train_dataset]
    valid_energies = [sample['energy'] for sample in valid_dataset]

    plt.figure(figsize=(12, 6))

    plt.plot(range(len(train_energies)), train_energies, label='Train Energies', marker='o', linestyle='-', alpha=0.7)

    plt.plot(range(len(train_energies), len(train_energies) + len(valid_energies)),
             valid_energies, label='Validation Energies', marker='o', linestyle='-', alpha=0.7)


    plt.legend()
    plt.title(f'Continuous Sampling Molecular Energies for {args.molecular.capitalize()}')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.output_file)
    print(f"Plot saved as {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot molecular energies with continuous sampling.')
    parser.add_argument('--molecular', type=str, help='Name of the molecular (e.g., benzene).', default='benzene')
    parser.add_argument('--train_size', type=int, help='Number of training samples.', default=1000)
    parser.add_argument('--valid_size', type=int, help='Number of validation samples.', default=1000)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset splitting.')
    parser.add_argument('--output_file', type=str, default=None, help='Output PNG file name.')

    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = os.path.join('data/MD17', args.molecular, f'consistency-size{args.train_size}-seed{args.seed}.png')
    main(args)