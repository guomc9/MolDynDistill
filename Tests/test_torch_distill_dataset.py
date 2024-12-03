import sys
sys.path.append('.')
import torch
from torch_geometric.loader import DataLoader
from utils.dataset import DistillDataset, get_dataset, split_dataset


def test_to_torch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    molecular = 'benzene'
    num_atoms = 12
    batch_size = 1
    dataset = get_dataset(dataset_name='MD17', root='data/MD17', name=molecular)
    train_dataset, _, _ = split_dataset(dataset=dataset, train_size=1000, valid_size=1000, seed=42)
    source_dataset = DistillDataset(source_dataset=train_dataset, distill_rate=0.6, distill_lr=3.0e-10, device=device)
    
    
    torch_dataset = source_dataset.to_torch()
    
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    
    for batch_idx, data in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f'z: {data.z}')
        print(f"  z.shape: {data.z.shape}")
        print(f"  pos.shape: {data.pos.shape}")
        print(f"  y.shape (energy): {data.y.shape}")
        print(f"  force.shape: {data.force.shape}")
        assert data.z.shape[0] == num_atoms * batch_size
        assert data.pos.shape[0] == num_atoms * batch_size
        assert data.y.shape[0] == batch_size
        assert data.force.shape[0] == num_atoms * batch_size

    print("Test passed!")

if __name__ == '__main__':
    test_to_torch()