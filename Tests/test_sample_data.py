import torch
from torch_geometric.datasets import MD17
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

dataset = MD17(root='data/MD17', name='benzene')
ids = np.arange(0, len(dataset))
np.random.shuffle(ids)
train_indices = ids[:1000]
train_dataset = dataset[train_indices]
sampled_indices = torch.randperm(1000)[:600]
sampled_dataset = train_dataset[sampled_indices]
# sampled_dataset = [train_dataset[i] for i in sampled_indices]

saved_data = {
    'pos': sampled_dataset.pos,
    'y': sampled_dataset.y,
    'force': sampled_dataset.force, 
    'z': sampled_dataset.z, 
    'distill_lr': torch.tensor(1e-9)
}

save_dir = './'

save_path = os.path.join(save_dir, 'check.pt')
torch.save(saved_data, save_path)

loaded_data = torch.load(save_path)
print("保存的数据形状:")
for key, tensor in loaded_data.items():
    print(f"{key}: {tensor.shape}")