import torch
from torch_geometric.datasets import MD17
import numpy as np

# 加载MD17 benzene数据集
dataset = MD17(root='data/MD17', name='benzene')

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

total_len = len(dataset)

train_indices = torch.randperm(total_len)[:1000]
train_dataset = dataset[train_indices]

sampled_indices = np.random.choice(len(train_dataset), 600, replace=False)
sampled_dataset = train_dataset[sampled_indices]

for data in sampled_dataset:
    print(f'z: {data.z}')
    print(f'z.shape: {data.z.shape}')
    print(f'pos.shape: {data.pos.shape}')