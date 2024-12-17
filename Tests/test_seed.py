import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
def unset_seed():
    random.seed(None)
    np.random.seed(None)
    torch.seed()
    torch.cuda.seed()
    torch.cuda.seed_all()

dataset = ExampleDataset(data=list(range(10)))

set_seed(42)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for epoch in range(3):
    print(f"Epoch {epoch + 1}:")
    for batch in dataloader:
        print(batch)

set_seed(42)


dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for epoch in range(3):
    print(f"Epoch {epoch + 1}:")
    for batch in dataloader:
        print(batch)

print('set seed')
set_seed(42)
print(np.random.randint(0, 100))
print(np.random.randint(0, 100))

print('unset seed')
unset_seed()
print(np.random.randint(0, 100))
print(np.random.randint(0, 100))

print('set seed')
set_seed(42)
print(np.random.randint(0, 100))
print(np.random.randint(0, 100))

print('unset seed')
unset_seed()
print(np.random.randint(0, 100))
print(np.random.randint(0, 100))

print('set seed')
set_seed(42)
print(np.random.randint(0, 100))
print(np.random.randint(0, 100))

print('unset seed')
unset_seed()
print(np.random.randint(0, 100))
print(np.random.randint(0, 100))