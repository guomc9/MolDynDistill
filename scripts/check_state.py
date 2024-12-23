import sys
sys.path.append('.')
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.autograd import grad
from utils.net.schnet import SchNet
from utils.dataset import get_dataset, split_dataset


model = SchNet()
model.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-12-23-11-07-19/best_valid_checkpoint.pt')['model_state_dict'])
model.train()
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, id: {id(param)}, shape: {param.shape}")
    
optimizer = Adam(params=model.parameters(), lr=1e-5)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
molecular = 'benzene'
num_atoms = 12
batch_size = 1
dataset = get_dataset(dataset_name='MD17', root='data/MD17', name=molecular)
train_dataset, _, _ = split_dataset(dataset=dataset, train_size=1000, valid_size=1000, seed=42)

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_func = torch.nn.MSELoss()
p = 100.0
for batch_idx, batch_data in enumerate(dataloader):
    batch_data.pos.requires_grad_(True)
    out = model(batch_data)
    force = -grad(outputs=out, inputs=batch_data.pos,
                grad_outputs=torch.ones_like(out),
                create_graph=True, retain_graph=True)[0]
    e_loss = loss_func(out, batch_data.y.unsqueeze(1))
    f_loss = loss_func(force, batch_data.force)
    loss = 1 / p * e_loss + f_loss
    loss.backward()
    optimizer.step()
    break

print('optimizer state:')
for k, v in optimizer.state_dict()['state'].items():
    print(f"Parameter id: {k}, State: {v['exp_avg'].shape}")