import os
import sys
sys.path.append('.')
import torch
from datetime import datetime
from utils.dataset import get_dataset, split_dataset
from utils.net import get_network
from utils.run.trainer import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_cfg = {
    "dataset": 'MD17', 
    "root": 'data/MD17', 
    "name": 'benzene', 
    "train_ratio": 0.7, 
    "valid_ratio": 0.1, 
    "seed": 42
}

network_cfg ={
    "name": 'schnet', 
    'hidden_channels': 128,
    'out_channels': 1,
    'num_filters': 128,
    'cutoff': 10.0,
    'energy_and_force': False,
    'num_layers': 6, 
    'num_gaussians': 50
}

train_cfg = {
    'project_name': 'MoleculeDynamics-Expert-Trajectory', 
    'epochs': 500, 
    'batch_size': 32, 
    'vt_batch_size': 32, 
    'optimizer_name': 'Adam', 
    'weight_decay': 0, 
    'save_step': 1, 
    'lr': 5e-4, 
    'scheduler_name': 'stepLR', 
    'lr_decay_step_size': 50, 
    'lr_decay_factor': 0.5, 
    'energy_and_force': False, 
    'p': 100, 
    'save_dir': os.path.join('.log', network_cfg["name"], data_cfg["dataset"], '' if data_cfg["name"] is None else data_cfg["name"], datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
}

dataset = get_dataset(dataset='MD17', root='data/MD17', name='benzene')
train_dataset, valid_dataset, test_dataset = split_dataset(dataset=dataset, train_ratio=0.7, valid_ratio=0.1, seed=42)
network = get_network(name='schnet')
trainer = Trainer()
cfg = train_cfg.copy()
cfg.update(network_cfg)
cfg.update(data_cfg)

trainer.train(
    device=device, 
    train_dataset=train_dataset, 
    valid_dataset=valid_dataset, 
    test_dataset=test_dataset, 
    model=network, 
    **cfg
    )