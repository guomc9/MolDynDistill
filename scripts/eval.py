import sys
sys.path.append('.')
import torch
from utils.dataset import get_dataset, split_dataset
from utils.net import get_network
from utils.run.infer import Infer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# molecular = 'aspirin'
molecular = 'benzene'

dataset = get_dataset(dataset_name='MD17', root='data/MD17', name=molecular)
train_dataset, valid_dataset, test_dataset = split_dataset(dataset=dataset, train_size=1000, valid_size=10000, seed=42)
# network = get_network(name='spherenet', energy_and_force=True)
# network.load_state_dict(torch.load(f'.ckpt/sphnet/{molecular}/{molecular}_checkpoint.pt', map_location='cpu')['model_state_dict'])

network = get_network(name='schnet', energy_and_force=True)
# network.load_state_dict(torch.load(f'.log/expert_trajectory/schnet/MD17/benzene/2024-10-31-21-18-39/checkpoint_epoch_86.pt', map_location='cpu')['model_state_dict'])
# network.load_state_dict(torch.load(f'.log/expert_trajectory/schnet/MD17/benzene/2024-11-02-18-07-20/best_valid_checkpoint.pt', map_location='cpu')['model_state_dict'])
# network.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-04-17-27-23/checkpoint_epoch_89.pt', map_location='cpu')['model_state_dict'])
# network.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-05-18-56-10/best_valid_checkpoint.pt', map_location='cpu')['model_state_dict'])  # (0.54615, 3.68093)
# network.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-06-00-27-15/best_valid_checkpoint.pt', map_location='cpu')['model_state_dict'])  # (0.52049, 3.55423)
# network.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-06-13-03-09/best_valid_checkpoint.pt', map_location='cpu')['model_state_dict'])  # (0.38602, 2.98603)
network.load_state_dict(torch.load('.log/data_distill/mtt/MD17/benzene/2024-12-02-16-08-21/eval/200/best_valid_checkpoint.pt', map_location='cpu')['model_state_dict'])  # (146464.8125, 14.406352996826172)

# network.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-12-02-11-28-35/checkpoint_epoch_20.pt', map_location='cpu')['model_state_dict'])  # (356.4842224121094, 82.54507446289062)


network = network.to(device)
infer = Infer()

res = infer.inference(
    device=device, 
    test_dataset=valid_dataset, 
    model=network, 
    energy_and_force=True, 
    batch_size=64
    )

print(res)