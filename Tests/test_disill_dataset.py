import sys
sys.path.append('.')
import torch
from utils.dataset import get_dataset, split_dataset, DistillDataset
from utils.net.schnet import SchNet
from utils.distill.reparam_module import ReparamModule
from torch.autograd import grad
from tqdm import trange


class Infer:
    """
    Trainer class for 3DGN methods with wandb logging support
    """
    def __init__(self):
        """Initialize trainer"""
        self.best_valid = float('inf')
        self.best_test = float('inf')
        
    def inference(self, device, distill_dataset, 
              reparam_model, params, batch_size=32,
              energy_and_force=False):
        """
        Main training loop with wandb integration
        
        Args:
            device (torch.device): Device for computation
            distill_dataset: Distill data
            model: 3DGN model (SchNet, SphereNet etc.)
            evaluation (function): Evaluation function
            batch_size (int): Batch size 
            energy_and_force (bool): Whether to predict energy and forces
            p (int): Force weight in joint loss
            save_dir (str): Directory to save model checkpoints
            project_name (str): Name for wandb project
            save_step (int): Save checkpoint every N epochs
        """
        num_params = sum(p.numel() for p in reparam_model.parameters())
        print(f'#Params: {num_params}')
            
        # Testing  
        print('\n\nTesting...', flush=True)
        loss_list = self._evaluate(reparam_model, params, distill_dataset=distill_dataset, energy_and_force=energy_and_force, batch_size=batch_size)
        return loss_list

    def _evaluate(self, reparam_model, params, distill_dataset, energy_and_force, batch_size):
        """Evaluation step"""
        reparam_model.eval()
        
        num_iteration = (len(distill_dataset) + batch_size - 1) // batch_size
        energy_criterion = torch.nn.MSELoss()
        force_criterion = torch.nn.MSELoss()
        loss_list = []
        for i in trange(num_iteration):
            indices = torch.arange(i * batch_size, (i + 1) * batch_size if (i + 1) * batch_size < len(distill_dataset) else len(distill_dataset), device=device, dtype=torch.long)
            batch_data = distill_dataset.get_batch(indices)
            if energy_and_force:
                batch_data.pos.requires_grad_(True)
            out = reparam_model(batch_data, flat_param=params)
            e_loss = energy_criterion(out, batch_data.y.unsqueeze(1))
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos,
                            grad_outputs=torch.ones_like(out),
                            create_graph=True, retain_graph=True)[0]
                f_loss = force_criterion(force, batch_data.force)
                del force
                torch.cuda.empty_cache()
                
            loss_list.append((e_loss * 0.01 + f_loss).detach().cpu().item())

            del batch_data, out, e_loss
            if energy_and_force:
                del f_loss
            torch.cuda.empty_cache()
        
        return loss_list
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    molecular = 'benzene'
    dataset = get_dataset(dataset_name='MD17', root='data/MD17', name=molecular)
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset=dataset, train_size=1000, valid_size=1000, seed=42)
    distill_dataset = DistillDataset(source_dataset=valid_dataset, distill_rate=0.6, distill_lr=3.0e-10, device=device)
    load_net = SchNet()
    # load_net.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-04-17-27-23/checkpoint_epoch_800.pt')['model_state_dict'])
    load_net.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-04-17-27-23/best_valid_checkpoint.pt')['model_state_dict'])
    params = []
    for param in load_net.parameters():
        params.append(param.data.reshape(-1))
    params = torch.cat(params, dim=0).to(device)
    print(params.shape)
    reparam_net = SchNet().to(device)
    reparam_net = ReparamModule(reparam_net)
    infer = Infer()
    loss_list = infer.inference(device, distill_dataset, reparam_net, params, energy_and_force=True, batch_size=64)
    print(loss_list)
    print(sum(loss_list) / len(loss_list))