import sys
sys.path.append('.')
import torch
from utils.dataset import get_dataset, split_dataset
from utils.net.schnet import SchNet
from utils.distill.reparam_module import ReparamModule
from torch_geometric.data import DataLoader
from torch.autograd import grad
from tqdm import tqdm
from dig.threedgraph.evaluation import ThreeDEvaluator


class Infer:
    """
    Trainer class for 3DGN methods with wandb logging support
    """
    def __init__(self):
        """Initialize trainer"""
        self.best_valid = float('inf')
        self.best_test = float('inf')
        
    def inference(self, device, test_dataset, 
              reparam_model, params, loss_func=None, evaluation=None, batch_size=32,
              energy_and_force=False):
        """
        Main training loop with wandb integration
        
        Args:
            device (torch.device): Device for computation
            test_dataset: Test data
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
        
        # Initialize scheduler
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            
        if evaluation is None:
            evaluation = ThreeDEvaluator()
        
        if loss_func is None:    
            loss_func = torch.nn.L1Loss()
            
        # Testing  
        print('\n\nTesting...', flush=True)
        test_mae = self._evaluate(reparam_model, params, data_loader=test_loader, energy_and_force=energy_and_force,
                                evaluation=evaluation, device=device)

        metrics = {
            'test_mae': test_mae,
        }
        
        print(f"\nMetrics: {metrics}")
        return metrics

    def _evaluate(self, reparam_model, params, data_loader, energy_and_force, 
                evaluation, device):
        """Evaluation step"""
        reparam_model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])

        if energy_and_force:
            preds_force = torch.Tensor([])
            targets_force = torch.Tensor([])
        
        for batch_data in tqdm(data_loader):
            batch_data = batch_data.to(device)
            if energy_and_force:
                batch_data.pos.requires_grad_(True)
            out = reparam_model(batch_data, flat_param=params)
            
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos,
                            grad_outputs=torch.ones_like(out),
                            create_graph=True, retain_graph=False)[0]
                preds_force = torch.cat([preds_force, force.detach().cpu()], dim=0)
                targets_force = torch.cat([targets_force, batch_data.force.detach().cpu()], dim=0)
                del force
                torch.cuda.empty_cache()
            preds = torch.cat([preds, out.detach().cpu()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1).detach().cpu()], dim=0)

            del batch_data, out
            torch.cuda.empty_cache()

        input_dict = {"y_true": targets, "y_pred": preds}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            return energy_mae, force_mae

        return evaluation.eval(input_dict)['mae']
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    molecular = 'benzene'
    dataset = get_dataset(dataset_name='MD17', root='data/MD17', name=molecular)
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset=dataset, train_size=1000, valid_size=10000, seed=42)
    
    load_net = SchNet()
    load_net.load_state_dict(torch.load('.log/expert_trajectory/schnet/MD17/benzene/2024-11-04-17-27-23/checkpoint_epoch_89.pt')['model_state_dict'])
    params = []
    for param in load_net.parameters():
        params.append(param.data.reshape(-1))
    params = torch.cat(params, dim=0).to(device)
    print(params.shape)
    reparam_net = SchNet().to(device)
    reparam_net = ReparamModule(reparam_net)
    infer = Infer()
    infer.inference(device, valid_dataset, reparam_net, params, energy_and_force=True, batch_size=64)