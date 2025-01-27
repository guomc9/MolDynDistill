import os
import re
import torch
import wandb
from torch.optim import Adam, AdamW, SGD
from torch_geometric.loader import DataLoader
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm
from dig.threedgraph.evaluation import ThreeDEvaluator

class Trainer:
    """
    Trainer class for 3DGN methods with wandb logging support
    """
    def __init__(self):
        """Initialize trainer"""
        self.best_valid = float('inf')
        self.best_valid_energy = float('inf')
        self.best_valid_force = float('inf')
        self.best_test = float('inf')
        
    def train(self, device, train_dataset, valid_dataset, test_dataset, 
              model, assistant_model=None, loss_func=None, evaluation=None, epochs=500, batch_size=32,
              vt_batch_size=32, optimizer_name='Adam', lr=0.0005,
              scheduler_name=None, lr_decay_factor=0.5, 
              lr_decay_step_size=50, weight_decay=0,
              energy_and_force=False, p=100, save_dir='',
              project_name='3DGN-Training', val_step=10, test_step=10, save_step=50, early_epoch=10, early_save_iters=50, shuffle=False, 
              enable_log=True, wandb_run_id=None, seed_hook=None, **kwargs):
        """
        Main training loop with wandb integration
        
        Args:
            device (torch.device): Device for computation
            train_dataset: Training data
            valid_dataset: Validation data  
            test_dataset: Test data
            model: 3DGN model (SchNet, SphereNet etc.)
            loss_func (function): Loss function
            evaluation (function): Evaluation function
            epochs (int): Total training epochs
            batch_size (int): Training batch size 
            vt_batch_size (int): Validation/test batch size
            optimizer_name (str): Name of optimizer ('Adam', 'AdamW', 'SGD')
            lr (float): Initial learning rate
            scheduler_name (str): Name of scheduler (None or 'StepLR')
            lr_decay_factor (float): Learning rate decay factor
            lr_decay_step_size (int): Steps between LR decay
            weight_decay (float): Weight decay factor
            energy_and_force (bool): Whether to predict energy and forces
            p (int): Force weight in joint loss
            save_dir (str): Directory to save model checkpoints
            project_name (str): Name for wandb project
            save_step (int): Save checkpoint every N epochs
        """
        
        # Initialize wandb
        if enable_log: 
            wandb.init(project=project_name, id=wandb_run_id, resume="allow")
            wandb.config.update({
                "epochs": epochs,
                "batch_size": batch_size,
                "vt_batch_size": vt_batch_size, 
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name, 
            })
            wandb.config.update(d=kwargs)
        
        model = model.to(device)
        num_params = sum(param.numel() for param in model.parameters())
        print(f'#Params: {num_params}')
        
        # Initialize optimizer
        if str.lower(optimizer_name) == 'adam':
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif str.lower(optimizer_name) == 'adamw':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif str.lower(optimizer_name) == 'sgd':
            print(f'optimizer: {str.lower(optimizer_name)}, lr: {lr}')
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
            
        # Initialize scheduler
        scheduler = None
        if scheduler_name is not None:
            if str.lower(scheduler_name) == 'steplr':
                scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
            elif str.lower(scheduler_name) == 'expdecaylr':
                decay_lambda = lambda step: lr_decay_factor ** (step / lr_decay_step_size)
                scheduler = LambdaLR(optimizer, lr_lambda=decay_lambda)
            else:
                scheduler = None
        if seed_hook is not None:
            seed_hook()
            # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False) if valid_dataset is not None else None
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False) if test_dataset is not None else None
        
        # print(f'len(train_loader): {len(train_loader)}')
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self._save_checkpoint(save_dir, 'checkpoint_epoch_0.pt',
                                            model, optimizer, scheduler, None, None, 0, 0)
        self._save_checkpoint(save_dir, 'checkpoint_iters_0.pt',
                                            model, optimizer, scheduler, None, None, 0, 0)
        
        if evaluation is None:
            evaluation = ThreeDEvaluator()
        
        if loss_func == 'l1':
            loss_func = torch.nn.L1Loss()
        else:
            loss_func = torch.nn.MSELoss()
        
        # Check for existing checkpoints if resume is enabled
        start_epoch = 1
        if save_dir:
            checkpoint_files = [f for f in os.listdir(save_dir) if re.match(r'checkpoint_epoch_\d+\.pt', f)]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda f: int(re.search(r'\d+', f).group()))
                checkpoint_path = os.path.join(save_dir, latest_checkpoint)
                start_epoch = self._load_checkpoint(checkpoint_path, model, optimizer, scheduler) + 1
        
        for epoch in range(start_epoch, epochs + 1):
            print(f"\n=====Epoch {epoch}", flush=True)
            
            # Training
            print('\nTraining...', flush=True)
            train_err = self._train_epoch(model, assistant_model, optimizer, scheduler, train_loader, 
                                        energy_and_force, p, loss_func, epoch, early_epoch, early_save_iters, device, save_dir)
            print({'train_err': train_err, 'lr': optimizer.param_groups[0]['lr']})
            if enable_log:
                wandb.log({'train_err': train_err, 'lr': optimizer.param_groups[0]['lr']}, step=epoch)

            # Validation
            if valid_loader is not None and epoch % val_step == 0:
                print('\n\nEvaluating...', flush=True)
                valid_err, energy_valid_err, force_valid_err = self._evaluate(model, valid_loader, energy_and_force, 
                                        1, evaluation, device)
                    
                # Save checkpoint on validation improvement
                if valid_err < self.best_valid:
                    self.best_valid = valid_err
                    self.best_valid_energy = energy_valid_err
                    self.best_valid_force = force_valid_err
                    if save_dir:
                        self._save_checkpoint(save_dir, 'best_valid_checkpoint.pt',
                                            model, optimizer, scheduler, None, valid_err, epoch, (epoch-1)*len(train_loader))
                print({'valid_err': valid_err, 'best_valid': self.best_valid, 'best_valid_energy': self.best_valid_energy, 'best_valid_force': self.best_valid_force})
                if enable_log:
                    wandb.log({'valid_err': valid_err, 'valid_energy_err': energy_valid_err, 'valid_force_err': force_valid_err, \
                        'best_valid': self.best_valid, 'best_valid_energy': self.best_valid_energy, 'best_valid_force': self.best_valid_force}, step=epoch)

            # Testing  
            if test_loader is not None and epoch % test_step == 0:
                print('\n\nTesting...', flush=True)
                test_err, energy_test_err, force_test_err = self._evaluate(model, test_loader, energy_and_force,
                                        1, evaluation, device)
                
                print({'test_err': test_err, 'energy_test_err': energy_test_err, 'force_test_err': force_test_err})
                if enable_log:
                    wandb.log({'test_err': test_err, 'energy_test_err': energy_test_err, 'force_test_err': force_test_err}, step=epoch)
            
                    
            # Periodic checkpoint saving
            if save_dir and epoch % save_step == 0:
                self._save_checkpoint(save_dir, f'checkpoint_epoch_{epoch}.pt',
                                    model, optimizer, scheduler, train_err, None, epoch, epoch*len(train_loader))
                self._save_checkpoint(save_dir, f'checkpoint_iters_{epoch*len(train_loader)}.pt',
                                    model, optimizer, scheduler, train_err, None, epoch, epoch*len(train_loader))
        print(f'Best validation error: {self.best_valid}')
        if enable_log:
            wandb.finish()
        
        return self.best_valid, self.best_valid_energy, self.best_valid_force
        
    def _train_epoch(self, model, assistant_model, optimizer, scheduler, train_loader, 
                     energy_and_force, p, loss_func, epoch, early_epoch, early_save_iters, device, save_dir):
        """Training for one epoch"""
        model.train()
        iters = (epoch - 1) * len(train_loader)
        last_loss = None
        if assistant_model is not None:
            assistant_model.eval()
            assistant_model.requires_grad_(False)
        loss_list = []
        for i, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            if energy_and_force:
                batch_data.pos.requires_grad_(True)
            # print(f'pos: {batch_data.pos}, z: {batch_data.z}, energy: {batch_data.energy}, force: {batch_data.force}')
            
            if assistant_model is not None:
                assistant_outs = assistant_model(batch_data)
                out = model(batch_data, assistant_outs=assistant_outs)
            else:
                out = model(batch_data)
            
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos,
                            grad_outputs=torch.ones_like(out),
                            create_graph=True, retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = 1 / p * e_loss + f_loss
                # print(f'e_loss: {e_loss}, f_loss: {f_loss}')
            else:
                loss = loss_func(out, batch_data.y.unsqueeze(1))
                
            loss.backward()
            all_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.view(-1))
            if all_grads:
                all_grads = torch.cat(all_grads)
            #     print(f'max: {all_grads.max().item()}')
            #     print(f'min: {all_grads.min().item()}')
            #     print(f'mean: {all_grads.mean().item()}')
            # print(f'lr: {optimizer.param_groups[0]["lr"]}')
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            iters += 1
            if scheduler:
                scheduler.step()
            
            loss_list.append(loss.detach().cpu().item())
            
            if last_loss is None:
                last_loss = sum(loss_list) / len(loss_list)
                
            if epoch <= early_epoch and iters % early_save_iters == 0:
                last_loss = sum(loss_list) / len(loss_list)
                self._save_checkpoint(save_dir, f'checkpoint_iters_{iters}.pt', model, optimizer, scheduler, last_loss, None, epoch, iters)
            
            del loss, out, batch_data
            if energy_and_force:
                del e_loss, f_loss
            torch.cuda.empty_cache()
            
            
        return sum(loss_list) / len(loss_list)

    def _evaluate(self, model, data_loader, energy_and_force, 
                p, evaluation, device):
        """Evaluation step"""
        model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])

        if energy_and_force:
            preds_force = torch.Tensor([])
            targets_force = torch.Tensor([])
        
        for batch_data in tqdm(data_loader):
            batch_data = batch_data.to(device)
            out = model(batch_data)
            
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
            return energy_mae + p * force_mae, energy_mae, force_mae

        return evaluation.eval(input_dict)['mae']
    
    def _save_checkpoint(self, save_dir, filename, model, optimizer, 
                        scheduler, train_err, valid_err, epoch, iters):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'iters': iters, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_mae': self.best_valid, 
            'best_valid_force_mae': self.best_valid_force, 
            'best_valid_energy_mae': self.best_valid_energy, 
            'train_err': train_err, 
            'valid_err': valid_err
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, os.path.join(save_dir, filename))
        print(f'Saved checkpoint to {os.path.join(save_dir, filename)}')
        
    def _load_checkpoint(self, checkpoint_path, model, optimizer, scheduler):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_valid = checkpoint.get('best_valid_mae', float('inf'))
        
        print(f"Loaded checkpoint from '{checkpoint_path}' (Epoch {checkpoint['epoch']})")
        return checkpoint['epoch']