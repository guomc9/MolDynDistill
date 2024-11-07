import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
from tqdm import tqdm, trange
import wandb
import re
from .reparam_module import ReparamModule
from ..dataset import DistillDatset
from ..net import get_network
from ..run.trainer import Trainer

class MTT:
    def __init__(self):
        pass
    
    def distill(
        self, 
        project_name, 
        num_iteration: int, 
        num_step_per_iteration: int, 
        expert_network_name: str, 
        expert_network_dict: dict, 
        expert_trajectory_dir: str, 
        max_start_epoch: int, 
        num_expert_epoch: int, 
        eval_step: int, 
        eval_network_pool: list, 
        eval_train_epoch: int, 
        # eval_batch_size: int, 
        eval_vt_batch_size: int, 
        eval_scheduler_name: str, 
        eval_lr_decay_factor: float, 
        eval_lr_decay_step_size: int, 
        save_step: int, 
        save_dir: str, 
        distill_rate: float, 
        train_dataset, 
        eval_dataset, 
        device: str, 
        distill_batch: int, 
        distill_lr_assistant_net: float, 
        distill_lr_lr: float, 
        distill_base_lr: float, 
        all_distill_data_per_iteration: bool = True, 
        distill_lr_pos: float = None, 
        p: float = 100, 
        enable_assistant_net: bool = True, 
        distill_energy_and_force: bool = True, 
        lr_requires_grad: bool = False, 
        pos_requires_grad: bool = False, 
        energy_requires_grad: bool = False, 
        force_requires_grad: bool = False, 
        enable_log: bool = True, 
        **kwargs
    ):
        if enable_log:
            wandb.init(project=project_name, config=kwargs)
            wandb.config.update({
                'num_iteration': num_iteration, 
                'num_step_per_iteration': num_step_per_iteration, 
                'expert_network': expert_network_name, 
                'expert_trajectory_dir' : expert_trajectory_dir, 
                'max_start_epoch': max_start_epoch, 
                'distill_rate': distill_rate, 
                'distill_dataset_size': int(distill_rate * len(train_dataset))
            })
            wandb.config.update(expert_network_dict)
            
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        
        best_valid = float('inf')
        if enable_assistant_net:
            student_net, assistant_net = get_network(name=expert_network_name, return_assistant_net=True, **expert_network_dict)
            student_net = student_net.to(device)
            assistant_net = assistant_net.to(device)
        else:
            student_net = get_network(name=expert_network_name, return_assistant_net=False, **expert_network_dict).to(device)
            assistant_net = None
        
        load_net = get_network(name=expert_network_name, return_assistant_net=False, **expert_network_dict)
        student_net = ReparamModule(student_net)
        distill_dataset = DistillDatset(source_dataset=train_dataset, distill_rate=distill_rate, distill_lr=distill_base_lr, device=device, pos_requires_grad=pos_requires_grad or distill_energy_and_force, energy_requires_grad=energy_requires_grad, force_requires_grad=force_requires_grad)

        # optimizers
        self.optimizer_pos = None
        self.optimizer_assistant = None
        self.optimizer_lr = None
        if enable_assistant_net:
            self.optimizer_assistant = optim.SGD(assistant_net.parameters(), lr=distill_lr_assistant_net, momentum=0.0)
        if pos_requires_grad and distill_lr_pos is not None:
            self.optimizer_pos = optim.SGD([distill_dataset.get_pos()], lr=distill_lr_pos, momentum=0.0)
        if lr_requires_grad and distill_lr_lr is not None:
            self.optimizer_lr = optim.SGD([distill_dataset.get_lr()], lr=distill_lr_lr, momentum=0.0)
        
        expert_file_list = []
        for file in os.listdir(expert_trajectory_dir):
            if re.search(r'\d', os.path.basename(file)):
                expert_file_list.append(os.path.join(expert_trajectory_dir, file))
                
        expert_file_list = sorted(expert_file_list)
        with tqdm(range(1, num_iteration+1)) as pbar:
            for it in pbar:
                if self.optimizer_assistant is not None:
                    self.optimizer_assistant.zero_grad()
                if self.optimizer_pos is not None:
                    self.optimizer_pos.zero_grad()
                if self.optimizer_lr is not None:
                    self.optimizer_lr.zero_grad()
                
                start_epoch = np.random.randint(0, max_start_epoch)
                end_epoch = start_epoch + num_expert_epoch
                
                # Load Expert Trajectory
                start_expert_params = []
                load_net.load_state_dict(torch.load(expert_file_list[start_epoch])['model_state_dict'])
                for param in load_net.parameters():
                    start_expert_params.append(param.data.reshape(-1))
                start_expert_params = torch.cat(start_expert_params, dim=0).to(device)
                
                target_expert_params = []
                load_net.load_state_dict(torch.load(expert_file_list[end_epoch])['model_state_dict'])
                for param in load_net.parameters():
                    target_expert_params.append(param.data.reshape(-1))
                target_expert_params = torch.cat(target_expert_params, dim=0).to(device)
                
                student_params_list = [start_expert_params.detach().clone().requires_grad_(True)]
                
                # Evaluate
                if it % eval_step == 0:
                    print(f'Evaluating Distill Dataset...')
                    for name in eval_network_pool:
                        eval_trainer = Trainer()
                        res = eval_trainer.train(
                            device=device, 
                            train_dataset=distill_dataset.to_torch(), 
                            valid_dataset=eval_dataset, 
                            test_dataset=None, 
                            model=get_network(name=name), 
                            assistant_model=assistant_net, 
                            loss_func=torch.nn.MSELoss(), 
                            evaluation=None, 
                            epochs=eval_train_epoch, 
                            lr=distill_dataset.get_lr(detach=True), 
                            batch_size=distill_batch, 
                            vt_batch_size=eval_vt_batch_size, 
                            optimizer_name='SGD', 
                            scheduler_name=eval_scheduler_name, 
                            lr_decay_factor=eval_lr_decay_factor, 
                            lr_decay_step_size=eval_lr_decay_step_size, 
                            energy_and_force=True, 
                            val_step=1, 
                            p=100, 
                            enable_log=False, 
                        )
                        if res < best_valid:
                            best_valid = loss
                            print(f'Best loss at iteration {it}: {best_valid}')
                            distill_dataset.save(os.path.join(save_dir, 'best', '.pt'))

                        if enable_log:
                            wandb.log({"distill/best_valid": best_valid}, step=it)

                # Save
                if it % save_step == 0:
                    distill_dataset.save(os.path.join(save_dir, str(it), '.pt'))

                # Distill
                energy_criterion = torch.nn.MSELoss()
                force_criterion = torch.nn.MSELoss()
                student_net.train()
                if assistant_net is not None:
                    assistant_net.train()
                # param_loss_list = []
                # param_dist_list = []
                num_params = sum([np.prod(param.size()) for param in (student_net.parameters())])
                if all_distill_data_per_iteration:
                    num_step_per_iteration = (len(distill_dataset) + distill_batch - 1) // distill_batch
                    
                for step in trange(num_step_per_iteration):
                    if all_distill_data_per_iteration:
                        indices = torch.arange(step * distill_batch, (step + 1) * distill_batch, device=device, dtype=torch.long)
                    else:
                        indices = torch.randperm(len(distill_dataset), device=device)[:distill_batch]
                    batch_data = distill_dataset.get_batch(indices)
                    if distill_energy_and_force:
                        batch_data.pos.requires_grad_(True)
                    if assistant_net is not None:
                        assistant_outs = assistant_net(batch_data)
                        output = student_net(batch_data=batch_data, assistant_outs=assistant_outs, flat_param=student_params_list[-1])
                    else:
                        output = student_net(batch_data=batch_data, flat_param=student_params_list[-1])
                    
                    loss = 1 / p * energy_criterion(output, batch_data.y.unsqueeze(1))
                    if distill_energy_and_force:
                        force = -torch.autograd.grad(outputs=output, inputs=batch_data.pos, 
                                    grad_outputs=torch.ones_like(output), 
                                    create_graph=True, retain_graph=True)[0]
                        loss += force_criterion(force, batch_data.force)
                    grad = torch.autograd.grad(loss, student_params_list[-1], create_graph=True)[0]
                    print(f'start_epoch: {start_epoch}, loss: {loss.detach().cpu().item()}, grad max: {torch.max(grad)}, lr: {distill_dataset.get_lr()}')
                    student_params_list.append(student_params_list[-1] - distill_dataset.get_lr() * grad)
                            
                param_loss = torch.tensor(0.0).to(device)
                param_dist = torch.tensor(0.0).to(device)
                param_loss += torch.nn.functional.mse_loss(student_params_list[-1], target_expert_params, reduction="sum")
                param_dist += torch.nn.functional.mse_loss(start_expert_params, target_expert_params, reduction="sum")

                # param_loss_list.append(param_loss)
                # param_dist_list.append(param_dist)

                param_loss /= num_params
                param_dist /= num_params

                param_loss /= param_dist

                grad_loss = param_loss

                grad_loss.backward()

                if self.optimizer_assistant is not None:
                    self.optimizer_assistant.step()
                if self.optimizer_pos is not None:
                    self.optimizer_pos.step()
                if self.optimizer_lr is not None:
                    self.optimizer_lr.step()
                
                if enable_log:
                    wandb.log({"start_epoch": start_epoch, "distill/grad_loss": grad_loss.detach().cpu()}, step=it)
                print(f'iter: {it}, distill_loss: {loss.detach().cpu().item()}')

                pbar.set_postfix(grad_loss=grad_loss.detach().cpu(), step=it)

                for _ in student_params_list:
                    del _
        
        if enable_log:
            wandb.finish()
