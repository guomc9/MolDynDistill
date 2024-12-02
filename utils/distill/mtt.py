import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
from tqdm import tqdm, trange
import wandb
import re
import random
from .reparam_module import ReparamModule
from ..dataset import DistillDatset
from ..net import get_network
from ..run.trainer import Trainer
from ..optimizers import get_dynamic_optimizer

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
        max_start_iter: int, 
        min_start_iter: int, 
        num_expert: int, 
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
        distill_lr_assistant_net: float = None, 
        distill_lr_lr: float = None, 
        distill_base_lr: float = 1.0e-4, 
        all_distill_data_per_iteration: bool = True, 
        noise_pos: bool = True, 
        distill_lr_pos: float = None, 
        distill_lr_energy: float = None, 
        distill_lr_force: float = None, 
        p: float = 100, 
        enable_assistant_net: bool = True, 
        distill_energy_and_force: bool = True, 
        distill_optimizer_type: str = 'adam', 
        dynamic_optimizer_type: str = 'sgd', 
        distill_scheduler_type: str = "stepLR", 
        distill_scheduler_decay_step: int = 100, 
        distill_scheduler_decay_rate: float = 0.2, 
        lr_requires_grad: bool = False, 
        pos_requires_grad: bool = False, 
        energy_requires_grad: bool = False, 
        force_requires_grad: bool = False, 
        max_grad_norm_clip: float = None, 
        enable_log: bool = True, 
        revise_energy_and_force: bool = True, 
        check_energy_and_force: bool = True, 
        **kwargs
    ):
        if enable_log:
            wandb.init(project=project_name, config=kwargs)
            wandb.config.update({
                'num_iteration': num_iteration, 
                'num_step_per_iteration': num_step_per_iteration, 
                'expert_trajectory_dir' : expert_trajectory_dir, 
                'max_start_iter': max_start_iter, 
                'min_start_iter': min_start_iter, 
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
            assistant_net.requires_grad_(True)
        else:
            student_net = get_network(name=expert_network_name, return_assistant_net=False, **expert_network_dict).to(device)
            assistant_net = None
        
        load_net = get_network(name=expert_network_name, return_assistant_net=False, **expert_network_dict)
        student_net = ReparamModule(student_net)
        distill_dataset = DistillDatset(source_dataset=train_dataset, distill_rate=distill_rate, distill_lr=distill_base_lr, device=device, pos_requires_grad=pos_requires_grad or distill_energy_and_force, energy_requires_grad=energy_requires_grad, force_requires_grad=force_requires_grad, noise_pos=noise_pos)

        # optimizers
        self.optimizer_pos = None
        self.optimizer_assistant = None
        self.optimizer_lr = None
        self.optimizer_energy = None
        self.optimizer_force = None
        
        self.scheduler_pos = None
        self.scheduler_assistant = None
        self.scheduler_lr = None
        self.scheduler_energy = None
        self.scheduler_force = None
        
        # if enable_assistant_net and distill_lr_assistant_net is not None:
        #     self.optimizer_assistant = optim.SGD(assistant_net.parameters(), lr=distill_lr_assistant_net, momentum=0.0)
        # if pos_requires_grad and distill_lr_pos is not None:
        #     self.optimizer_pos = optim.SGD([distill_dataset.get_pos()], lr=distill_lr_pos, momentum=0.0)
        # if lr_requires_grad and distill_lr_lr is not None:
        #     self.optimizer_lr = optim.SGD([distill_dataset.get_lr()], lr=distill_lr_lr, momentum=0.0)
        
        if enable_assistant_net and distill_lr_assistant_net is not None:
            if distill_optimizer_type == 'adam':
                self.optimizer_assistant = optim.Adam(assistant_net.parameters(), lr=distill_lr_assistant_net)
                print(f'assistant net optimizer: Adam, lr: {distill_lr_assistant_net}')
            else:
                self.optimizer_assistant = optim.SGD(assistant_net.parameters(), lr=distill_lr_assistant_net, momentum=0.0)
                print(f'assistant net optimizer: SGD, lr: {distill_lr_assistant_net}')
            
            if distill_scheduler_type == 'step':
                self.scheduler_assistant_net = optim.lr_scheduler.StepLR(self.optimizer_assistant, step_size=distill_scheduler_decay_step, gamma=distill_scheduler_decay_rate)
                
        if pos_requires_grad and distill_lr_pos is not None:
            if distill_optimizer_type == 'adam':
                self.optimizer_pos = optim.Adam([distill_dataset.get_pos()], lr=distill_lr_pos)
                print(f'pos optimizer: Adam, lr: {distill_lr_pos}')
            else:
                self.optimizer_pos = optim.SGD([distill_dataset.get_pos()], lr=distill_lr_pos, momentum=0.0)
                print(f'pos optimizer: SGD, lr: {distill_lr_pos}')
                
            if distill_scheduler_type == 'step':
                self.scheduler_pos = optim.lr_scheduler.StepLR(self.optimizer_pos, step_size=distill_scheduler_decay_step, gamma=distill_scheduler_decay_rate)
                
        if lr_requires_grad and distill_lr_lr is not None:
            if distill_optimizer_type == 'adam':
                self.optimizer_lr = optim.Adam([distill_dataset.get_lr()], lr=distill_lr_lr)
                print(f'lr optimizer: Adam, lr: {distill_lr_lr}')
            else:
                self.optimizer_lr = optim.SGD([distill_dataset.get_lr()], lr=distill_lr_lr, momentum=0.0)
                print(f'lr optimizer: SGD, lr: {distill_lr_lr}')
                
            if distill_scheduler_type == 'step':
                self.scheduler_lr = optim.lr_scheduler.StepLR(self.optimizer_lr, step_size=distill_scheduler_decay_step, gamma=distill_scheduler_decay_rate)
                
        if force_requires_grad and distill_lr_force is not None:
            if distill_optimizer_type == 'adam':
                self.optimizer_force = optim.Adam([distill_dataset.get_force()], lr=distill_lr_force)
                print(f'force optimizer: Adam, lr: {distill_lr_force}')
            else:
                self.optimizer_force = optim.SGD([distill_dataset.get_force()], lr=distill_lr_force, momentum=0.0)
                print(f'force optimizer: SGD, lr: {distill_lr_force}')
                
            if distill_scheduler_type == 'step':
                self.scheduler_force = optim.lr_scheduler.StepLR(self.optimizer_force, step_size=distill_scheduler_decay_step, gamma=distill_scheduler_decay_rate)
                
        if energy_requires_grad and distill_lr_energy is not None:
            if distill_optimizer_type == 'adam':
                self.optimizer_energy = optim.Adam([distill_dataset.get_y()], lr=distill_lr_energy)
                print(f'energy optimizer: Adam, lr: {distill_lr_energy}')
            else:
                self.optimizer_energy = optim.SGD([distill_dataset.get_y()], lr=distill_lr_energy, momentum=0.0)
                print(f'energy optimizer: SGD, lr: {distill_lr_energy}')
                
            if distill_scheduler_type == 'step':
                self.scheduler_energy = optim.lr_scheduler.StepLR(self.optimizer_energy, step_size=distill_scheduler_decay_step, gamma=distill_scheduler_decay_rate)
            
        expert_iter_list = []
        expert_file_list = {}
        for file in os.listdir(expert_trajectory_dir):
            if re.search(r'checkpoint_iters_\d', os.path.basename(file)):
                idx = int(re.findall(r'\d+', os.path.basename(file))[0])
                expert_file_list[idx] = os.path.join(expert_trajectory_dir, file)
                expert_iter_list.append(idx)
                
        expert_file_list = dict(sorted(expert_file_list.items()))
        expert_iter_list = sorted(expert_iter_list)
        best_expert_net = None
        if revise_energy_and_force or check_energy_and_force:
            best_expert_net = get_network(name=expert_network_name, return_assistant_net=False, **expert_network_dict)
            best_expert_net.load_state_dict(torch.load(os.path.join(expert_trajectory_dir, 'best_valid_checkpoint.pt'))['model_state_dict'])
            best_expert_net.to(device)
        
        param_loss_list = []
        with tqdm(range(1, num_iteration+1)) as pbar:
            for it in pbar:
                if self.optimizer_assistant is not None:
                    self.optimizer_assistant.zero_grad()
                if self.optimizer_pos is not None:
                    self.optimizer_pos.zero_grad()
                if self.optimizer_lr is not None:
                    self.optimizer_lr.zero_grad()
                if self.optimizer_force is not None:
                    self.optimizer_force.zero_grad()
                if self.optimizer_energy is not None:
                    self.optimizer_energy.zero_grad()
                filtered_expert_iter_list = [(i, it) for i, it in enumerate(expert_iter_list) if min_start_iter <= it < max_start_iter]
                start_i, start_it = random.choice(filtered_expert_iter_list)
                end_it = expert_iter_list[start_i + num_expert]
                print(f'start_it -> end_it: {start_it}->{end_it}')
                
                # Load Expert Trajectory
                start_expert_params = []
                load_net.load_state_dict(torch.load(expert_file_list[start_it])['model_state_dict'])
                for param in load_net.parameters():
                    start_expert_params.append(param.data.reshape(-1))
                start_expert_params = torch.cat(start_expert_params, dim=0).to(device)
                
                target_expert_params = []
                load_net.load_state_dict(torch.load(expert_file_list[end_it])['model_state_dict'])
                for param in load_net.parameters():
                    target_expert_params.append(param.data.reshape(-1))
                target_expert_params = torch.cat(target_expert_params, dim=0).to(device)
                
                student_params_list = [start_expert_params.detach().clone().requires_grad_(True)]
                dynamic_optimizer = get_dynamic_optimizer(optimizer_type=dynamic_optimizer_type, params=student_params_list[-1])
                
                # Evaluate
                if it % eval_step == 0:
                    print(f'Evaluating Distill Dataset...')
                    os.makedirs(os.path.join(save_dir, 'eval', f'{it}'), exist_ok=True)
                    for name in eval_network_pool:
                        eval_trainer = Trainer()
                        res = eval_trainer.train(
                            device=device, 
                            train_dataset=distill_dataset.to_torch(), 
                            valid_dataset=eval_dataset, 
                            test_dataset=None, 
                            model=get_network(name=name, **expert_network_dict), 
                            assistant_model=assistant_net, 
                            loss_func=torch.nn.MSELoss(), 
                            evaluation=None, 
                            epochs=eval_train_epoch, 
                            lr=distill_dataset.get_lr(detach=True), 
                            batch_size=distill_batch, 
                            vt_batch_size=eval_vt_batch_size, 
                            optimizer_name=dynamic_optimizer_type, 
                            scheduler_name=eval_scheduler_name, 
                            lr_decay_factor=eval_lr_decay_factor, 
                            lr_decay_step_size=eval_lr_decay_step_size, 
                            energy_and_force=True, 
                            val_step=1, 
                            p=100, 
                            save_dir=os.path.join(save_dir, 'eval', f'{it}'), 
                            enable_log=False, 
                        )
                        if res < best_valid:
                            best_valid = loss
                            print(f'Best loss at iteration {it}: {best_valid}')
                            distill_dataset.save(os.path.join(save_dir, 'best_valid.pt'))

                        if enable_log:
                            wandb.log({"distill/best_valid": best_valid}, step=it)

                # Save
                if it % save_step == 0:
                    distill_dataset.save(os.path.join(save_dir, str(it)+'.pt'))

                # Distill
                energy_criterion = torch.nn.MSELoss()
                force_criterion = torch.nn.MSELoss()
                student_net.train()
                if assistant_net is not None:
                    assistant_net.train()
                
                num_params = sum([np.prod(param.size()) for param in (student_net.parameters())])
                if all_distill_data_per_iteration:
                    num_step_per_iteration = (len(distill_dataset) + distill_batch - 1) // distill_batch
                for step in trange(num_step_per_iteration):
                    if all_distill_data_per_iteration:
                        indices = torch.arange(step * distill_batch, (step + 1) * distill_batch if (step + 1) * distill_batch < len(distill_dataset) else len(distill_dataset), device=device, dtype=torch.long)
                    else:
                        begin = random.randint(0, len(distill_dataset) - distill_batch)
                        indices = torch.arange(begin, begin+distill_batch, device=device)
                    batch_data = distill_dataset.get_batch(indices)
                    # print("Pos requires_grad:", distill_dataset.pos.requires_grad)
                    # print("Batch pos requires_grad:", batch_data.pos.requires_grad)
                    # print("Batch pos grad_fn:", batch_data.pos.grad_fn)
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
                        f_loss = force_criterion(force, batch_data.force)
                        loss += f_loss
                    grad = torch.autograd.grad(loss, student_params_list[-1], create_graph=True, retain_graph=True)[0]
                    if torch.isnan(grad).sum().item() > 0:
                        print(f'nan values exist in dynamic grad.')
                    student_params_list.append(dynamic_optimizer.step(student_params_list[-1], grad, distill_dataset.get_lr()))
                
                param_loss = torch.tensor(0.0).to(device)
                param_dist = torch.tensor(0.0).to(device)
                param_loss += torch.nn.functional.mse_loss(student_params_list[-1], target_expert_params, reduction="sum")
                param_dist += torch.nn.functional.mse_loss(start_expert_params, target_expert_params, reduction="sum")

                # param_dist_list.append(param_dist)

                param_loss /= num_params
                param_dist /= num_params

                param_loss /= param_dist
                param_loss_list.append(param_loss.detach().cpu().item())
                
                param_loss.backward()
                if self.optimizer_pos is not None:
                    pos_grad = distill_dataset.get_pos().grad.abs()
                    if torch.isnan(pos_grad).sum().item() > 0:
                        print(f'pos_grad.shape: {pos_grad.shape}')
                        print(f'pos_grad is NAN: {torch.isnan(pos_grad).sum().item()}')
                        print(f'pos_grad is zero: {(torch.abs(pos_grad) < 1e-5).sum().item()}')
                        print(f"pos gradient for param loss: max={pos_grad.max()}, min={pos_grad.min()}, mean={pos_grad.mean()}")
                        

                if max_grad_norm_clip:
                    if self.optimizer_pos is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=distill_dataset.get_pos(), max_norm=max_grad_norm_clip)
                    
                    if self.optimizer_assistant is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=assistant_net.parameters(), max_norm=max_grad_norm_clip)
                    
                    if self.optimizer_lr is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=distill_dataset.get_lr(), max_norm=max_grad_norm_clip)
                        
                    if self.optimizer_energy is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=distill_dataset.get_y(), max_norm=max_grad_norm_clip)
                        
                    if self.optimizer_force is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=distill_dataset.get_force(), max_norm=max_grad_norm_clip)

                if self.optimizer_assistant is not None:
                    self.optimizer_assistant.step()
                    if self.scheduler_assistant is not None:
                        self.scheduler_assistant.step()
                    
                if self.optimizer_pos is not None:
                    self.optimizer_pos.step()
                    if self.scheduler_pos is not None:
                        self.scheduler_pos.step()
                        
                if self.optimizer_lr is not None:
                    self.optimizer_lr.step()
                    if self.scheduler_lr is not None:
                        self.scheduler_lr.step()
                        
                if self.optimizer_energy is not None:
                    self.optimizer_energy.step()
                    if self.scheduler_energy is not None:
                        self.scheduler_energy.step()
                        
                if self.optimizer_force is not None:
                    self.optimizer_force.step()
                    if self.scheduler_force is not None:
                        self.scheduler_force.step()
                
                if enable_log:
                    wandb.log({"start_it": start_it, "end_it": end_it, "distill/param_loss": param_loss.detach().cpu().item(), "distill/mean_param_loss": sum(param_loss_list) / len(param_loss_list), "distill/min_param_loss": min(param_loss_list), "distill/dynamic_lr": distill_dataset.get_lr(detach=True).item()}, step=it)
                print(f'iter: {it}, distill_loss: {param_loss.detach().cpu().item()}, mean_distill_loss: {sum(param_loss_list) / len(param_loss_list)}, min_distill_loss: {min(param_loss_list)}, "distill/dynamic_lr": {distill_dataset.get_lr(detach=True).item()}')
                if self.optimizer_pos is not None:
                    print(f'distill_pos_lr: {self.optimizer_pos.param_groups[0]["lr"]}')
                if self.optimizer_assistant is not None:
                    print(f'distill_assistant_lr: {self.optimizer_assistant.param_groups[0]["lr"]}')
                if self.optimizer_lr is not None:
                    print(f'distill_lr_lr: {self.optimizer_lr.param_groups[0]["lr"]}')
                if self.optimizer_energy is not None:
                    print(f'distill_energy_lr: {self.optimizer_energy.param_groups[0]["lr"]}')
                if self.optimizer_force is not None:
                    print(f'distill_force_lr: {self.optimizer_force.param_groups[0]["lr"]}')
                    

                pbar.set_postfix(param_loss=param_loss.detach().cpu(), step=it)

                if revise_energy_and_force and best_expert_net is not None:
                    self._revise_energy_and_force(distill_dataset, best_expert_net, batch_size=distill_batch, device=device, enable_log=enable_log, step=it, energy_and_force=distill_energy_and_force)

                if check_energy_and_force and best_expert_net is not None:
                    self._check_energy_and_force(distill_dataset, best_expert_net, batch_size=distill_batch, device=device, enable_log=enable_log, step=it, energy_and_force=distill_energy_and_force)
                
                for _ in student_params_list:
                    del _
                del output, batch_data, grad
                if distill_energy_and_force:
                    del force
                torch.cuda.empty_cache()
        
        if enable_log:
            wandb.finish()

    def _revise_energy_and_force(self, distill_dataset, model, batch_size, device, enable_log, step, energy_and_force: bool=True):
        print(f'revising energy and force in distill dataset')
        num_steps = (len(distill_dataset) + batch_size - 1) // batch_size
        update_info = None
        for i in range(num_steps):
            indices = torch.arange(i * batch_size, (i + 1) * batch_size if (i + 1) * batch_size < len(distill_dataset) else len(distill_dataset), device=device, dtype=torch.long)
            batch_data = distill_dataset.get_batch(indices)
            if energy_and_force:
                batch_data.pos.requires_grad_(True)
            output = model(batch_data)
            if energy_and_force:
                force = -torch.autograd.grad(outputs=output, inputs=batch_data.pos, 
                            grad_outputs=torch.ones_like(output), retain_graph=True)[0]
            temp_update_info = distill_dataset.update_batch(idx=indices, energy=output.squeeze(dim=1).detach().clone(), force=force.detach().clone())
            if update_info is None:
                update_info = temp_update_info
                for key in update_info:
                    update_info[key]['mean'] = update_info[key]['mean'] / num_steps
            else:
                for key in temp_update_info:
                    if key in update_info:
                        for sub_key in temp_update_info[key]:
                            if sub_key in update_info[key]:
                                if sub_key == 'max':
                                    update_info[key][sub_key] = max(update_info[key][sub_key], temp_update_info[key][sub_key])
                                if sub_key == 'min':
                                    update_info[key][sub_key] = min(update_info[key][sub_key], temp_update_info[key][sub_key])
                                if sub_key == 'mean':
                                    update_info[key][sub_key] = update_info[key][sub_key] + temp_update_info[key][sub_key] / num_steps
            if energy_and_force:
                del force
            del output
            
            torch.cuda.empty_cache()
        if enable_log:
            for key in update_info:
                wandb.log({
                    f"{key}/mean": update_info[key]['mean'],
                    f"{key}/max": update_info[key]['max'],
                    f"{key}/min": update_info[key]['min'],
                }, step=step)
            
        print(update_info)
        return update_info
    
    
    def _check_energy_and_force(self, distill_dataset, model, batch_size, device, enable_log, step, energy_and_force: bool=True):
        print(f'checking energy and force in distill dataset')
        num_steps = (len(distill_dataset) + batch_size - 1) // batch_size
        check_info = None
        for i in range(num_steps):
            indices = torch.arange(i * batch_size, (i + 1) * batch_size if (i + 1) * batch_size < len(distill_dataset) else len(distill_dataset), device=device, dtype=torch.long)
            batch_data = distill_dataset.get_batch(indices)
            if energy_and_force:
                batch_data.pos.requires_grad_(True)
            output = model(batch_data)
            if energy_and_force:
                force = -torch.autograd.grad(outputs=output, inputs=batch_data.pos, 
                            grad_outputs=torch.ones_like(output), retain_graph=True)[0]
            temp_check_info = distill_dataset.check_batch(idx=indices, energy=output.squeeze(dim=1).detach().clone(), force=force.detach().clone())
            if check_info is None:
                check_info = temp_check_info
                for key in check_info:
                    check_info[key]['mean'] = check_info[key]['mean'] / num_steps
            else:
                for key in temp_check_info:
                    if key in check_info:
                        for sub_key in temp_check_info[key]:
                            if sub_key in check_info[key]:
                                if sub_key == 'max':
                                    check_info[key][sub_key] = max(check_info[key][sub_key], temp_check_info[key][sub_key])
                                if sub_key == 'min':
                                    check_info[key][sub_key] = min(check_info[key][sub_key], temp_check_info[key][sub_key])
                                if sub_key == 'mean':
                                    check_info[key][sub_key] = check_info[key][sub_key] + temp_check_info[key][sub_key] / num_steps
            if energy_and_force:
                del force
            del output
            
            torch.cuda.empty_cache()
            
        if enable_log:
            for key in check_info:
                wandb.log({
                    f"{key}/mean": check_info[key]['mean'],
                    f"{key}/max": check_info[key]['max'],
                    f"{key}/min": check_info[key]['min'],
                }, step=step)
            
        print(check_info)
        return check_info