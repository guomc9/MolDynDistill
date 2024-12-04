import os
import sys
import yaml
import argparse
from datetime import datetime
sys.path.append('.')
import random
import numpy as np
import torch
from utils.dataset import get_dataset, split_dataset
from utils.distill import get_distill_algorithm
os.environ["WANDB_MODE"] = "offline"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Distillation script with YAML configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Path to the YAML configuration file'
    )
    
    parser.add_argument(
        '-s',
        '--save_dir',
        type=str,
        help='Save directory'
    )
    
    parser.add_argument(
        '-e',
        '--expert_dir',
        type=str,
        help='Expert trajectory directory'
    )
    
    return parser.parse_args()

def load_config(config_path, args):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.expert_dir:
            config['distill_cfg']['expert_trajectory_dir'] = args.expert_dir
            
        expert_config_path = os.path.join(config['distill_cfg']['expert_trajectory_dir'], 'config.yaml')
        with open(expert_config_path, 'r') as f:
            expert_config = yaml.safe_load(f)
            config['data_cfg'] = expert_config['data_cfg']
            config['distill_cfg']['p'] = expert_config['train_cfg']['p']
            config['network_cfg'] = expert_config['network_cfg']
            
        if args.save_dir:
            config['distill_cfg']['save_dir'] = args.save_dir
        else:
            config['distill_cfg']['save_dir'] = os.path.join(
                '.log',
                'data_distill', 
                config['distill_cfg']["algorithm"],
                config['data_cfg']["dataset_name"], 
                '' if config['data_cfg']["name"] is None else config['data_cfg']["name"],
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            )
            os.makedirs(config['distill_cfg']['save_dir'])
        
        
        cfg_save_path = os.path.join(config['distill_cfg']['save_dir'], 'config.yaml')
        with open(cfg_save_path, 'w') as f:
            yaml.safe_dump(config, f)
            
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")

def main():
    args = parse_args()
    config = load_config(args.config, args)
    
    # Parse configurations
    distill_cfg, data_cfg, expert_network_cfg = config['distill_cfg'], config['data_cfg'], config['network_cfg']
    set_seed(data_cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset and split into train/valid
    dataset = get_dataset(**data_cfg)
    train_dataset, valid_dataset, _ = split_dataset(dataset=dataset, **data_cfg)
    
    # Initialize algorithm for distillation
    expert_network_name = expert_network_cfg.pop('name')
    expert_network_cfg['num_clusters'] = distill_cfg.pop('num_clusters') if 'num_clusters' in distill_cfg.keys() else None
    distiller = get_distill_algorithm(algorithm=distill_cfg['algorithm'])
    
    # # Perform distillation
    # distiller.distill(
    #     project_name=distill_cfg['project_name'],
    #     num_iteration=distill_cfg['num_iteration'],
    #     num_step_per_iteration=distill_cfg['num_step_per_iteration'],
    #     enable_assistant_net=distill_cfg['enable_assistant_net'], 
    #     expert_network_name=expert_network_name, 
    #     expert_network_dict=expert_network_cfg, 
    #     expert_trajectory_dir=distill_cfg['expert_trajectory_dir'],
    #     max_start_epoch=distill_cfg['max_start_epoch'],
    #     num_expert_epoch=distill_cfg['num_expert_epoch'],
    #     eval_step=distill_cfg['eval_step'],
    #     eval_network_pool=distill_cfg['eval_network_pool'],
    #     eval_train_epoch=distill_cfg['eval_train_epoch'],
    #     eval_batch_size=distill_cfg['eval_batch_size'],
    #     eval_vt_batch_size=distill_cfg['eval_vt_batch_size'],
    #     eval_lr_decay_factor=distill_cfg['eval_lr_decay_factor'],
    #     eval_lr_decay_step_size=distill_cfg['eval_lr_decay_step_size'],
    #     eval_scheduler_name=distill_cfg['eval_scheduler_name'], 
    #     save_step=distill_cfg['save_step'],
    #     save_dir=distill_cfg['save_dir'],
    #     distill_rate=distill_cfg['distill_rate'],
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     device=device,
    #     distill_batch=distill_cfg['distill_batch'],
    #     distill_lr_assistant_net=distill_cfg['distill_lr_assistant_net'],
    #     distill_lr_lr=distill_cfg['distill_lr_lr'],
    #     distill_base_lr=distill_cfg['distill_base_lr'],
    #     distill_lr_pos=distill_cfg['distill_lr_pos'], 
    #     distill_energy_and_force=distill_cfg['distill_energy_and_force'],
    #     lr_requires_grad=distill_cfg['lr_requires_grad'], 
    #     pos_requires_grad=distill_cfg['pos_requires_grad'], 
    #     energy_requires_grad=distill_cfg['energy_requires_grad'],
    #     force_requires_grad=distill_cfg['force_requires_grad'], 
    #     p=distill_cfg['p']
    # )
    
    # Perform distillation
    distiller.distill(
        expert_network_name=expert_network_name, 
        expert_network_dict=expert_network_cfg, 
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        device=device,
        **distill_cfg
    )

if __name__ == "__main__":
    main()