import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch_geometric as pyg
from datetime import datetime
sys.path.append('.')

from functools import partial
import torch
from utils.dataset import get_dataset, split_dataset
from utils.net import get_network
from utils.run.trainer import Trainer
from utils.prune import get_prune_algorithm

os.environ["WANDB_MODE"] = "offline"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    pyg.seed_everything(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Training script with YAML configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Path to the YAML configuration file'
    )
    
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        help='Dataset name to override config'
    )
    
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        help='Dataset subset name to override config'
    )
    
    parser.add_argument(
        '-s',
        '--save_dir',
        type=str,
        help='Save directory'
    )
    
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='Train epoch'
    )
        
    parser.add_argument(
        '-w',
        '--wandb_run_id',
        type=str,
        help='wandb run id for resume'
    )
    return parser.parse_args()

def load_config(config_path, args):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.dataset_name:
            config['data_cfg']['dataset_name'] = args.dataset_name
            
        if args.name:
            config['data_cfg']['name'] = args.name
            
        if args.epochs:
            config['train_cfg']['epochs'] = args.epochs
            
        if args.save_dir:
            config['train_cfg']['save_dir'] = args.save_dir
        else:
            config['train_cfg']['save_dir'] = os.path.join(
                '.log',
                'expert_trajectory', 
                config['network_cfg']["name"],
                config['data_cfg']["dataset_name"],
                '' if config['data_cfg']["name"] is None else config['data_cfg']["name"],
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            )
            os.makedirs(config['train_cfg']['save_dir'])
        cfg_save_path = os.path.join(config['train_cfg']['save_dir'], 'config.yaml')
        with open(cfg_save_path, 'w') as f:
            yaml.safe_dump(config, f)
            
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")

def main():
    args = parse_args()
    config = load_config(args.config, args)
    prune_cfg, data_cfg, network_cfg, train_cfg = config['prune_cfg'], config['data_cfg'], config['network_cfg'], config['train_cfg']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(data_cfg['seed'])
    seed_hook = partial(set_seed, seed=data_cfg['seed'])
    dataset = get_dataset(**data_cfg)
    
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset=dataset, **data_cfg)
    
    pruner = get_prune_algorithm(**prune_cfg)
    prune_dataset = pruner(train_dataset)
    
    seed_hook()
    network = get_network(**network_cfg)
    
    trainer = Trainer()
    
    cfg = train_cfg.copy()
    cfg.update(config['network_cfg'])
    cfg.update(config['data_cfg'])
    trainer.train(
        device=device,
        train_dataset=prune_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        model=network,
        wandb_run_id=args.wandb_run_id, 
        seed_hook=seed_hook, 
        **cfg
    )

if __name__ == "__main__":
    main()