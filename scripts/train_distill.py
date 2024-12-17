import os
import sys
sys.path.append('.')
import yaml
import argparse
from utils.dataset import DistillDataset
from utils.net import get_network
import random
import numpy as np
import torch
import torch_geometric as pyg
from utils.dataset import get_dataset, split_dataset
from utils.run.trainer import Trainer
from functools import partial
import shutil

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
        description='Evaluate distill data with YAML configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-d',
        '--distill_dir',
        type=str,
        help='Distill directory'
    )
    
    parser.add_argument(
        '-e',
        '--epoch',
        type=int, 
        help='Epoch for training distill data', 
        default=None
    )
    
    parser.add_argument(
        '-c',
        '--ckpt_id',
        type=int, 
        help='Checkpoint id for distill data', 
        default=None
    )
    
    parser.add_argument(
        '-s', 
        '--save_dir', 
        type=str, 
        help='save directory of model checkpoints', 
        default=None
    )
       
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int, 
        help='Batch size', 
        default=None
    )
    
    parser.add_argument(
        '--no_scheduler',
        type=bool, 
        help='no scheduler', 
        default=False
    )
    
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {str(e)}")

def main():
    args = parse_args()
    train_epoch = args.epoch
    batch_size = args.batch_size
    config_path = os.path.join(args.distill_dir, 'config.yaml')
    config = load_config(config_path)
    save_dir = args.save_dir
    
    # Parse configurations
    distill_cfg, data_cfg, network_cfg = config['distill_cfg'], config['data_cfg'], config['network_cfg']
    set_seed(data_cfg['seed'])
    seed_hook = partial(set_seed, seed=data_cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset and split into train/valid
    dataset = get_dataset(**data_cfg)
    _, valid_dataset, _ = split_dataset(dataset=dataset, **data_cfg)
    
    if args.ckpt_id is None:
        distill_dataset = DistillDataset.load(load_path=os.path.join(distill_cfg['save_dir'], 'best_valid.pt'), device=device)
    else:
        distill_dataset = DistillDataset.load(load_path=os.path.join(distill_cfg['save_dir'], f'{args.ckpt_id}.pt'), device=device)
    seed_hook()
    model = get_network(**network_cfg)
    # model.requires_grad_(True)
    if os.path.exists(os.path.join(args.distill_dir, 'eval_best_valid')):
        shutil.rmtree(os.path.join(args.distill_dir, 'eval_best_valid'))
    os.makedirs(os.path.join(args.distill_dir, 'eval_best_valid'))
    trainer = Trainer()
    trainer.train(
        device=device, 
        train_dataset=distill_dataset.to_torch(), 
        valid_dataset=valid_dataset, 
        test_dataset=None, 
        model=model, 
        assistant_model=None,
        epochs=distill_cfg['eval_train_epoch'] if train_epoch is None else train_epoch, 
        batch_size=distill_cfg['distill_batch'] if batch_size is None else batch_size, 
        vt_batch_size=distill_cfg['eval_vt_batch_size'], 
        optimizer_name=distill_cfg['dynamic_optimizer_type'], 
        lr=distill_dataset.get_lr(detach=True), 
        scheduler_name=distill_cfg['eval_scheduler_name'] if not args.no_scheduler else None, 
        lr_decay_factor=distill_cfg['eval_lr_decay_factor'] if not args.no_scheduler else None, 
        lr_decay_step_size=distill_cfg['eval_lr_decay_step_size'] if not args.no_scheduler else None, 
        energy_and_force=network_cfg['energy_and_force'], 
        p=distill_cfg['p'], 
        save_dir=save_dir if save_dir is not None else os.path.join(args.distill_dir, 'eval_best_valid'), 
        project_name='MoleculeDynamics',
        val_step=1,
        test_step=10,
        save_step=50, 
        early_epoch=-1, 
        early_save_iters = 50,
        enable_log=True, 
        seed_hook=seed_hook, 
    )
    
if __name__ == "__main__":
    main()