import os
import sys
sys.path.append('.')
import yaml
import argparse
from utils.dataset import DistillDatset
from utils.net import get_network

import torch
from utils.dataset import get_dataset, split_dataset
from utils.run.trainer import Trainer

os.environ["WANDB_MODE"] = "offline"

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
    config_path = os.path.join(args.distill_dir, 'config.yaml')
    config = load_config(config_path)
    
    # Parse configurations
    distill_cfg, data_cfg, network_cfg = config['distill_cfg'], config['data_cfg'], config['network_cfg']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset and split into train/valid
    dataset = get_dataset(**data_cfg)
    train_dataset, valid_dataset, _ = split_dataset(dataset=dataset, **data_cfg)
    distill_dataset = DistillDatset.load(load_path=os.path.join(distill_cfg['save_dir'], 'best_valid.pt'), device=device)
    
    model = get_network(**network_cfg)
    os.makedirs(os.path.join(args.distill_dir, 'eval_best_valid'), exist_ok=True)
    trainer = Trainer()
    trainer.train(
        device=device, 
        train_dataset=distill_dataset.to_torch(), 
        valid_dataset=valid_dataset, 
        test_dataset=None, 
        model=model, 
        assistant_model=None,
        epochs=distill_cfg['eval_train_epoch'] if train_epoch is None else train_epoch, 
        batch_size=distill_cfg['distill_batch'], 
        vt_batch_size=distill_cfg['eval_vt_batch_size'], 
        optimizer_name=distill_cfg['dynamic_optimizer_type'], 
        lr=distill_dataset.get_lr(detach=True), 
        scheduler_name=distill_cfg['eval_scheduler_name'], 
        lr_decay_factor=distill_cfg['eval_lr_decay_factor'],
        lr_decay_step_size=distill_cfg['eval_lr_decay_factor'], 
        energy_and_force=distill_cfg['force_requires_grad'], 
        p=distill_cfg['p'], 
        save_dir=os.path.join(args.distill_dir, 'eval_best_valid'), 
        project_name='MoleculeDynamics-Expert-Trajectory',
        val_step=1,
        test_step=10,
        save_step=50, 
        early_epoch=-1, 
        early_save_iters = 50,
        enable_log=True, 
    )
    
if __name__ == "__main__":
    main()