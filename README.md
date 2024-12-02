## Moleculers Dynamics & Data Distillation

## Expert Trajectory

```shell
CUDA_VISIBLE_DEVICES=0 python scripts/train_et.py -c configs/expert_trajectory/schnet_sgd.yaml -d MD17 -n benzene

# resume training
CUDA_VISIBLE_DEVICES=0 python scripts/train_et.py -c configs/expert_trajectory/schnet_ef_1000.yaml -d MD17 -n benzene -e 3000 -s .log/expert_trajectory/schnet/MD17/benzene/2024-11-02-18-07-20
```

## Data Distill

```shell
CUDA_VISIBLE_DEVICES=1 python scripts/distill.py -c configs/data_distill/mtt_lr_force_and_energy.yaml -e .log/expert_trajectory/schnet/MD17/benzene/xxx
```