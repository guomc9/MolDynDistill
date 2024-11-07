## Moleculers Dynamics & Data Distillation

## Expert Trajectory

```shell
CUDA_VISIBLE_DEVICES=0 python scripts/train_et.py -c configs/expert_trajectory/sphnet.yaml -d MD17 -n benzene
CUDA_VISIBLE_DEVICES=1 python scripts/train_et.py -c configs/expert_trajectory/sphnet_ef.yaml -d MD17 -n benzene
 
# resume training
CUDA_VISIBLE_DEVICES=0 python scripts/train_et.py -c configs/expert_trajectory/schnet_ef_1000.yaml -d MD17 -n benzene -e 3000 -s .log/expert_trajectory/schnet/MD17/benzene/2024-11-02-18-07-20
```

## Data Distill

```shell
CUDA_VISIBLE_DEVICES=1 python scripts/distill.py -c configs/data_distill/mtt_assist.yaml -e .log/expert_trajectory/schnet/MD17/benzene/2024-11-04-17-27-23
```