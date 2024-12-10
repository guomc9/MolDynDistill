import torch
import numpy as np
from pathlib import Path

def compare_model_parameters(model_path1, model_path2, rtol=1e-5, atol=1e-8):
    """
    比较两个PyTorch模型参数的差异
    
    Args:
        model_path1: 第一个模型的路径
        model_path2: 第二个模型的路径
        rtol: 相对容差 (默认: 1e-5)
        atol: 绝对容差 (默认: 1e-8)
    """
    state_dict1 = torch.load(model_path1)['model_state_dict']
    state_dict2 = torch.load(model_path2)['model_state_dict']
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        print("模型参数名称不完全相同:")
        print("仅在模型1中存在:", keys1 - keys2)
        print("仅在模型2中存在:", keys2 - keys1)
        return
    
    # 比较每个参数
    differences = []
    for key in keys1:
        param1 = state_dict1[key].cpu().numpy()
        param2 = state_dict2[key].cpu().numpy()
        
        # 检查形状是否相同
        if param1.shape != param2.shape:
            print(f"参数 {key} 的形状不同:")
            print(f"模型1形状: {param1.shape}")
            print(f"模型2形状: {param2.shape}")
            continue
            
        # 计算差异
        if not np.allclose(param1, param2, rtol=rtol, atol=atol):
            abs_diff = np.abs(param1 - param2)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            differences.append({
                'name': key,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shape': param1.shape
            })
    
    if differences:
        print("\n发现参数差异:")
        for diff in sorted(differences, key=lambda x: x['max_diff'], reverse=True):
            print(f"\n参数名称: {diff['name']}")
            print(f"形状: {diff['shape']}")
            print(f"最大绝对差异: {diff['max_diff']:.8e}")
            print(f"平均绝对差异: {diff['mean_diff']:.8e}")
    else:
        print("\n两个模型的参数完全相同")

if __name__ == "__main__":
    # 使用示例
    model_path1 = ".temp/00/checkpoint_epoch_0.pt"
    # model_path1 = ".log/data_distill/mtt/MD17/benzene/2024-12-05-17-05-50/eval/5/checkpoint_epoch_0.pt"
    model_path2 = ".temp/0/checkpoint_epoch_0.pt"
    compare_model_parameters(model_path1, model_path2, 1e-12, 1e-15)