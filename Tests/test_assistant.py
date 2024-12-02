import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch

class SchNetAdapter(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_interactions=3, num_clusters=100):
        super(SchNetAdapter, self).__init__()
        self.num_interactions = num_interactions
        self.num_layers = self.num_interactions + 2
        self.atom_embedder = torch.nn.Embedding(num_embeddings=100, embedding_dim=hidden_channels)
        self.cluster_embedder = torch.nn.Embedding(num_embeddings=num_clusters, embedding_dim=hidden_channels)
        self.in_layers = torch.nn.ModuleList()
        self.out_layers = torch.nn.ModuleList()
        self.input_dim = 3 + hidden_channels * 2
        input_dim = self.input_dim
        for _ in range(self.num_layers-1):
            self.in_layers.append(torch.nn.Linear(input_dim, hidden_channels))
            self.out_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            # input_dim = hidden_channels
        self.in_layers.append(torch.nn.Linear(input_dim, hidden_channels))
        self.out_layers.append(torch.nn.Linear(hidden_channels, hidden_channels // 2))
        
        self.activation = torch.nn.ReLU()
        self._init_weights()

    def forward(self, batch_data):
        x, z, cz = batch_data.pos, batch_data.z, batch_data.cz
        x = torch.cat([x, self.atom_embedder(z), self.cluster_embedder(cz)], dim=-1)
        outputs = []
        for in_layer, out_layer in zip(self.in_layers, self.out_layers):
            # x = out_layer(self.activation(in_layer(x)))
            outputs.append(out_layer(self.activation(in_layer(x))))
        return outputs

    def _init_weights(self):
        for out_layer in self.out_layers:
            if isinstance(out_layer, torch.nn.Linear):
                torch.nn.init.zeros_(out_layer.weight)  # Set the weight matrix to all zeros
                if out_layer.bias is not None:
                    torch.nn.init.zeros_(out_layer.bias)  # Set the bias vector to all zeros
        
        for in_layer in self.in_layers:
            if isinstance(in_layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(in_layer.weight)  # Set the weight matrix to all zeros
                if in_layer.bias is not None:
                    torch.nn.init.zeros_(in_layer.bias)  # Set the bias vector to all zeros
        
        
# 模拟的冻结网络
class FrozenNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 1)  # 假设hidden_channels//2 = 64
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.linear(x)

# # 测试代码
# def test_gradient_flow():
#     # 创建模型
#     adapter = SchNetAdapter(hidden_channels=128)
#     adapter.requires_grad_(True)
#     frozen_net = FrozenNetwork()
    
#     # 创建示例数据
#     pos = torch.randn(10, 3)  # 10个原子，每个3维坐标
#     z = torch.randint(0, 100, (10,))  # 原子类型
#     cz = torch.randint(0, 100, (10,))  # 簇类型
#     batch_data = Data(pos=pos, z=z, cz=cz)
    
#     # 注册梯度钩子
#     gradient_flow = {}
#     def make_hook(name):
#         def hook(grad):
#             gradient_flow[name] = True
#         return hook
    
#     # 为adapter的参数注册钩子
#     hooks = []
#     for name, param in adapter.named_parameters():
#         gradient_flow[name] = False
#         hooks.append(param.register_hook(make_hook(name)))
    
#     # 前向传播
#     adapter_outputs = adapter(batch_data)
#     frozen_output = frozen_net(adapter_outputs[-1])
#     loss = frozen_output.mean()
    
#     # 反向传播
#     loss.backward()
    
#     # 检查梯度
#     print("\n检查参数梯度:")
#     for name, param in adapter.named_parameters():
#         print(f"\n{name}:")
#         print(f"梯度是否传播: {gradient_flow[name]}")
#         print(f"梯度是否为None: {param.grad is None}")
#         if param.grad is not None:
#             print(f"梯度范数: {param.grad.norm()}")
#             print(f"梯度是否全0: {(param.grad == 0).all()}")
    
#     # 清理钩子
#     for hook in hooks:
#         hook.remove()
    
#     return gradient_flow

# # 运行测试
# gradient_flow = test_gradient_flow()

# # 验证梯度更新
# def verify_parameter_updates():
#     adapter = SchNetAdapter(hidden_channels=128)
#     frozen_net = FrozenNetwork()
    
#     # 保存初始参数的副本
#     initial_params = {name: param.clone() for name, param in adapter.named_parameters()}
    
#     # 创建优化器
#     optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    
#     # 训练几步
#     for i in range(3):
#         # 创建新的随机数据
#         pos = torch.randn(10, 3)
#         z = torch.randint(0, 100, (10,))
#         cz = torch.randint(0, 100, (10,))
#         batch_data = Data(pos=pos, z=z, cz=cz)
        
#         # 前向传播
#         adapter_outputs = adapter(batch_data)
#         frozen_output = frozen_net(adapter_outputs[-1])
#         loss = frozen_output.mean()
        
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         print(f"\n步骤 {i+1}:")
#         print(f"Loss: {loss.item()}")
        
#         # 检查参数是否更新
#         for name, param in adapter.named_parameters():
#             param_change = (param.data - initial_params[name]).norm()
#             print(f"{name} 参数变化范数: {param_change}")

# verify_parameter_updates()

# 添加一个完整的训练循环测试
def full_training_test():
    adapter = SchNetAdapter(hidden_channels=128)
    frozen_net = FrozenNetwork()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    
    losses = []
    param_changes = {name: [] for name, _ in adapter.named_parameters()}
    initial_params = {name: param.clone() for name, param in adapter.named_parameters()}
    
    for epoch in range(2):
        batch_size = 32
        pos = torch.randn(batch_size, 10, 3)
        z = torch.randint(0, 100, (batch_size, 10))
        cz = torch.randint(0, 100, (batch_size, 10))
        
        total_loss = 0
        for i in range(batch_size):
            batch_data = Data(pos=pos[i], z=z[i], cz=cz[i])
            
            adapter_outputs = adapter(batch_data)
            frozen_output = frozen_net(adapter_outputs[-1])
            loss = frozen_output.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录参数变化
            for name, param in adapter.named_parameters():
                param_changes[name].append(
                    (param.data - initial_params[name]).norm().item()
                )
        
        avg_loss = total_loss / batch_size
        losses.append(avg_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Average Loss: {avg_loss}")
        print("参数变化:")
        for name in param_changes:
            print(f"{name}: {param_changes[name][-1]}")
    
    return losses, param_changes

losses, param_changes = full_training_test()