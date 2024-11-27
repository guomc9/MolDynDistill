import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

def generate_original_data(num_samples=100):
    x = torch.randn(num_samples, 2) * 2
    y = torch.sin(x[:, 0:1]) + torch.cos(x[:, 1:2])
    return x, y

def analyze_gradients(grads):
    """分析梯度"""
    print(grads)
    return {
        "总梯度范数": grads.norm().item(),
        "平均梯度": grads.abs().mean().item(),
        "最大梯度": grads.abs().max().item(),
        "最小梯度": grads.abs().min().item(),
        "梯度标准差": grads.std().item(),
        "非零梯度比例": (grads != 0).float().mean().item()
    }

def train_distillation(num_epochs=50, num_iters=10, batch_size=4):
    torch.manual_seed(42)
    
    model = SimpleNet()
    for param in model.parameters():
        param.requires_grad = False
    num_samples = 20
    target_x, target_y = generate_original_data(num_samples)
    synthetic_data = torch.randn(num_samples, 2, requires_grad=True)
    optimizer = optim.Adam([synthetic_data], lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for iter_idx in range(num_iters):
            indices = torch.randint(0, num_samples, (batch_size,))
            batch_synthetic = synthetic_data[indices]
            batch_target_y = target_y[indices]
            
            optimizer.zero_grad()
            
            output = model(batch_synthetic)
            loss = criterion(output, batch_target_y)
            # 分析所有synthetic_data的梯度
            grad_stats = analyze_gradients(torch.autograd.grad(loss, synthetic_data, retain_graph=True)[0])
            loss.backward()
            
            
            if grad_stats and iter_idx == 0:  # 每个epoch的第一次迭代打印梯度信息
                print(f"\nEpoch {epoch}, Iter {iter_idx}, Loss: {loss.item():.4f}")
                for key, value in grad_stats.items():
                    print(f"{key}: {value:.6f}")
            
            # 梯度裁剪
            if synthetic_data.grad is not None:
                grad_norm = synthetic_data.grad.norm()
                if grad_norm > 1.0:
                    synthetic_data.grad.mul_(1.0 / grad_norm)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_iters
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
            
        if avg_loss < 1e-4:
            print("达到目标精度，提前停止训练")
            break

if __name__ == "__main__":
    train_distillation()