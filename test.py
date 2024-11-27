import torch

class DataContainer:
    def __init__(self):
        # 创建一个2D tensor作为成员变量
        self.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    def get_data_slice(self):
        return self.data[0]  # 返回第一行

# 测试代码
container = DataContainer()
x = container.get_data_slice()

# 检查状态
print("x.requires_grad:", x.requires_grad)  # True
print("x.grad_fn:", x.grad_fn)             # <SelectBackward0>
print("x.is_leaf:", x.is_leaf)             # False
print("x.grad:", x.grad)

# 进行运算和反向传播
y = x.sum()
y.backward()

# 检查梯度
print("container.data.grad:", container.data.grad)  
# tensor([[1., 1.],
#         [0., 0.]])