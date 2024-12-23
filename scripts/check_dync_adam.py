import torch
import torch.nn as nn
from torch.optim import Adam

class DyncAdam:
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, t=None, m=None, v=None, **kwargs):
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a single torch.Tensor")

        self.params = params
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.m = torch.zeros_like(params) if m is None or not isinstance(m, torch.Tensor) else m
        self.v = torch.zeros_like(params) if v is None or not isinstance(v, torch.Tensor) else v

        self.t = torch.tensor(0) if t is None or not isinstance(t, torch.Tensor) else t

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor | float):
        if grad is None or not isinstance(grad, torch.Tensor):
            raise ValueError(f"grad must be a torch.Tensor. But grad is {type(grad)}.")
        if lr is None or not (isinstance(lr, float) or isinstance(lr, torch.Tensor)):
            raise TypeError(f"lr must be a float or torch.Tensor. But lr is {type(lr)}.")

        device = params.device

        self.t = self.t.to(device)
        self.m = self.m.to(device)
        self.v = self.v.to(device)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - lr * m_hat / (torch.sqrt(v_hat) + self.eps)

def test_adam_implementations():
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    model_dync = nn.Linear(3, 2)
    model_dync.load_state_dict(model.state_dict())

    torch_adam = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    dync_adams = [
        DyncAdam(param, betas=(0.9, 0.999), eps=1e-8)
        for param in model_dync.parameters()
    ]

    x = torch.randn(5, 3)
    y = torch.randn(5, 2)

    criterion = nn.MSELoss()
    y_pred = model(x)
    loss = criterion(y_pred, y)

    y_pred_dync = model_dync(x)
    loss_dync = criterion(y_pred_dync, y)

    loss.backward()
    loss_dync.backward()

    torch_adam.step()

    with torch.no_grad():
        for param, grad, dync_adam in zip(
            model_dync.parameters(), [p.grad for p in model_dync.parameters()], dync_adams
        ):
            updated_param = dync_adam.step(param, grad, lr=0.01)
            param.copy_(updated_param)

    print("\nComparing parameter updates:")
    for (name, param_torch), param_dync in zip(model.named_parameters(), model_dync.parameters()):
        print(f"{name}:")
        print(f"  Torch Adam: {param_torch.data}")
        print(f"  DyncAdam:  {param_dync.data}")
        assert torch.allclose(param_torch.data, param_dync.data, atol=1e-6), \
            f"Mismatch found in parameter {name}!"

    print("\nAll parameters match between Torch Adam and DyncAdam!")

test_adam_implementations()