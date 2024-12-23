import torch
import torch.nn.functional as F


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
        
        # return params - lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return params - lr * m_hat


def trajectory_matching_test():
    torch.manual_seed(0)
    W = torch.randn(5, 3, requires_grad=True)
    x = torch.randn(3, requires_grad=True)
    y_true = torch.randn(5)
    
    print("Raw x:", x)
    optimizer_A = torch.optim.Adam([W], lr=0.1)

    W_initial = W.clone().detach()
    for _ in range(10):
        optimizer_A.zero_grad()
        y_pred = W @ x
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer_A.step()

    W_prime = W.clone().detach()

    W_prime.requires_grad = False

    optimizer_B = torch.optim.Adam([x], lr=0.1)

    dyncadam_optimizer = DyncAdam(W_initial.clone())
    W_double_prime = W_initial.clone().detach().requires_grad_(True)
    for _ in range(5):
        y_pred = W_double_prime @ x
        loss = F.mse_loss(y_pred, y_true)
        grad_W = torch.autograd.grad(loss, W_double_prime, create_graph=True, retain_graph=True)[0]
        W_double_prime = dyncadam_optimizer.step(W_double_prime, grad_W, lr=0.1)

    alignment_loss = F.mse_loss(W_prime, W_double_prime)

    optimizer_B.zero_grad()
    alignment_loss.backward()
    print(f'x Grad:', x.grad)
    optimizer_B.step()

    print("W_initial:", W_initial)
    print("W_prime:", W_prime)
    print("W_double_prime:", W_double_prime)
    print("Alignment Loss:", alignment_loss.item())
    print("Updated x:", x)

trajectory_matching_test()