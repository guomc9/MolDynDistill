import torch

class DyncAdam:
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, lr=None, t=None, m=None, v=None, v_hat_eps=1e-44, **kwargs):
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a single torch.Tensor")

        self.params = params
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.v_hat_eps = v_hat_eps
        self.m = torch.zeros_like(params) if m is None or not isinstance(m, torch.Tensor) else m
        self.v = torch.zeros_like(params) if v is None or not isinstance(v, torch.Tensor) else v
        self.t = torch.tensor(0) if t is None or not isinstance(t, torch.Tensor) else t
        self.lr = lr

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor | float=None):
        if grad is None or not isinstance(grad, torch.Tensor):
            raise ValueError(f"grad must be a torch.Tensor. But grad is {type(grad)}.")
        
        if lr is None or not (isinstance(lr, float) or isinstance(lr, torch.Tensor)):
            if self.lr is not None:
                lr = self.lr
            else:
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
        
        print(f'before clamp: v_hat.mean: {torch.mean(v_hat)}, max: {torch.max(v_hat)}, min: {torch.min(v_hat)}')
        v_hat = torch.clamp(v_hat, min=self.v_hat_eps)
        print(f'after clamp: v_hat.mean: {torch.mean(v_hat)}, max: {torch.max(v_hat)}, min: {torch.min(v_hat)}')
        return params - lr * m_hat / (torch.sqrt(v_hat) + self.eps)