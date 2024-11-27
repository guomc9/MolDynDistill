import torch

class DyncAdam:
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, t=0, **kwargs):
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a single torch.Tensor")

        self.params = params
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.m = torch.zeros_like(params)
        self.v = torch.zeros_like(params)

        self.t = t

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor | float):
        if grad is None or not isinstance(grad, torch.Tensor):
            raise ValueError(f"grad must be a torch.Tensor. But grad is {type(grad)}.")
        if lr is None or not (isinstance(lr, float) or isinstance(lr, torch.Tensor)):
            raise TypeError(f"lr must be a float or torch.Tensor. But lr is {type(lr)}.")

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        print(f't: {self.t}, adam fix scale: {(m_hat / (torch.sqrt(v_hat) + self.eps)).abs().max()}')
        # updated_params = params.clone()
        # updated_params -= lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        # return updated_params
        return params - lr * m_hat / (torch.sqrt(v_hat) + self.eps)