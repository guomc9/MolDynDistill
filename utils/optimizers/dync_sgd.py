import torch

class DyncSGD:
    def __init__(self, params, momentum=0.0, **kwargs):
        if not isinstance(params, torch.Tensor):
            raise TypeError("params must be a single torch.Tensor.")

        self.params = params
        self.momentum = momentum

        self.velocity = torch.zeros_like(params) if momentum > 0 else None

    def step(self, params: torch.Tensor, grad: torch.Tensor, lr: torch.Tensor | float):
        if grad is None or not isinstance(grad, torch.Tensor):
            raise ValueError(f"grad must be a torch.Tensor. But grad is {type(grad)}.")
        if lr is None or not (isinstance(lr, float) or isinstance(lr, torch.Tensor)):
            raise TypeError(f"lr must be a float or torch.Tensor. But lr is {type(lr)}.")

        # updated_params = params.clone()
        if self.momentum > 0:
            self.velocity = self.momentum * self.velocity - lr * grad
            # updated_params -= self.velocity
            return params - self.velocity
        else:
        #     updated_params -= lr * grad

        # # return updated_params
            return params - lr * grad