from .dync_sgd import DyncSGD
from .dync_adam import DyncAdam

def get_dynamic_optimizer(optimizer_type, params, **kwargs):
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "sgd":
        return DyncSGD(params=params, **kwargs)
    elif optimizer_type == "adam":
        return DyncAdam(params=params, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported types: 'sgd', 'adam'.")