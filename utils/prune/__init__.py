from .box import Box
from .kmeans import KMeans

def get_prune_algorithm(algorithm, prune_rate, **kwargs):
    if algorithm.lower() == 'kmeans':
        return KMeans(prune_rate=prune_rate, **kwargs)
    elif algorithm.lower() == 'box':
        return Box(prune_rate=prune_rate, **kwargs)
    else:
        raise ValueError(f"Unknown prune algorithm: {algorithm}, choosing from 'kmeans' or 'box'. ")