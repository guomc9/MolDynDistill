import numpy as np
from utils.dataset import PruneDataset

class Box:
    def __init__(self, prune_rate, n_boxes=10, **kwargs):
        self.n_boxes = n_boxes
        self.prune_rate = prune_rate

    def __call__(self, dataset):
        energies = []
        for data in dataset:
            energies.append(data.y.item())
        energies = np.array(energies)
        
        min_e, max_e = energies.min(), energies.max()
        box_edges = np.linspace(min_e, max_e, self.n_boxes + 1)
        
        box_indices = np.digitize(energies, box_edges) - 1
        
        samples_per_box = int(len(dataset) * self.prune_rate) // self.n_boxes
        
        selected_indices = []
        for i in range(self.n_boxes):
            box_data_indices = np.where(box_indices == i)[0]
            if len(box_data_indices) > 0:
                if len(box_data_indices) <= samples_per_box:
                    selected_indices.extend(box_data_indices)
                else:
                    sampled = np.random.choice(box_data_indices, samples_per_box, replace=False)
                    selected_indices.extend(sampled)
                    
        return PruneDataset([dataset[i] for i in selected_indices])
