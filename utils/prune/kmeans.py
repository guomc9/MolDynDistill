import torch
import numpy as np
from sklearn.cluster import KMeans as _KMeans
from utils.dataset import PruneDataset

class KMeans:
    def __init__(self, prune_rate, n_clusters=10, cluster_keys=['pos', 'y', 'force'], **kwargs):
        self.n_clusters = n_clusters
        self.prune_rate = prune_rate
        self.cluster_keys = cluster_keys
        
    def __call__(self, dataset):
        features = []
        for data in dataset:
            data_features = []
            for key in self.cluster_keys:
                if hasattr(data, key):
                    feat = getattr(data, key)
                    if isinstance(feat, torch.Tensor):
                        feat = feat.view(-1)
                    if isinstance(feat, (int, float)):
                        feat = [feat]
                    data_features.extend(feat.cpu().numpy())
            features.append(data_features)
            
        features = np.array(features)
        
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        kmeans = _KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        samples_per_cluster = int(len(dataset) * self.prune_rate) // self.n_clusters
        selected_indices = []
        
        for i in range(self.n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            if len(cluster_indices) > 0:
                if len(cluster_indices) <= samples_per_cluster:
                    selected_indices.extend(cluster_indices)
                else:
                    sampled = np.random.choice(cluster_indices, samples_per_cluster, replace=False)
                    selected_indices.extend(sampled)
        
        return PruneDataset([dataset[i] for i in selected_indices])
