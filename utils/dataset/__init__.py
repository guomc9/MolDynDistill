import os
import os.path as osp
import torch.utils
import torch.utils.data
from torch_geometric.datasets import QM9
from .md17 import MD17
import torch
from torch.utils.data import Subset
import numpy as np

def split_dataset(dataset, seed, train_ratio=0.1, valid_ratio=0.1, train_size=None, valid_size=None, **kwargs):
    data_size = len(dataset)
    if train_size is None:
        assert train_ratio <= 1. and train_ratio > 0., f"train_ratio({train_ratio}) must be in (0, 1]."
        train_size = int(data_size * train_ratio)
    if valid_size is None:
        assert valid_ratio <= 1. and valid_ratio > 0., f"valid_ratio({valid_ratio}) must be in (0, 1]."
        valid_size = int(data_size * valid_ratio)
    # assert train_ratio + valid_ratio <= 1., f"train_ratio + valid_ratio({train_ratio} + {valid_ratio}) must be not greater than 1."
    
    ids = np.arange(data_size)
    np.random.seed(seed)
    np.random.shuffle(ids)

    train_idx = ids[:train_size]
    val_idx = ids[train_size:train_size + valid_size]
    test_idx = ids[train_size + valid_size:]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    train_dataset, valid_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    return train_dataset, valid_dataset, test_dataset


def get_dataset(dataset_name: str, root: str, name: str = None, **kwargs):
    """
    Factory function to get molecular datasets from PyG (PyTorch Geometric)
    
    Args:
        dataset_name (str): Type of dataset, can be 'MD17' or 'QM9'
        root (str): Root directory to store the dataset
        name (str, optional): Name of molecule for MD17, available options:
                            ['aspirin', 'benzene', 'ethanol', 'malonaldehyde',
                             'naphthalene', 'salicylic acid', 'toluene', 'uracil']
                            Not needed for QM9 dataset
    
    Returns:
        dataset: PyG dataset object (MD17 or QM9)
    
    Examples:
        >>> md17_data = get_molecular_dataset('MD17', './data', 'aspirin')
        >>> qm9_data = get_molecular_dataset('QM9', './data')
    """
    
    # Create directory if not exists
    if not osp.exists(root):
        os.makedirs(root)
    
    # Convert dataset name to uppercase for comparison
    dataset_name = dataset_name.upper()
    
    # Check if dataset type is valid
    valid_datasets = ['MD17', 'QM9']
    if dataset_name not in valid_datasets:
        raise ValueError(f"Dataset must be one of {valid_datasets}")
    
    try:
        if dataset_name == 'MD17':
            # List of valid molecule names for MD17
            valid_names = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde',
                          'naphthalene', 'salicylic acid', 'toluene', 'uracil']
            
            # Check if molecule name is provided and valid
            if name is None:
                raise ValueError("Molecule name must be provided for MD17 dataset")
            if name.lower() not in valid_names:
                raise ValueError(f"Molecule name must be one of {valid_names}")
            
            # Load the MD17 dataset
            return MD17(root=root, name=name)
            
        else:  # dataset == 'QM9'
            # Load the QM9 dataset
            return QM9(root=root)
            
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as PygData
class TorchDistillDataset(TorchDataset):
    def __init__(self, z, pos, y, force, cz, num_atom_per_molecule):
        self.data_list = []
        begin = 0
        for i in range(len(y)):
            end = (i + 1) * num_atom_per_molecule
            self.data_list.append(
                PygData(
                    z=z[begin:end], 
                    pos=pos[begin:end], 
                    y=y[i], 
                    energy=y[i], 
                    force=force[begin:end], 
                    cz=cz[begin:end]
                    )
                )
            begin = end

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
class DistillData:
    def __init__(
        self, 
        z: torch.LongTensor, 
        cz: torch.LongTensor, 
        pos: torch.FloatTensor,
        y: torch.FloatTensor,
        force: torch.FloatTensor, 
        batch: torch.LongTensor
    ):
        self.z = z
        self.cz = cz
        self.pos = pos
        self._y = y
        self.force = force
        self.batch = batch
        
    @property
    def y(self):
        return self._y
    
    @property
    def energy(self):
        return self._y
    
    # def to_(self, device):
    #     self.z = self.z.to(device)
    #     self.pos = self.pos.to(device)
    #     self._y = self._y.to(device)
    #     self.force = self.force.to(device)

    # def requires_grad_(self, requires: bool, enable_y: bool=False, enable_force: bool=False):
    #     self.pos.requires_grad_(requires)
    #     if enable_y:
    #         self._y.requires_grad_(requires)
    #     if enable_force:
    #         self.force.requires_grad_(requires)
        
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
class DistillDatset:
    def __init__(
        self, 
        source_dataset, 
        distill_rate: float, 
        distill_lr: float, 
        device: str, 
        cluster_rate: float=0.5, 
        pos_requires_grad=False, 
        energy_requires_grad=False, 
        force_requires_grad: bool=False, 
        noise_pos: bool=True, 
        ):
        self.source_size = len(source_dataset)
        self.size = int(self.source_size * distill_rate)
        ids = torch.randperm(self.source_size)[:self.size]
        self.pos = source_dataset[ids].pos.clone().to(device)
        if noise_pos:
            N = self.pos.shape[0]
            mean = torch.mean(self.pos, dim=0, keepdim=True)
            std = torch.std(self.pos, dim=0, unbiased=True, keepdim=True)
            print(f'mean: {mean}, std: {std}')
            self.pos = self.pos + torch.normal(mean=mean.expand(N, 3), std=std.expand(N, 3))
            
        self.z = source_dataset[ids].z.clone().to(device)
        self.y = source_dataset[ids].y.clone().to(device)
        self.force = source_dataset[ids].force.clone().to(device)
        self.num_atom_per_molecule = self.pos.shape[0] // self.y.shape[0]
        
        self.num_cluster = int(self.source_size * cluster_rate)
        with torch.no_grad():
            _, self.cz = self.molecular_clustering(self.num_cluster)
            self.cz = torch.from_numpy(self.cz).long().to(device)
            
        self.distill_lr = torch.tensor(distill_lr, device=device, requires_grad=True)
        self.pos.requires_grad_(pos_requires_grad)
        self.y.requires_grad_(energy_requires_grad)
        self.force.requires_grad_(force_requires_grad)
        self.batch_data = None
        
    def to_torch(self):
        return TorchDistillDataset(
            z=self.z.detach().cpu().clone().requires_grad_(False), 
            cz=self.cz.detach().cpu().clone().requires_grad_(False), 
            pos=self.pos.detach().cpu().clone().requires_grad_(False), 
            y=self.y.detach().cpu().clone().requires_grad_(False), 
            force=self.force.detach().cpu().clone().requires_grad_(False), 
            num_atom_per_molecule=self.num_atom_per_molecule
            )
    
    def __len__(self):
        return self.size
    
    def get_batch(self, idx):
        batch_size = idx.shape[0]
        begin = idx[0] * self.num_atom_per_molecule
        end = (idx[-1] + 1) * self.num_atom_per_molecule
        self.batch_data = DistillData(z=self.z[begin:end], pos=self.pos[begin:end], y=self.y[idx], force=self.force[begin:end], cz=self.cz[begin:end], batch=torch.arange(0, batch_size).repeat_interleave(self.num_atom_per_molecule).long().to(self.force.device))
        return self.batch_data
    
    def get_lr(self, detach: bool=False):
        if detach:
            distill_lr = self.distill_lr.detach().cpu().clone()
            distill_lr.requires_grad_(False)
            return distill_lr
        
        return self.distill_lr
    
    def get_z(self):
        return self.z
    
    def get_y(self):
        return self.y
    
    def get_pos(self):
        return self.pos
    
    def get_force(self):
        return self.force
    
    def update_batch(self, idx, force=None, energy=None, pos=None):
        update_info = {}
        begin = idx[0] * self.num_atom_per_molecule
        end = (idx[-1] + 1) * self.num_atom_per_molecule
        if force is not None:
            force_diff = torch.abs(force - self.force[begin:end]).detach().cpu()
            update_info['force_update'] = {}
            update_info['force_update']['max'] = torch.max(force_diff).item()
            update_info['force_update']['mean'] = torch.mean(force_diff).item()
            update_info['force_update']['min'] = torch.min(force_diff).item()
            self.force[begin:end] = force
        
        if energy is not None:
            energy_diff = torch.abs(energy - self.y[idx]).detach().cpu()
            update_info['energy_update'] = {}
            update_info['energy_update']['max'] = torch.max(energy_diff).item()
            update_info['energy_update']['mean'] = torch.mean(energy_diff).item()
            update_info['energy_update']['min'] = torch.min(energy_diff).item()
            self.y[idx] = energy
            
        if pos is not None:
            pos_diff = torch.abs(pos - self.pos[begin:end]).detach().cpu()
            update_info['pos_update'] = {}
            update_info['pos_update']['max'] = torch.max(pos_diff).item()
            update_info['pos_update']['mean'] = torch.mean(pos_diff).item()
            update_info['pos_update']['min'] = torch.min(pos_diff).item()
            self.pos[begin:end] = pos
        
        return update_info
    
    def save(self, save_path):
        torch.save({
            'z': self.z.detach().cpu(),
            'pos': self.pos.detach().cpu(),
            'y': self.y.detach().cpu(),
            'force': self.force.detach().cpu(),
            'distill_lr': self.distill_lr.detach().cpu()
            }, save_path)
        print(f"Distill dataset saved to {save_path}")
        
    def molecular_clustering(self, num_clusters=1000):
        """
        Clustering for molecular
        """
        num_molecules = len(self.y)
        
        molecular_features = []
        for i in range(num_molecules):
            begin = self.num_atom_per_molecule * i
            end = self.num_atom_per_molecule * (i + 1)
            pos = self.pos[begin:end].reshape(-1)
            force = self.force[begin:end].reshape(-1)
            energy = self.y[i].repeat(pos.shape[0] // 3)
            molecular_vector = torch.cat([pos, force, energy], dim=0)
            molecular_features.append(molecular_vector)
        
        molecular_features = torch.stack(molecular_features).cpu().numpy()
        scaler = StandardScaler()
        molecular_features_scaled = scaler.fit_transform(molecular_features)
        
        kmeans_molecule = KMeans(n_clusters=num_clusters, random_state=42)
        molecule_cluster_ids = kmeans_molecule.fit_predict(molecular_features_scaled)   # [M]
        atom_cluster_ids = np.repeat(molecule_cluster_ids, repeats=self.num_atom_per_molecule)     # [M*A]
        return molecule_cluster_ids, atom_cluster_ids
    
    def get_num_cluster(self):
        return self.num_cluster
    
    def get_source_size(self):
        return self.source_size