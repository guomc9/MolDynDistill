import os
import os.path as osp
import torch.utils
import torch.utils.data
from torch_geometric.datasets import QM9
from .md17 import MD17
import torch
from torch.utils.data import Subset
import numpy as np

# def split_dataset(dataset, seed, train_ratio=0.1, valid_ratio=0.1, train_size=None, valid_size=None, **kwargs):
#     data_size = len(dataset)
#     if train_size is None:
#         assert train_ratio <= 1. and train_ratio > 0., f"train_ratio({train_ratio}) must be in (0, 1]."
#         train_size = int(data_size * train_ratio)
#     if valid_size is None:
#         assert valid_ratio <= 1. and valid_ratio > 0., f"valid_ratio({valid_ratio}) must be in (0, 1]."
#         valid_size = int(data_size * valid_ratio)
#     # assert train_ratio + valid_ratio <= 1., f"train_ratio + valid_ratio({train_ratio} + {valid_ratio}) must be not greater than 1."
    
#     ids = np.arange(data_size)
#     np.random.seed(seed)
#     np.random.shuffle(ids)

#     train_idx = ids[:train_size]
#     val_idx = ids[train_size:train_size + valid_size]
#     test_idx = ids[train_size + valid_size:]

#     train_dataset = Subset(dataset, train_idx)
#     valid_dataset = Subset(dataset, val_idx)
#     test_dataset = Subset(dataset, test_idx)
#     train_dataset, valid_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]
#     return train_dataset, valid_dataset, test_dataset

def split_dataset(dataset, seed, train_ratio=0.1, valid_ratio=0.1, train_size=None, valid_size=None, shuffle=True, **kwargs):
    """
    Split a dataset into train, validation, and test subsets.

    Args:
        dataset: The dataset to be split.
        seed: Random seed for reproducibility (used only when shuffle=True).
        train_ratio: Proportion of the dataset to be used for training.
        valid_ratio: Proportion of the dataset to be used for validation.
        train_size: Absolute size of the training set (overrides train_ratio if provided).
        valid_size: Absolute size of the validation set (overrides valid_ratio if provided).
        shuffle: Whether to shuffle the dataset before splitting.
        **kwargs: Additional arguments (unused).

    Returns:
        train_dataset, valid_dataset, test_dataset: The three subsets of the dataset.
    """
    data_size = len(dataset)
    
    # Calculate train and validation sizes if not provided
    if train_size is None:
        assert 0. < train_ratio <= 1., f"train_ratio({train_ratio}) must be in (0, 1]."
        train_size = int(data_size * train_ratio)
    if valid_size is None:
        assert 0. < valid_ratio <= 1., f"valid_ratio({valid_ratio}) must be in (0, 1]."
        valid_size = int(data_size * valid_ratio)
    
    # Ensure train + valid does not exceed dataset size
    assert train_size + valid_size <= data_size, (
        f"train_size + valid_size ({train_size} + {valid_size}) must not exceed dataset size ({data_size})."
    )
    
    # Generate indices
    ids = np.arange(data_size)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(ids)
    
    # Split indices
    train_idx = ids[:train_size]
    val_idx = ids[train_size:train_size + valid_size]
    test_idx = ids[train_size + valid_size:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
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
    def __init__(self, z, pos, y, force, num_atom_per_molecule):
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
                    force=force[begin:end]
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
        pos: torch.FloatTensor,
        y: torch.FloatTensor,
        force: torch.FloatTensor, 
        batch: torch.LongTensor
    ):
        self.z = z
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
class DistillDataset:
    def __init__(
        self,
        source_dataset, 
        distill_rate: float, 
        distill_lr: float, 
        device: str, 
        enable_cluster: bool = True, 
        num_cluster: int = 10, 
        pos_requires_grad=False, 
        energy_requires_grad=False, 
        force_requires_grad: bool = False
    ):
        self.source_dataset = source_dataset
        if source_dataset is not None:
            self.source_size = len(source_dataset)
            self.size = int(self.source_size * distill_rate)
            self.distill_lr = torch.tensor(distill_lr, device=device, requires_grad=True)
            self.enable_cluster = enable_cluster
            self.source_pos = []
            self.source_y = []
            self.source_force = []
            for data in self.source_dataset:
                self.source_pos.append(data.pos)
                self.source_y.append(data.y)
                self.source_force.append(data.force)
            
            self.source_pos = torch.cat(self.source_pos, dim=0)
            self.source_y = torch.cat(self.source_y, dim=0)
            self.source_force = torch.cat(self.source_force, dim=0)
            self.num_atom_per_molecule = self.source_dataset[0].pos.shape[0] // self.source_dataset[0].y.shape[0]
            if self.enable_cluster:
                self.num_cluster = num_cluster
                with torch.no_grad():
                    cluster_ids = self.molecular_clustering(self.num_cluster)
                    sampled_ids = self.sample_from_clusters(cluster_ids, self.size // self.num_cluster)
            else:
                sampled_ids = torch.randperm(self.source_size)[:self.size]
            print(sampled_ids)
            print(len(sampled_ids))
            self.pos = source_dataset[sampled_ids].pos.clone().to(device)
            self.z = source_dataset[sampled_ids].z.clone().to(device)
            self.y = source_dataset[sampled_ids].y.clone().to(device)
            self.force = source_dataset[sampled_ids].force.clone().to(device)

            self.pos.requires_grad_(pos_requires_grad)
            self.y.requires_grad_(energy_requires_grad)
            self.force.requires_grad_(force_requires_grad)
            self.batch_data = None

    def to_torch(self):
        return TorchDistillDataset(
            z=self.z.detach().cpu().clone().requires_grad_(False), 
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
        self.batch_data = DistillData(
            z=self.z[begin:end], 
            pos=self.pos[begin:end], 
            y=self.y[idx], 
            force=self.force[begin:end], 
            batch=torch.arange(0, batch_size).repeat_interleave(self.num_atom_per_molecule).long().to(self.force.device)
        )
        return self.batch_data

    def get_lr(self, detach: bool = False):
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
    
    def resample_batch(self, idx):
        update_info = {}
        begin = idx[0] * self.num_atom_per_molecule
        end = (idx[-1] + 1) * self.num_atom_per_molecule
        batch_size = len(idx)

        # Reshape `self.pos` for the batch: [B * num_atom_per_molecule, C] -> [B, num_atom_per_molecule * C]
        batch_pos = self.pos[begin:end].view(batch_size, -1)

        # Reshape `source_dataset.pos`: [num_molecule * num_atom_per_molecule, C] -> [num_molecule, num_atom_per_molecule * C]
        num_molecule = len(self.source_y)
        source_pos = self.source_pos.view(num_molecule, -1)

        # Find the closest molecule in `source_dataset` for each batch molecule
        distances = torch.cdist(batch_pos, source_pos)  # [B, num_molecule]
        closest_indices = torch.from_numpy(distances.argmin(axis=1))  # Closest indices in `source_dataset` [B]

        new_pos = []
        new_force = []
        new_energy = []

        for i, molecule_idx in enumerate(closest_indices):
            start = molecule_idx * self.num_atom_per_molecule
            stop = (molecule_idx + 1) * self.num_atom_per_molecule

            new_pos.append(self.source_pos[start:stop])
            new_force.append(self.source_force[start:stop])
            new_energy.append(self.source_y[molecule_idx])

        new_pos = torch.cat(new_pos).to(self.pos.device)
        new_force = torch.cat(new_force).to(self.force.device)
        new_energy = torch.stack(new_energy).to(self.y.device)

        pos_diff = torch.abs(new_pos - self.pos[begin:end]).detach().cpu()
        force_diff = torch.abs(new_force - self.force[begin:end]).detach().cpu()
        energy_diff = torch.abs(new_energy - self.y[idx]).detach().cpu()

        update_info['pos_update'] = {
            'max': torch.max(pos_diff).item(),
            'mean': torch.mean(pos_diff).item(),
            'min': torch.min(pos_diff).item(),
        }
        update_info['force_update'] = {
            'max': torch.max(force_diff).item(),
            'mean': torch.mean(force_diff).item(),
            'min': torch.min(force_diff).item(),
        }
        update_info['energy_update'] = {
            'max': torch.max(energy_diff).item(),
            'mean': torch.mean(energy_diff).item(),
            'min': torch.min(energy_diff).item(),
        }

        self.pos[begin:end] = new_pos
        self.force[begin:end] = new_force
        self.y[idx] = new_energy

        return update_info

    def update_batch(self, idx, force=None, energy=None, pos=None):
        update_info = {}
        begin = idx[0] * self.num_atom_per_molecule
        end = (idx[-1] + 1) * self.num_atom_per_molecule
        
        if force is not None:
            force_diff = torch.abs(force - self.force[begin:end]).detach().cpu()
            update_info['force_update'] = {
                'max': torch.max(force_diff).item(),
                'mean': torch.mean(force_diff).item(),
                'min': torch.min(force_diff).item(),
            }
            self.force[begin:end] = force
        
        if energy is not None:
            energy_diff = torch.abs(energy - self.y[idx]).detach().cpu()
            update_info['energy_update'] = {
                'max': torch.max(energy_diff).item(),
                'mean': torch.mean(energy_diff).item(),
                'min': torch.min(energy_diff).item(),
            }
            self.y[idx] = energy
            
        if pos is not None:
            pos_diff = torch.abs(pos - self.pos[begin:end]).detach().cpu()
            update_info['pos_update'] = {
                'max': torch.max(pos_diff).item(),
                'mean': torch.mean(pos_diff).item(),
                'min': torch.min(pos_diff).item(),
            }
            self.pos[begin:end] = pos
        
        return update_info
    
    def check_batch(self, idx, force=None, energy=None, pos=None):
        update_info = {}
        begin = idx[0] * self.num_atom_per_molecule
        end = (idx[-1] + 1) * self.num_atom_per_molecule
        
        if force is not None:
            force_diff = torch.abs(force - self.force[begin:end]).detach().cpu()
            update_info['force_update'] = {
                'max': torch.max(force_diff).item(),
                'mean': torch.mean(force_diff).item(),
                'min': torch.min(force_diff).item(),
            }
        
        if energy is not None:
            energy_diff = torch.abs(energy - self.y[idx]).detach().cpu()
            update_info['energy_update'] = {
                'max': torch.max(energy_diff).item(),
                'mean': torch.mean(energy_diff).item(),
                'min': torch.min(energy_diff).item(),
            }
            
        if pos is not None:
            pos_diff = torch.abs(pos - self.pos[begin:end]).detach().cpu()
            update_info['pos_update'] = {
                'max': torch.max(pos_diff).item(),
                'mean': torch.mean(pos_diff).item(),
                'min': torch.min(pos_diff).item(),
            }
        
        return update_info

    def molecular_clustering(self, num_clusters):
        """
        Perform clustering based only on the `pos` feature of molecules.
        """
        num_molecules = len(self.source_dataset)
        molecular_features = []

        for i in range(num_molecules):
            begin = self.num_atom_per_molecule * i
            end = self.num_atom_per_molecule * (i + 1)
            pos = self.source_pos[begin:end].reshape(-1)
            molecular_features.append(pos)

        molecular_features = torch.stack(molecular_features).cpu().numpy()
        scaler = StandardScaler()
        molecular_features_scaled = scaler.fit_transform(molecular_features)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(molecular_features_scaled)
        
        return cluster_ids

    def sample_from_clusters(self, cluster_ids, num_samples_per_cluster):
        """
        Uniformly sample one molecule from each cluster.
        """
        sampled_ids = []
        unique_clusters = set(cluster_ids)
        for cluster in unique_clusters:
            cluster_indices = torch.where(torch.tensor(cluster_ids) == cluster)[0]
            sampled_id = cluster_indices[torch.randint(len(cluster_indices), (num_samples_per_cluster,))]
            sampled_ids.append(sampled_id)
        return torch.concat(sampled_ids, dim=0)
    
    def save(self, save_path):
        torch.save({
            'z': self.z.detach().cpu(),
            'pos': self.pos.detach().cpu(),
            'y': self.y.detach().cpu(),
            'force': self.force.detach().cpu(),
            'distill_lr': self.distill_lr.detach().cpu()
            }, save_path)
        print(f"Distill dataset saved to {save_path}")
    
    @classmethod
    def load(cls, load_path, device, pos_requires_grad=False, energy_requires_grad=False, force_requires_grad=False, lr_requires_grad=False, cluster_rate=0.5):
        checkpoint = torch.load(load_path)
        dataset = cls(
            source_dataset=None,
            distill_rate=1.0,  
            distill_lr=checkpoint['distill_lr'], 
            device=device,
            pos_requires_grad=pos_requires_grad,
            energy_requires_grad=energy_requires_grad,
            force_requires_grad=force_requires_grad,
            noise_pos=False
        )
        
        dataset.z = checkpoint['z'].to(device)
        dataset.pos = checkpoint['pos'].to(device)
        dataset.y = checkpoint['y'].to(device)
        dataset.force = checkpoint['force'].to(device)
        dataset.distill_lr = checkpoint['distill_lr'].to(device)
        dataset.size = len(dataset.y)
        dataset.num_atom_per_molecule = dataset.pos.shape[0] // dataset.y.shape[0]
        
        dataset.pos.requires_grad_(pos_requires_grad)
        dataset.y.requires_grad_(energy_requires_grad)
        dataset.force.requires_grad_(force_requires_grad)
        dataset.distill_lr.requires_grad_(lr_requires_grad)
        dataset.num_atom_per_molecule = dataset.pos.shape[0] // dataset.y.shape[0]
        
        dataset.num_cluster = int(dataset.size * cluster_rate)
        with torch.no_grad():
            _, dataset.cz = dataset.molecular_clustering(dataset.num_cluster)
            dataset.cz = torch.from_numpy(dataset.cz).long().to(device)
        
        print(f"Distill dataset loaded from {load_path}")
        return dataset

from torch_geometric.data import Data
from torch.utils.data import Dataset
class PruneDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = self._convert_to_pyg_data(data_list)

    def _convert_to_pyg_data(self, data_list):
        pyg_data_list = []
        for data in data_list:
            pyg_data = Data(
                z=data.z if hasattr(data, 'z') else None, 
                pos=data.pos if hasattr(data, 'pos') else None, 
                y=data.y if hasattr(data, 'y') else None, 
                force=data.force if hasattr(data, 'force') else None, 
                energy=data.energy if hasattr(data, 'energy') else None
            )
            pyg_data_list.append(pyg_data)
        return pyg_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
