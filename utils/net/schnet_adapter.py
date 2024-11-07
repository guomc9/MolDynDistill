import torch

class SchNetAdapter(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_interactions=3, num_clusters=100):
        super(SchNetAdapter, self).__init__()
        self.num_interactions = num_interactions
        self.num_layers = self.num_interactions + 2
        self.atom_embedder = torch.nn.Embedding(num_embeddings=100, embedding_dim=hidden_channels)
        self.cluster_embedder = torch.nn.Embedding(num_embeddings=num_clusters, embedding_dim=hidden_channels)
        self.layers = torch.nn.ModuleList()
        self.input_dim = 3 + hidden_channels * 2
        input_dim = self.input_dim
        for _ in range(self.num_layers-1):
            self.layers.append(torch.nn.Linear(input_dim, hidden_channels))
            input_dim = hidden_channels
        self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels // 2))
        
        self.activation = torch.nn.ReLU()
        self._init_zero_weights()

    def forward(self, batch_data):
        x, z, cz = batch_data.pos, batch_data.z, batch_data.cz
        x = torch.cat([x, self.atom_embedder(z), self.cluster_embedder(cz)], dim=-1)
        outputs = []
        for layer in self.layers:
            x = self.activation(layer(x))
            outputs.append(x)
        return outputs

    def _init_zero_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.zeros_(layer.weight)  # Set the weight matrix to all zeros
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)  # Set the bias vector to all zeros