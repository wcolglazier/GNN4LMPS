import torch
from torch.nn import Linear, ReLU, Sequential, Dropout, ModuleList
from torch_geometric.nn import GINEConv

class MPNN(torch.nn.Module):
    def __init__(self, in_node_feats=3, in_edge_feats=7, hidden_dim=64, num_layers=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = in_edge_feats
        self.num_layers = num_layers
        self.edge_embed = Sequential(
            Linear(in_edge_feats, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.node_embed = Linear(in_node_feats, hidden_dim)
        self.gine_layers = ModuleList([
            GINEConv(
                nn=Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                    ReLU()
                ),
                edge_dim=hidden_dim
            ) for _ in range(num_layers)
        ])
        self.edge_predictor = Sequential(
            Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            ReLU(),
            Dropout(0.1),
            Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for i in range(0, self.num_layers, 2):
            if i + 1 < self.num_layers:
                x1 = self.gine_layers[i](x, edge_index, edge_attr)
                x2 = self.gine_layers[i + 1](x1, edge_index, edge_attr)
                x = x1 + x2
            else:
                x = self.gine_layers[i](x, edge_index, edge_attr)
        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col], edge_attr], dim=1)
        pred = self.edge_predictor(edge_feat).squeeze()
        return pred
