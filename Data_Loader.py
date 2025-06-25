import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np

npz = np.load("Path/to/Data.npz", allow_pickle=True)

edge_index_list = npz["edge_index"]
generation_list = npz["generation"]
load_list = npz["load"]
net_injection_list = npz["net_injection"]
reactance_list = npz["reactance"]
line_limit_list = npz["line_limit"]
power_flow_list = npz["power_flow"]
lmp_spread_edge_list = npz["lmp_spread_edge"]
distance_list = npz["distance"]
num_nodes_list = npz["num_nodes"]
num_edges_list = npz["num_edges"]

all_lmp_spreads = torch.cat([torch.tensor(arr[:num_edges_list[i]], dtype=torch.float32) for i, arr in enumerate(lmp_spread_edge_list)])
lmp_std = all_lmp_spreads.std()
normalized_lmp_list = [torch.tensor(arr[:num_edges_list[i]], dtype=torch.float32) / lmp_std for i, arr in enumerate(lmp_spread_edge_list)]

class Data_loader(Dataset):
    def __init__(self):
        super().__init__()

    def len(self):
        return len(num_nodes_list)

    def get(self, idx):
        num_nodes = num_nodes_list[idx]
        num_edges = num_edges_list[idx]

        x = torch.stack([
            torch.tensor(generation_list[idx][:num_nodes], dtype=torch.float32),
            torch.tensor(load_list[idx][:num_nodes], dtype=torch.float32),
            torch.tensor(net_injection_list[idx][:num_nodes], dtype=torch.float32),
        ], dim=1)

        edge_index = torch.tensor(edge_index_list[idx][:, :num_edges], dtype=torch.long)

        src = edge_index[0].float() / (num_nodes - 1)
        dst = edge_index[1].float() / (num_nodes - 1)
        edge_attr = torch.stack([
            torch.tensor(reactance_list[idx][:num_edges], dtype=torch.float32),
            torch.tensor(line_limit_list[idx][:num_edges], dtype=torch.float32),
            torch.tensor(power_flow_list[idx][:num_edges], dtype=torch.float32),
            torch.tensor(distance_list[idx][:num_edges], dtype=torch.float32),
            normalized_lmp_list[idx],
            src,
            dst
        ], dim=1)

        y = normalized_lmp_list[idx]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

dataset = Data_loader()
num_graphs = len(dataset)
train_dataset = dataset[:int(0.8 * num_graphs)]
val_dataset = dataset[int(0.8 * num_graphs):int(0.9 * num_graphs)]
test_dataset = dataset[int(0.9 * num_graphs):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

