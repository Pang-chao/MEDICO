import numpy as np
import torch
import random
from sklearn.utils import shuffle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def one_hot(data, out_size, num_max_id):
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id))
    # data = data[data > 0]
    # 6 is C: Carbon, we adopt 6:C, 7:N, 8:O, 9:F only. the last place (4) is for padding virtual node.
    indices = np.where(data >= 6, data - 6, num_max_id - 1)
    b[np.arange(out_size), indices] = 1
    # print('[DEBUG] data', data, 'b', b)
    return b

def pick_data(dataset, batch_size):
    res_dataset = []
    dock_sort = np.sort(list({d.dock for d in dataset}))
    min_dock, max_dock = dock_sort[1], dock_sort[-1]  # 除零以外的最小值
    for data in dataset:
        random_float = random.uniform(min_dock, max_dock)
        if data.dock >= random_float:
            res_dataset.append(data)
    train_loader = DataLoader(res_dataset, batch_size, shuffle=True)
    return train_loader

def load_qm9(train_size, seed):
    num_molecule = 133885
    dataset = []
    data_xyz = np.load('data/qm9_xyz.npz')
    data_dock = np.load('data/qm9_dock.npz')
    R = data_xyz['R']
    Z = data_xyz['Z']
    N = data_xyz['N']
    dock = data_dock['dock']
    split = np.cumsum(N)
    R_qm9 = np.split(R, split)
    Z_qm9 = np.split(Z, split)
    data_2d = np.load('data/qm9_relgcn.npz')
    node, adj = data_2d['arr_0'], data_2d['arr_1']

    for i in range(num_molecule):
        R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
        z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
        node_i = torch.tensor(one_hot(node[i], 9, 5).astype(np.float32), dtype=torch.float32)
        adj_i = torch.tensor(np.concatenate([adj[i][:3], 1 - np.sum(adj[i][:3], axis=0, keepdims=True)], axis=0).astype(np.float32), dtype=torch.float32)
        dock_i = dock[i]
        data = Data(pos=R_i, z=z_i, atom=node_i, adj=adj_i, dock=dock_i)
        dataset.append(data)
    idx = shuffle(range(num_molecule), random_state=seed)
    train_idx = np.array(idx[:train_size])

    train_dataset = [dataset[int(i)] for i in train_idx]

    return train_dataset

def load_qm9_all(train_size, batch_size, seed):
    num_molecule = 133885
    dataset = []
    data_xyz = np.load('data/qm9_xyz.npz')
    data_dock = np.load('data/qm9_dock.npz')
    R = data_xyz['R']
    Z = data_xyz['Z']
    N = data_xyz['N']
    dock = data_dock['dock']
    split = np.cumsum(N)
    R_qm9 = np.split(R, split)
    Z_qm9 = np.split(Z, split)
    data_2d = np.load('data/qm9_relgcn.npz')
    node, adj = data_2d['arr_0'], data_2d['arr_1']

    for i in range(num_molecule):
        R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
        z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
        node_i = torch.tensor(one_hot(node[i], 9, 5).astype(np.float32), dtype=torch.float32)
        adj_i = torch.tensor(np.concatenate([adj[i][:3], 1 - np.sum(adj[i][:3], axis=0, keepdims=True)], axis=0).astype(np.float32), dtype=torch.float32)
        dock_i = dock[i]
        data = Data(pos=R_i, z=z_i, atom=node_i, adj=adj_i, dock=dock_i)
        dataset.append(data)
    idx = shuffle(range(num_molecule), random_state=seed)

    train_idx, test_idx = np.array(idx[:train_size]), np.array(idx[train_size:])

    train_dataset = [dataset[int(i)] for i in train_idx]
    test_dataset = [dataset[int(i)] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader