import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle

def process_qm9_xyz(molecule_num = 130831, atom_num = 29, atom_type_num = 5, dim_num = 3):
    data = np.load('data/qm9_xyz.npz')
    R = data['R']
    Z = data['Z']
    N = data['N']
    split = np.cumsum(N)
    R_split = np.split(R, split)
    Z_split = np.split(Z, split)

    # 原子矩阵的one-hot编码 (batch, 29, 5)
    atom_tensor = torch.empty((molecule_num, atom_num, atom_type_num), dtype=torch.float32)
    for i in range(molecule_num):
        one_hot = torch.zeros((atom_num, atom_type_num), dtype=torch.float32)
        z_slice = np.zeros(atom_num, dtype=int)
        z_slice[:len(Z_split[i])] = Z_split[i]
        for j, zs in enumerate(z_slice):
            if zs == 1:
                one_hot[j][0] = 1
            elif zs > 1:
                one_hot[j][zs - 5] = 1
        atom_tensor[i] = one_hot

    # 3D xyz 坐标的编码（不是one-hot）(batch, 3, 29, 5)
    pos_xyz_tensor = torch.empty((molecule_num, dim_num, atom_num, atom_type_num), dtype=torch.float32)
    for i in range(molecule_num):
        r_pad = torch.zeros((dim_num, atom_num, atom_type_num), dtype=torch.float32)
        r_slice = R_split[i].T
        for j in range(len(r_slice[0])):
            if Z_split[i][j] == 1:
                r_pad[0][j][0] = r_slice[0][j]
                r_pad[1][j][0] = r_slice[1][j]
                r_pad[2][j][0] = r_slice[2][j]
            elif Z_split[i][j] > 1:
                r_pad[0][j][Z_split[i][j] - 5] = r_slice[0][j]
                r_pad[1][j][Z_split[i][j] - 5] = r_slice[1][j]
                r_pad[2][j][Z_split[i][j] - 5] = r_slice[2][j]
        pos_xyz_tensor[i] = r_pad

    return atom_tensor, pos_xyz_tensor

def load_qm9_xyz(atom_tensor, pos_tensor, batch_size, train_size, seed, shuffle1):
    mol_num = 130831
    dataset = []

    for i in range(mol_num):
        atom = atom_tensor[i]
        pos = pos_tensor[i]
        data = Data(atom=atom, pos=pos)  # edge_index
        dataset.append(data)

    ids = shuffle(range(mol_num), random_state=seed)
    train_idx, test_idx = np.array(ids[:train_size]), np.array(ids[train_size:])

    train_dataset = [dataset[int(i)] for i in train_idx]
    test_dataset = [dataset[int(i)] for i in test_idx]

    train_dataset = train_dataset[:(train_size//batch_size)*batch_size]
    test_dataset = test_dataset[:((mol_num-train_size)//batch_size)*batch_size]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle1)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader
