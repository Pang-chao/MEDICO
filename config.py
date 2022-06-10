import json
import os
import numpy as np
from tabulate import tabulate
import torch

def get_config():
    return Config()

def get_hyperparams():
    return Hyperparameters()

def get_gen_conf():
    return GenerateConfig()

def get_sphere_config():
    return SphereConfig()

class Hyperparameters:
    def __init__(self,
                 # For bond
                 b_n_flow=10, b_n_block=1, b_n_squeeze=3, b_hidden_ch=[128, 128], b_affine=True, b_conv_lu=1,
                 # For atom
                 a_hidden_gnn=[64], a_hidden_lin=[128, 64], a_n_flow=27, a_n_block=1,
                 mask_row_size_list=[1], mask_row_stride_list=[1], a_affine=True,
                 # General
                 path=None, learn_dist=True, noise_scale=0.6):
        self.b_n_flow = b_n_flow  # Number of masked glow coupling layers in each block for bond tensor
        self.b_n_block = b_n_block
        self.b_n_squeeze = b_n_squeeze  # Number of squeeze, 3 for qm9
        self.b_hidden_ch = b_hidden_ch  # [128,128]
        self.b_affine = b_affine
        self.b_conv_lu = b_conv_lu  # True

        self.a_hidden_gnn = a_hidden_gnn  # [64]
        self.a_hidden_lin = a_hidden_lin  # [128, 64]
        self.a_n_flow = a_n_flow  # 27
        self.a_n_block = a_n_block  # 1
        self.mask_row_size_list = mask_row_size_list  # [9]
        self.mask_row_stride_list = mask_row_stride_list  # [True]
        self.a_affine = a_affine

        self.path = path  # None
        self.learn_dist = learn_dist  # Whether to learn the distribution of the feature matrix
        self.noise_scale = noise_scale

        # load function in the initialization by path argument
        if path is not None:
            if os.path.exists(path) and os.path.isfile(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        setattr(self, key, value)
            else:
                raise Exception("{} does not exist".format(path))

    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))

class Config():
    data_name = 'qm9'  # qm9
    data_dir = 'data'  # Dataset Storage Location
    save_dir = '/mnt/sdb/home/pc/codes/3dmoflow/results/qm9'  # Storage location for parameter checkpoints
    load_params = False  # Whether to start training from previous checkpoints

    learning_rate = 0.0001
    lr_decay = 0.999995  # Learning rate decay for each step
    max_epochs = 1000
    gpu = 3
    multi_gpu = False # Whether to use multi-gpu training
    save_epochs = 200  # How many rounds to store the model once

    batch_size = 512
    train_size = 120000
    seed = 2022

class SphereConfig():
    model_dir = './results/qm9/dock_1000/'
    snapshot_path = 'model_snapshot_epoch_5000'
    params_path = 'flow_params.json'
    data_name = 'qm9'
    gpu = 3
    batch_size = 256
    train_size = 120000
    max_epoch = 200
    learning_rate = 0.001
    lr_decay_factor = 0.95
    lr_decay_step_size = 50
    save_interval = 200

class GenerateConfig():
    model_dir = './results/qm9/dock_1000/'
    data_name = 'qm9'
    snapshot_path = 'model_snapshot_epoch_5000'
    params_path = 'flow_params.json'
    sphere_params = 'sphere_params_1000.pt'
    gpu = 3
    batch_size = 256
    train_size = 120000
    n_experiments = 1
    noise_scale = 0.2

    gen_with_3d = False
    calculate_similarity = False

    random_gen = False
    correct_validity = False
    gen_calculate_dock_score = False
    gen_calculate_similarity = False
    save_mol_fig = False
    draw_tSNE = False
    draw_mols = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    hyper = Hyperparameters()
    hyper.save('/mnt/sdb/home/pc/codes/3dmoflow/results/qm9/dock_1000/flow_params.json')
