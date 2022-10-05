import os
import torch
import torch.nn as nn
from generate import rescale_adj
import utils.environment as env
from modules.utils import check_validity, adj_to_smiles
from rdkit import Chem
from data.data_process import load_qm9_prop
from config import OptimPropConfig, Hyperparameters
from data.three_dim_process import spherenet
from generate import load_model, gaussian_noise
from utils.draw import draw_mols

config = OptimPropConfig()
batch_size = config.batch_size
if config.gpu >= 0:
    device = torch.device('cuda:' + str(config.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

if config.data_name == 'qm9':
    atom_max_num = 9
    atom_type = 5
    bond_type = 4
    atomic_num_list = [6, 7, 8, 9, 0]
    out_dim = 369

class FlowProp(nn.Module):
    def __init__(self, model, hidden_size):
        super(FlowProp, self).__init__()
        self.model = model
        self.latent_size = model.b_size + model.a_size
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                modules.append(nn.Tanh())
        self.propNN = nn.Sequential(*modules)

    def encode(self, adj, x):
        with torch.no_grad():
            self.model.eval()
            adj_normalized = rescale_adj(adj).to(adj)
            z, sum_log_det_jacs = self.model(adj, x, adj_normalized)
            h = torch.cat([z[0].reshape(z[0].shape[0], -1), z[1].reshape(z[1].shape[0], -1)], dim=1)
        return h, sum_log_det_jacs

    def reverse(self, z):
        with torch.no_grad():
            self.model.eval()
            adj, x = self.model.reverse(z, true_adj=None)
        return adj, x

    def forward(self, adj, x):
        h, sum_log_det_jacs = self.encode(adj, x)
        output = self.propNN(h)
        return output, h,  sum_log_det_jacs

def fit_model(model, train_loader):
    if config.gpu >= 0:
        device = torch.device('cuda:' + str(config.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.train()
    metrics = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate, weight_decay=config.learning_rate)

    for epoch in range(config.max_epochs):
        print("In epoch {}".format(epoch + 1))
        for batch_data in train_loader:
            if batch_data.atom.size()[0] != batch_size * atom_max_num:
                break
            batch_data.to(device)
            x = batch_data.atom.view(batch_size, atom_max_num, atom_type)
            adj = batch_data.adj.view(batch_size, bond_type, atom_max_num, atom_max_num)
            if config.property_name == 'qed':
                true_y = batch_data.qed
            elif config.property_name == 'plogp':
                true_y = batch_data.plogp

            optimizer.zero_grad()

            y, z, sum_log_det_jacs = model(adj, x)

            loss = metrics(y, true_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{config.max_epochs}], loss: {loss.item()}.')

    print("fit_model Ends")
    return model

def optimize_mol(model, property_model, sphere, batch_data,lr, num_iter=20,
              random=False):
    if config.property_name == 'qed':
        propf = env.qed
    elif config.property_name == 'plogp':
        propf = env.penalized_logp
    else:
        raise ValueError(f"Wrong property_name{config.property_name}")

    if config.gpu >= 0:
        device = torch.device('cuda:' + str(config.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model.eval()
    property_model.eval()
    with torch.no_grad():
        u = sphere(batch_data)
    z_dim = atom_type * atom_max_num + bond_type * atom_max_num * atom_max_num
    z = u * (1 - 0.2) + gaussian_noise(z_dim, batch_size=1,
                                       device=config.gpu) * 0.2
    adj_rev, x_rev = model.reverse(z)
    mol_vec, sum_log_det_jacs = property_model.encode(adj_rev, x_rev)
    smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)

    mol = Chem.MolFromSmiles(smiles[0])
    start = (smiles[0], propf(mol), None) # , mol)

    cur_vec = mol_vec.clone().detach().requires_grad_(True).to(device)  # torch.tensor(mol_vec, requires_grad=True).to(mol_vec)
    start_vec = mol_vec.clone().detach().requires_grad_(True).to(device)

    visited = []
    for step in range(num_iter):
        prop_val = property_model.propNN(cur_vec).squeeze()
        grad = torch.autograd.grad(prop_val, cur_vec)[0]
        if random:
            rad = torch.randn_like(cur_vec.data)
            cur_vec = start_vec.data + lr * rad / torch.sqrt(rad * rad)
        else:
            cur_vec = cur_vec.data + lr * grad.data / torch.sqrt(grad.data * grad.data)
        cur_vec = cur_vec.clone().detach().requires_grad_(True).to(device)  # torch.tensor(cur_vec, requires_grad=True).to(mol_vec)
        visited.append(cur_vec)

    hidden_z = torch.cat(visited, dim=0).to(device)
    adj, x = property_model.reverse(hidden_z)
    val_res = check_validity(adj, x, atomic_num_list)
    valid_mols = val_res['valid_mols']
    valid_smiles = val_res['valid_smiles']
    results = []
    sm_set = set()
    sm_set.add(smiles[0])
    for m, s in zip(valid_mols, valid_smiles):
        if s in sm_set:
            continue
        sm_set.add(s)
        p = propf(m)
        results.append((s, p, smiles[0]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results, start

def find_top_score_smiles(model, property_model, data_loader, topk):
    print('Finding top {} score'.format(config.property_name))
    result_list = []
    data = []
    train_smiles = set()
    for i, batch_data in enumerate(data_loader):
        if config.property_name == 'qed':
            data.append([batch_data.qed, batch_data])
        elif config.property_name == 'plogp':
            data.append([batch_data.plogp, batch_data])
    data_sorted = sorted(data, key=lambda x: x[0], reverse=True)
    test_scores = open(f'{config.property_name}_test_scores.txt', 'w')
    for i, d in enumerate(data_sorted):
        if i >= topk:
            break
        test_scores.write(str(d[0]))
        test_scores.write('\n')
        if i % 50 == 0:
            print(f'Optimization {i}/{topk}')
        d[1].to(device)
        try:
            results, ori = optimize_mol(model, property_model, sphere, d[1], lr=.005, num_iter=config.steps,random=False)
            result_list.extend(results)
            d[1].cpu()
            x = d[1].atom.view(1, atom_max_num, atom_type)
            adj = d[1].adj.view(1, bond_type, atom_max_num, atom_max_num)
            train_smile = adj_to_smiles(adj, x, atomic_num_list)
            train_smiles.add(train_smile[0])
        except:
            continue
    test_scores.close()

    result_list.sort(key=lambda x: x[1], reverse=True)

    # check novelty
    result_list_novel = []
    for i, r in enumerate(result_list):
        smile, score, smile_original = r
        if smile not in train_smile:
            result_list_novel.append(r)

    # dump results
    f = open(config.property_name + '_discovered_sorted.csv', "w")
    f.write('score,smile,origin_smile\n')
    for r in result_list_novel:
        smile, score, smile_original = r
        f.write('{},{},{}\n'.format(score, smile, smile_original))
        f.flush()
    f.close()
    print('Dump done!')

if __name__ == '__main__':
    sphere = spherenet(data_name=config.data_name, out_channels=out_dim)
    sphere.load_state_dict(torch.load(os.path.join(config.model_dir, config.sphere_params)))
    sphere = sphere.to(device)
    sphere.eval()

    params_path = os.path.join(config.model_dir, config.params_path)
    snapshot_path = os.path.join(config.model_dir, config.snapshot_path)
    model_params = Hyperparameters(path=params_path)
    flow = load_model(snapshot_path, model_params, config.data_name)
    flow.to(device)
    if config.mode == 'train':
        train_loader, test_loader = load_qm9_prop(config.train_size, batch_size, config.seed)
        property_model_path = os.path.join(config.model_dir, '{}_optim_model.pt'.format(config.property_name))
        OptimModel = FlowProp(flow, config.hidden_size)
        OptimModel = fit_model(OptimModel, train_loader)
        torch.save(OptimModel, property_model_path)
        print('Train and save model done!')
    elif config.mode == 'gen':
        optim_model_path = os.path.join(config.model_dir, '{}_optim_model.pt'.format(config.property_name))
        OptimModel = torch.load(optim_model_path, map_location=device)
        print('Load model done!')

        OptimModel.to(device)
        OptimModel.eval()

        train_loader, test_loader = load_qm9_prop(config.train_size, 1, config.seed)
        if config.optim_mol:
            # draw_mol_list = []
            f = open(f'optim_mol_{config.property_name}.txt', 'w')
            for i, batch_data in enumerate(test_loader):
                batch_data.to(device)
                try:
                    results, start_mol = optimize_mol(flow, OptimModel, sphere, batch_data, lr=0.005, num_iter=config.steps)
                except:
                    continue
                #draw_mol_list.append(start_mol, res_mol)
                try:
                    f.write(str(results[0][1] - start_mol[1]) + ', ')
                    f.write(str(results) + ', ' + str(start_mol))
                    f.write('\n')
                except:
                    continue
                if i >= 200:
                    break
            f.close()
            #draw_mols(draw_mol_list, 2, './optim_mol')
        if config.top_score:
            find_top_score_smiles(flow, OptimModel, test_loader, config.topk)
