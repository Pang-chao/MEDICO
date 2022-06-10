import os
import sys
# for linux env.
sys.path.insert(0,'..')
import torch
from modules.utils import check_validity, adj_to_smiles, check_novelty
from model import MoFlow
from config import get_gen_conf, Hyperparameters, get_config
from data.three_dim_process import spherenet
import time
import functools
from data.data_process import load_qm9_all
from utils.draw import *
from rdkit import Chem
from sklearn.utils import shuffle
print = functools.partial(print, flush=True)

def load_model(snapshot_path, model_params, data_name):
    print("loading snapshot: {}".format(snapshot_path))
    model = MoFlow(model_params, data_name)

    device = torch.device('cpu')
    model.load_state_dict(torch.load(snapshot_path, map_location=device))
    return model

def gaussian_noise(z_dim, temp=0.7, batch_size=20, device=-1):
    # xp = np
    if isinstance(device, torch.device):
        # xp = chainer.backends.cuda.cupy
        pass
    elif isinstance(device, int):
        if device >= 0:
            # device = args.gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', int(device))
        else:
            device = torch.device('cpu')
    else:
        raise ValueError("only 'torch.device' or 'int' are valid for 'device', but '%s' is "'given' % str(device))

    mu = np.zeros(z_dim)  # (369,)
    sigma_diag = np.ones(z_dim)  # (369,)
    sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag

    sigma = temp * sigma_diag

    with torch.no_grad():
        z = np.random.normal(mu, sigma, (batch_size, z_dim))  # .astype(np.float32)
        z = torch.from_numpy(z).float().to(device)

    return z

def check_valid_mol(x):
    s = Chem.MolFromSmiles(x)
    if s is not None and '.' not in x:
        return True
    else:
        return False

if __name__ == "__main__":
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    gen_config = get_gen_conf()
    snapshot_path = os.path.join(gen_config.model_dir, gen_config.snapshot_path)
    params_path = os.path.join(gen_config.model_dir, gen_config.params_path)
    model_params = Hyperparameters(path=params_path)
    batch_size = gen_config.batch_size
    seed = get_config().seed

    if gen_config.gpu >= 0:
        # device = args.gpu
        device = torch.device('cuda:'+str(get_gen_conf().gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = load_model(snapshot_path, model_params, gen_config.data_name)
    model.to(device)
    model.eval()

    if gen_config.data_name == 'qm9':
        train_loader, test_loader = load_qm9_all(gen_config.train_size, batch_size, seed)
        atom_max_num = 9
        atom_type = 5
        bond_type = 4
        atomic_num_list = [6, 7, 8, 9, 0]
        out_dim = 369

    sphere = spherenet(data_name=gen_config.data_name, out_channels=out_dim)
    sphere.load_state_dict(torch.load(os.path.join(gen_config.model_dir, gen_config.sphere_params)))
    sphere = sphere.to(device)
    sphere.eval()

    #draw molecules
    draw_mol = False
    if draw_mol:
        from rdkit.Chem import rdDepictor
        rdDepictor.SetPreferCoordGen(True)
        train_loader, test_loader = load_qm9_all(gen_config.train_size, 1, 666)
        draw_mol_list = []
        count = 0
        with open('draw_mol_sim.txt', 'w') as f:
            for i, batch_data in enumerate(test_loader):
                x = batch_data.atom.view(1, atom_max_num, atom_type)
                adj = batch_data.adj.view(1, bond_type, atom_max_num, atom_max_num)
                batch_data.to(device)
                with torch.no_grad():
                    u = sphere(batch_data)
                z_dim = atom_type * atom_max_num + bond_type * atom_max_num * atom_max_num
                gen_mols = []
                for i in range(5):
                    z = u * (1 - 0.2) + gaussian_noise(z_dim, batch_size=1,
                                                       device=gen_config.gpu) * 0.2
                    adj_rev, x_rev = model.reverse(z)
                    gen_mol = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
                    gen_mols.append(gen_mol[0])
                gen_mols = list(set(gen_mols))
                origin_mol = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
                gen_mols = [Chem.MolFromSmiles(s) for s in gen_mols]
                valid = [mol for mol in gen_mols if mol is not None]
                if len(valid) != 5:
                    continue
                origin_mol = Chem.MolFromSmiles(origin_mol[0])
                try:
                    sim = one2more_Fraggle(origin_mol, valid)
                    if min(sim) == 0:
                        continue
                except:
                    continue
                count += 1
                f.write(str(count) + ',')
                f.write(str(sim))
                f.write('\n')
                draw_mol_list.append(origin_mol)
                draw_mol_list.extend(valid)
                if count == 20:
                    draw_mols(draw_mol_list, 6, './figure')
                    break

    # 1. Gen with 3d information
    if get_gen_conf().gen_with_3d:
        dontsame_origin_smiles, dontsame_reverse_smiles = [], []
        max_iter = len(test_loader)
        for i, batch_data in enumerate(test_loader):
            if batch_data.atom.size()[0] != batch_size * atom_max_num:
                break
            x = batch_data.atom.view(batch_size, atom_max_num, atom_type)
            adj = batch_data.adj.view(batch_size, bond_type, atom_max_num, atom_max_num)
            batch_data.to(device)
            with torch.no_grad():
                u = sphere(batch_data)
            adj_rev, x_rev = model.reverse(u)
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            origin_smiles = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
            lb = np.array([int(a!=b) for a, b in zip(origin_smiles, reverse_smiles)])
            idx = np.where(lb)[0]
            if len(idx) > 0:
                for k in idx:
                    print(i*batch_size+k, 'train: ', origin_smiles[k], ' reverse: ', reverse_smiles[k])
                    if check_valid_mol(reverse_smiles[k]):
                        dontsame_origin_smiles.append(origin_smiles[k])
                        dontsame_reverse_smiles.append(reverse_smiles[k])

        if gen_config.calculate_similarity:
            dontsame_reverse_mols, dontsame_origin_mols = smile_list_to_mol_list(dontsame_reverse_smiles), smile_list_to_mol_list(dontsame_origin_smiles)
            Tanimoto_mean, Tanimoto_std, Tanimoto_sim = Tanimoto_similarity(dontsame_reverse_mols,
                                                                                   dontsame_origin_mols)
            Fraggle_mean, Fraggle_std, Fraggle_sim = Fraggle_similarity(dontsame_reverse_mols,
                                                                               dontsame_origin_mols)
            MACCS_mean, MACCS_std, MACCS_sim = MACCS_similarity(dontsame_reverse_mols, dontsame_origin_mols)
            print(f"Tanimoto_mean, Tanimoto_std, len:{Tanimoto_mean}, {Tanimoto_std}, {len(Tanimoto_sim)}")
            print(f"Fraggle_mean, Fraggle_std, len:{Fraggle_mean}, {Fraggle_std}, {len(Fraggle_sim)}")
            print(f"MACCS_mean, MACCS_std, len:{MACCS_mean}, {MACCS_std}, {len(MACCS_sim)}")
            np.savez('reconstuction_sim.npz', Tani=Tanimoto_sim, Frag=Fraggle_sim, MACCS=MACCS_sim)

            smiles_list = []
            if gen_config.data_name == 'qm9':
                with open('./data/qm9.csv') as f:
                    for i in range(133885):
                        line = f.readline()
                        words = line.split(',')
                        smiles_list.append(words[1])
                smiles_list = smiles_list[1:]
                mol_list = smile_list_to_mol_list(smiles_list)
                idx = shuffle(range(133885), random_state=2022)
                # dl, label = process_data(mol_list)
                # print(len(dl), label)
                # print(dl)
                # t_SNE(mol_list)
                idx = idx[:13000]
                a_mol_list, b_mol_list = [mol_list[i] for i in idx[:6500]], [mol_list[i] for i in idx[6500:]]

            mean, std, similarity = Tanimoto_similarity(a_mol_list, b_mol_list)
            print("Dataset Tanimoto_mean, Tanimoto_std:", mean, std)
            mean1, std1, similarity1 = Fraggle_similarity(a_mol_list, b_mol_list)
            print("Dataset Fraggle_mean, Fraggle_std:", mean1, std1)
            mean2, std2, similarity2 = MACCS_similarity(a_mol_list, b_mol_list)
            print("Dataset MACCS_mean, MACCS_std:", mean2, std2)
            density_map(similarity, Tanimoto_sim, '#fa4659', '#2eb872', 'Tanimoto')
            density_map(similarity1, Fraggle_sim, '#fa4659', '#2eb872', 'Fraggle')
            density_map(similarity2, MACCS_sim, '#fa4659', '#2eb872', 'MACCS')

    # 2. Random generation
    if gen_config.random_gen:
        train_smiles = []
        gen_smiles = []
        train_dock_score = []
        for i, batch_data in enumerate(test_loader):
            if batch_data.atom.size()[0] != batch_size * atom_max_num:
                break
            x = batch_data.atom.view(batch_size, atom_max_num, atom_type)
            adj = batch_data.adj.view(batch_size, bond_type, atom_max_num, atom_max_num)
            if gen_config.gen_calculate_dock_score:
                train_dock_score.extend(batch_data.dock)
            train_smile = adj_to_smiles(adj, x, atomic_num_list)
            train_smiles.extend(train_smile)
        print('Load trained model and data done! Time {:.2f} seconds'.format(time.time() - start))

        valid_ratio = []
        unique_ratio = []
        novel_ratio = []
        for i, batch_data in enumerate(test_loader):
            batch_data.to(device)
            if batch_data.atom.size()[0] != batch_size * atom_max_num:
                break
            with torch.no_grad():
                u = sphere(batch_data)
            z_dim = atom_type * atom_max_num + bond_type * atom_max_num * atom_max_num
            z = u * (1-gen_config.noise_scale) + gaussian_noise(z_dim, batch_size=gen_config.batch_size, device=gen_config.gpu) * gen_config.noise_scale
            adj, x = model.reverse(z)
            val_res = check_validity(adj, x, atomic_num_list, correct_validity=gen_config.correct_validity)
            novel_r, abs_novel_r = check_novelty(val_res['valid_smiles'], train_smiles, x.shape[0])
            novel_ratio.append(novel_r)

            unique_ratio.append(val_res['unique_ratio'])
            valid_ratio.append(val_res['valid_ratio'])
            n_valid = len(val_res['valid_mols'])
            gen_smiles.extend(val_res['valid_smiles'])

            # saves a png image of all generated molecules
            if gen_config.save_mol_fig:
                gen_dir = os.path.join(gen_config.model_dir, 'generated')
                os.makedirs(gen_dir, exist_ok=True)
                filepath = os.path.join(gen_dir, 'generated_mols_{}.png'.format(i))
                img = Draw.MolsToGridImage(val_res['valid_mols'], legends=val_res['valid_smiles'],
                                           molsPerRow=20, subImgSize=(300, 300))  # , useSVG=True
                img.save(filepath)

        print("validity: mean={:.2f}%, sd={:.2f}%".format(np.mean(valid_ratio), np.std(valid_ratio)))
        print("novelty: mean={:.2f}%, sd={:.2f}%".format(np.mean(novel_ratio), np.std(novel_ratio)))
        print("uniqueness: mean={:.2f}%, sd={:.2f}%".format(np.mean(unique_ratio), np.std(unique_ratio)))
        print('Task random generation done! Time {:.2f} seconds, Data: {}'.format(time.time() - start, time.ctime()))

        if gen_config.draw_tSNE:
            gen_mols, train_mols = smile_list_to_mol_list(gen_smiles), smile_list_to_mol_list(train_smiles)
            len_mols = 1000  # min(len(gen_mols), len(train_mols))
            gen_mols, train_mols = gen_mols[:len_mols], train_mols[:len_mols]

            t_SNE(train_mols, gen_mols)

        if gen_config.draw_mols:
            gen_mols = smile_list_to_mol_list(gen_smiles)
            idx = shuffle(range(len(gen_mols)), random_state=2022)
            for i in range(20):
                mols = [gen_mols[o] for o in idx[i*81:(i+1)*81]]
                draw_mols(mols, 9, f'./mol_figure{i}')

        # Calculating the docking score
        if gen_config.gen_calculate_dock_score:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Chem.rdmolfiles import MolToPDBFile

            origin_dock_energies, gen_dock_energies = [], []
            index = 0
            os.chdir('./gen_dock')
            gen_mols, train_mols = smile_list_to_mol_list(gen_smiles), smile_list_to_mol_list(train_smiles)
            len_mols = 1000  # min(len(gen_mols), len(train_mols))
            gen_mols, train_mols = gen_mols[:len_mols], train_mols[:len_mols]
            for i in range(len(gen_mols)):
                if i == 1000:
                    break
                try:
                    m1 = Chem.AddHs(train_mols[i])
                    m2 = Chem.AddHs(gen_mols[i])
                    AllChem.EmbedMultipleConfs(m1, numConfs=10)
                    AllChem.EmbedMultipleConfs(m2, numConfs=10)
                    res1 = AllChem.MMFFOptimizeMoleculeConfs(m1)
                    res2 = AllChem.MMFFOptimizeMoleculeConfs(m2)
                    conf1 = m1.GetConformer(0)
                    conf2 = m2.GetConformer(0)
                    MolToPDBFile(m1, 'origin_mol.pdb')
                    MolToPDBFile(m2, 'gen_mol.pdb')
                    command_file_prepare1 = 'timeout 10 python2 dock//MGLTools/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/prepare_ligand4.py -l origin_mol.pdb -o origin_mol.pdbqt -A bonds_hydrogens'
                    command_file_prepare2 = 'timeout 10 python2 dock/MGLTools/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/prepare_ligand4.py -l gen_mol.pdb -o gen_mol.pdbqt -A bonds_hydrogens'
                    command_dock1 = './vina --config conf_origin.txt'
                    command_dock2 = './vina --config conf_gen.txt'
                    ret_pre1 = os.system(command_file_prepare1)
                    ret_dock1 = os.system(command_dock1)
                    ret_pre2 = os.system(command_file_prepare2)
                    ret_dock2 = os.system(command_dock2)
                    index += 1
                    energy_ori, energy_gen = 0, 0
                    if ret_dock1 == 0 and ret_dock2 == 0:
                        print(f'In the process of docking {index}')
                        with open('out_origin.pdbqt') as f:
                            f.readline()
                            line = f.readline()
                            words = line.split(' ')
                            words = [w for w in words if len(w) > 1]
                            energy_ori = words[3]
                            origin_dock_energies.append(-float(energy_ori))
                        with open('out_gen.pdbqt') as f:
                            f.readline()
                            line = f.readline()
                            words = line.split(' ')
                            words = [w for w in words if len(w) > 1]
                            energy_gen = words[3]
                            gen_dock_energies.append(-float(energy_gen))
                        if (-float(energy_gen)) - (-float(energy_ori)) > 1.3:
                            print("ok!")
                            print(train_mols[i], gen_mols[i])
                            ret11 = os.system('cp origin_mol.pdbqt high_ori.pdbqt')
                            ret22 = os.system('cp gen_mol.pdbqt high_gen.pdbqt')
                            ret11 = os.system('cp origin_mol.pdb high_ori.pdb')
                            ret22 = os.system('cp gen_mol.pdb high_gen.pdb')
                            ret11 = os.system('cp out_origin.pdbqt high_out_ori.pdbqt')
                            ret22 = os.system('cp out_gen.pdbqt high_out_gen.pdbqt')
                            with open('high_smiles.txt', 'w') as f:
                                f.write(train_smiles[i])
                                f.write('\n')
                                f.write(gen_smiles[i])
                            sys.exit()
                    else:
                        continue
                    print('now docking:', i)
                except:
                    continue
            print('len_train_dock, len_gen_dock: ', len(origin_dock_energies), len(gen_dock_energies))
            np.savez('random_gen_dock.npz', origin=origin_dock_energies, gendock=gen_dock_energies)
            print(
                f'trainset_dock_mean: {np.mean(origin_dock_energies)}, trainset_dock_std: {np.std(origin_dock_energies)},'
                f' gen_dock_mean: {np.mean(gen_dock_energies)}, gen_dock_std: {np.std(gen_dock_energies)}')
            density_map(origin_dock_energies, gen_dock_energies, '#1e56a0', '#7f4a88', 'Docking scores')
            print('Docking finished!')

        if gen_config.gen_calculate_similarity:
            gen_mols, train_mols = smile_list_to_mol_list(gen_smiles), smile_list_to_mol_list(train_smiles)
            len_mols = min(len(gen_mols), len(train_mols))
            gen_mols, train_mols = gen_mols[:len_mols], train_mols[:len_mols]
            Tanimoto_mean, Tanimoto_std, Tanimoto_sim = Tanimoto_similarity(gen_mols,
                                                                                   train_mols)
            Fraggle_mean, Fraggle_std, Fraggle_sim = Fraggle_similarity(gen_mols,
                                                                               train_mols)
            MACCS_mean, MACCS_std, MACCS_sim = MACCS_similarity(gen_mols, train_mols)
            print(f"Tanimoto_mean, Tanimoto_std, len:{Tanimoto_mean}, {Tanimoto_std}, {len(Tanimoto_sim)}") # 0.1981, 0.1629  0.1052, 0.0721
            print(f"Fraggle_mean, Fraggle_std, len:{Fraggle_mean}, {Fraggle_std}, {len(Fraggle_sim)}") # 0.5195, 0.3273  0.2803, 0.2389
            print(f"MACCS_mean, MACCS_std, len:{MACCS_mean}, {MACCS_std}, {len(MACCS_sim)}") # 0.5118, 0.2155  0.3601, 0.1539
            np.savez('random_gen_sim.npz', Tani=Tanimoto_sim, Frag=Fraggle_sim, MACCS=MACCS_sim)

            smiles_list = []
            if gen_config.data_name == 'qm9':
                with open('./data/qm9.csv') as f:
                    for i in range(133885):
                        line = f.readline()
                        words = line.split(',')
                        smiles_list.append(words[1])
                smiles_list = smiles_list[1:]
                mol_list = smile_list_to_mol_list(smiles_list)
                idx = shuffle(range(133885), random_state=2022)
                # dl, label = process_data(mol_list)
                # print(len(dl), label)
                # print(dl)
                # t_SNE(mol_list)
                idx = idx[:13000]
                a_mol_list, b_mol_list = [mol_list[i] for i in idx[:6500]], [mol_list[i] for i in idx[6500:]]

            mean, std, similarity = Tanimoto_similarity(a_mol_list, b_mol_list)
            print("Dataset Tanimoto_mean, Tanimoto_std:", mean, std)
            mean1, std1, similarity1 = Fraggle_similarity(a_mol_list, b_mol_list)
            print("Dataset Fraggle_mean, Fraggle_std:", mean1, std1)
            mean2, std2, similarity2 = MACCS_similarity(a_mol_list, b_mol_list)
            print("Dataset MACCS_mean, MACCS_std:", mean2, std2)
            density_map(similarity, Tanimoto_sim, '#fa4659', '#2eb872', 'Tanimoto')
            density_map(similarity1, Fraggle_sim, '#fa4659', '#2eb872', 'Fraggle')
            density_map(similarity2, MACCS_sim, '#fa4659', '#2eb872', 'MACCS')