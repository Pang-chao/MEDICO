import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
import cairosvg

def density_map(data1, data2, color1, color2, title):
    sns.set(style="white")
    fig = sns.kdeplot(data1, shade=True, color=color1)
    fig = sns.kdeplot(data2, shade=True, color=color2)
    plt.title(title)
    plt.savefig(f'./{title}.jpg', dpi=750, bbox_inches = 'tight')
    plt.close()

def smile_list_to_mol_list(smile_list):
    mols = [Chem.MolFromSmiles(s) for s in smile_list]
    return mols

def Tanimoto_similarity(gen_mol_list, train_mol_list):
    similarity = []
    for i in range(len(gen_mol_list)):
        try:
            gen_fingerprint, train_fingerprint = Chem.RDKFingerprint(gen_mol_list[i]), Chem.RDKFingerprint(train_mol_list[i])
            similarity.append(DataStructs.TanimotoSimilarity(gen_fingerprint, train_fingerprint))
        except:
            continue
    mean = np.mean(similarity)
    std = np.std(similarity)
    return mean, std, similarity

def Fraggle_similarity(gen_mol_list, train_mol_list):
    similarity = []
    for i in range(len(gen_mol_list)):
        try:
            sim, match = FraggleSim.GetFraggleSimilarity(gen_mol_list[i], train_mol_list[i])
            similarity.append(sim)
        except:
            continue
    mean = np.mean(similarity)
    std = np.std(similarity)
    return mean, std, similarity

def MACCS_similarity(gen_mol_list, train_mol_list):
    similarity = []
    for i in range(len(gen_mol_list)):
        try:
            gen_MACCS, train_MACCS = MACCSkeys.GenMACCSKeys(gen_mol_list[i]), MACCSkeys.GenMACCSKeys(train_mol_list[i])
            similarity.append(DataStructs.DiceSimilarity(gen_MACCS, train_MACCS))
        except:
            continue
    mean = np.mean(similarity)
    std = np.std(similarity)
    return mean, std, similarity

def process_data(mol_list):
    fps = []
    label = []
    for mol in mol_list:
        label.append(Descriptors.MolWt(mol))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fps = np.array(fps)
    return fps, label

def plot_embedding_2D(data1, data2, label1, label2, title):
    x1 = [d[0] for d in data1]
    y1 = [d[1] for d in data1]
    x2 = [d[0] for d in data2]
    y2 = [d[1] for d in data2]
    x1_min, x1_max, x2_min, x2_max = min(x1), max(x1), min(x2), max(x2)
    y1_min, y1_max, y2_min, y2_max = min(y1), max(y1), min(y2), max(y2)
    x1_norm = [(d - x1_min) / (x1_max - x1_min) for d in x1]
    y1_norm = [(d - y1_min) / (y1_max - y1_min) for d in y1]
    x2_norm = [(d - x2_min) / (x2_max - x2_min) for d in x2]
    y2_norm = [(d - y2_min) / (y2_max - y2_min) for d in y2]
    np.savez('tSNE.npz', x1=x1_norm, y1=y1_norm, x2=x2_norm, y2=y2_norm, label1=label1, label2=label2)

    plt.figure(figsize=(10, 10), dpi=750)
    plt.scatter(x1_norm, y1_norm, s=1,c=label1, marker='o', cmap='RdYlGn', label='Dataset')
    plt.scatter(x2_norm, y2_norm, s=1, c=label2, marker='x', cmap='RdYlGn', label='Generated')
    plt.colorbar()
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig('t-SNE.png')

def t_SNE(dataset_mol_list, gen_mol_list):
    ds_data, ds_label = process_data(dataset_mol_list)
    gen_data, gen_label = process_data(gen_mol_list)
    print('Begining......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    ds_result_2D = tsne_2D.fit_transform(ds_data)
    gen_result_2D = tsne_2D.fit_transform(gen_data)
    print('Finished......')
    plot_embedding_2D(ds_result_2D,gen_result_2D, ds_label,gen_label, 't-SNE')

def plot_colormap():
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()
    # a = [.42, .41, .38, .32, .44, .37, .31, .53, .68, .57, .43,.32,1,.52, .57,.47,.39,.57,.68,
    #      .59,.79,.53,.71,.34,.54, .32, .2,.1]
    # a.sort()

    a = [[0.5063291139240507, 1.0, 0.6268656716417911, 0.5740740740740741, 0.5411764705882353]
            ,[0.7115384615384616, 0.5, 0.4642857142857143, 0.6724137931034483, 0.6]
            ,[0.4583333333333333, 1.0, 1.0, 0.2765957446808511, 0.40425531914893614]
            ,[1.0, 0.9333333333333333, 0.56, 0.49122807017543857, 1.0]
            ,[0.8235294117647058, 0.7, 0.8235294117647058, 1.0, 0.9285714285714286]
            ,[0.1188118811881188, 0.6818181818181818, 1.0, 1.0, 0.34615384615384615]]

    am = np.array(a)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = sns.heatmap(am, annot=True, fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size": 18}, vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.show()
    fig.savefig('a_all.pdf')

    idx = 0
    for idx in range(9):
        n = len(a[idx])
        aa = np.array(a[idx])
        aa = np.expand_dims(aa, axis=0)
        # aa = np.repeat(a, n, axis=0)
        fig, ax = plt.subplots(figsize=(14.4, .8))
        ax = sns.heatmap(aa, annot=True, fmt=".2f", linewidths=0.5, ax=ax, annot_kws={"size": 15}, cbar=False,
                         xticklabels=False, yticklabels=False, vmin=0, vmax=1)
        plt.show()
        fig.savefig('a{}.pdf'.format(idx))

def draw_mols(mols, mols_per_row, filepath):
    pic_size = (200, 200)
    svg = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row,
                               subImgSize=pic_size, useSVG=True)
    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=filepath + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=filepath + ".png")
    print('Draw mols finish!')

def one2more_Fraggle(train_mol, gen_mol_list):
    similarity = []
    for i in range(len(gen_mol_list)):
        sim, match = FraggleSim.GetFraggleSimilarity(train_mol, gen_mol_list[i])
        similarity.append(sim)
    return similarity

# rdkit.Chem.AllChem
# https://github.com/charnley/rmsd