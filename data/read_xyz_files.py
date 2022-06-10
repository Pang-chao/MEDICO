import numpy as np

mol_num = 133885
files_name = 'dsgdb9nsd_000000'
atom_type_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
R = []
Z = []
N = []
for i in range(1, mol_num+1):
    len_i = len(str(i))
    file_name = files_name[:-len_i] + str(i) + '.xyz'
    try:
        with open('./qm9_xyz/' + file_name) as f:
            lines = f.readlines()
            atom_num = int(lines[0])
            for j in range(2, 2+atom_num):
                line = lines[j]
                words = line.split('\t')
                z = atom_type_dict[words[0]]
                xyz = words[1:-1]
                xyz_pos = np.array([float(c) if '^' not in c else float(c[:-4])*(10**(-int(c[-1]))) for c in xyz])
                R.append(xyz_pos)
                Z.append(z)
            N.append(atom_num)
    except:
        print(i)
np.savez('qm9_xyz.npz', R=R, Z=Z, N=N)
print("xyz files processing completed!")

