from openbabel import pybel

mol = pybel.readstring("smi", "CCN(CC)CC")
mol.make3D()
output = pybel.Outputfile("pdb", "output.pdb")
output.write(mol)
output.close()