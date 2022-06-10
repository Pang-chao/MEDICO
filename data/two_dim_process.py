import os
import sys
sys.path.insert(0,'..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset
from logging import getLogger
import traceback

class NumpyTupleDataset(Dataset):
    """Dataset of a tuple of datasets.

        It combines multiple datasets into one dataset. Each example is represented
        by a tuple whose ``i``-th item corresponds to the i-th dataset.
        And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

        Args:
            datasets: Underlying datasets. The ``i``-th one is used for the
                ``i``-th item of each example. All datasets must have the same
                length.

        """
    def __init__(self, datasets, transform=None):

        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])  # 133885
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        # Initialization
        self._datasets = datasets
        self._length = length
        # self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)
        # self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                    for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)

        if self.transform:
            batches = self.transform(batches)
        return batches

    def get_datasets(self):
        return self._datasets


    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        np.savez(filepath, *numpy_tuple_dataset._datasets)
        print('Save {} done.'.format(filepath))

    @classmethod
    def load(cls, filepath, transform=None):
        print('Loading file {}'.format(filepath))
        if not os.path.exists(filepath):
            raise ValueError('Invalid filepath {} for dataset'.format(filepath))
            # return None
        load_data = np.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return cls(result, transform)

class DataFrameParser(object):
    """data frame parser

    This FileParser parses pandas dataframe.
    It should contain column which contain SMILES as input, and
    label column which is the target to predict.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        labels (str or list or None): labels column
        smiles_col (str): smiles column
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col='smiles',
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(DataFrameParser, self).__init__()
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels  # type: list
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)
        self.preprocessor = preprocessor

    def parse(self, df, return_smiles=False, target_index=None,
              return_is_successful=False):
        """parse DataFrame using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.
            return_smiles (bool): If set to `True`, smiles list is returned in
                the key 'smiles', it is a list of SMILES from which input
                features are successfully made.
                If set to `False`, `None` is returned in the key 'smiles'.
            target_index (list or None): target index list to partially extract
                dataset. If None (default), all examples are parsed.
            return_is_successful (bool): If set to `True`, boolean list is
                returned in the key 'is_successful'. It represents
                preprocessing has succeeded or not for each SMILES.
                If set to False, `None` is returned in the key 'is_success'.

        Returns (dict): dictionary that contains Dataset, 1-d numpy array with
            dtype=object(string) which is a vector of smiles for each example
            or None.

        """
        logger = self.logger
        pp = self.preprocessor
        smiles_list = []
        is_successful_list = []

        # counter = 0
        if isinstance(pp, GGNNPreprocessor):
            if target_index is not None:
                df = df.iloc[target_index]

            features = None
            smiles_index = df.columns.get_loc(self.smiles_col)
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles = row[smiles_index]
                # TODO(Nakago): Check.
                # currently it assumes list
                labels = [row[i] for i in labels_index]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        fail_count += 1
                        if return_is_successful:
                            is_successful_list.append(False)
                        continue
                    # Note that smiles expression is not unique.
                    # we obtain canonical smiles
                    canonical_smiles, mol = pp.prepare_smiles_and_mol(mol)
                    input_features = pp.get_input_features(mol)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels = self.postprocess_label(labels)

                    if return_smiles:
                        smiles_list.append(canonical_smiles)
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'
                                   .format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                # Initialize features: list of list
                if features is None:
                    if isinstance(input_features, tuple):
                        num_features = len(input_features)
                    else:
                        num_features = 1
                    if self.labels is not None:
                        num_features += 1
                    features = [[] for _ in range(num_features)]

                if isinstance(input_features, tuple):
                    for i in range(len(input_features)):
                        features[i].append(input_features[i])
                else:
                    features[0].append(input_features)
                if self.labels is not None:
                    features[len(features) - 1].append(labels)
                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)
            ret = []

            for feature in features:
                try:
                    feat_array = np.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = np.empty(len(feature), dtype=np.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        smileses = np.array(smiles_list) if return_smiles else None
        if return_is_successful:
            is_successful = np.array(is_successful_list)
        else:
            is_successful = None

        # if isinstance(result, tuple):
        #     if self.postprocess_fn is not None:
        #         result = self.postprocess_fn(*result)
        #     dataset = NumpyTupleDataset(*result)
        # else:
        #     if self.postprocess_fn is not None:
        #         result = self.postprocess_fn(result)
        #     dataset = NumpyTupleDataset(result)

        if isinstance(result, (tuple, list)):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            dataset = NumpyTupleDataset(result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            dataset = NumpyTupleDataset([result])

        return {"dataset": dataset,
                "smiles": smileses,
                "is_successful": is_successful}

    def extract_total_num(self, df):
        """Extracts total number of data which can be parsed

        We can use this method to determine the value fed to `target_index`
        option of `parse` method. For example, if we want to extract input
        feature from 10% of whole dataset, we need to know how many samples
        are in a file. The returned value of this method may not to be same as
        the final dataset size.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.

        Returns (int): total number of dataset can be parsed.

        """
        return len(df)

class GGNNPreprocessor(object):
    """GGNN Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(GGNNPreprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize

        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array

    def prepare_smiles_and_mol(self, mol):
        """Prepare `smiles` and `mol` used in following preprocessing.

        This method is called before `get_input_features` is called, by parser
        class.
        This method may be overriden to support custom `smile`/`mol` extraction

        Args:
            mol (mol): mol instance

        Returns (tuple): (`smiles`, `mol`)
        """
        # Note that smiles expression is not unique.
        # we obtain canonical smiles which is unique in `mol`
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol

    def get_label(self, mol, label_names=None):
        """Extracts label information from a molecule.

        This method extracts properties whose keys are
        specified by ``label_names`` from a molecule ``mol``
        and returns these values as a list.
        The order of the values is same as that of ``label_names``.
        If the molecule does not have a
        property with some label, this function fills the corresponding
        index of the returned list with ``None``.

        Args:
            mol (rdkit.Chem.Mol): molecule whose features to be extracted
            label_names (None or iterable): list of label names.

        Returns:
            list of str: label information. Its length is equal to
            that of ``label_names``. If ``label_names`` is ``None``,
            this function returns an empty list.

        """
        if label_names is None:
            return []

        label_list = []
        for label_name in label_names:
            if mol.HasProp(label_name):
                label_list.append(mol.GetProp(label_name))
            else:
                label_list.append(None)
        return label_list


class MolFeatureExtractionError(Exception):
    pass


# --- Type check ---
def type_check_num_atoms(mol, num_max_atoms=-1):
    """Check number of atoms in `mol` does not exceed `num_max_atoms`

    If number of atoms in `mol` exceeds the number `num_max_atoms`, it will
    raise `MolFeatureExtractionError` exception.

    Args:
        mol (Mol):
        num_max_atoms (int): If negative value is set, not check number of
            atoms.

    """
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        # Skip extracting feature. ignore this case.
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))


# --- Atom preprocessing ---
def construct_atomic_number_array(mol, out_size=-1):
    """Returns atomic numbers of atoms consisting a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of returned array.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the tail of
            the array is padded with zeros.

    Returns:
        numpy.ndarray: an array consisting of atomic numbers
            of atoms in the molecule.
    """

    atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_atom = len(atom_list)

    if out_size < 0:
        return np.array(atom_list, dtype=np.int32)
    elif out_size >= n_atom:
        # 'empty' padding for atom_list
        # 0 represents empty place for atom
        atom_array = np.zeros(out_size, dtype=np.int32)
        atom_array[:n_atom] = np.array(atom_list, dtype=np.int32)
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atom))


# --- Adjacency matrix preprocessing ---
def construct_adj_matrix(mol, out_size=-1, self_connection=True):
    """Returns the adjacent matrix of the given molecule.

    This function returns the adjacent matrix of the given molecule.
    Contrary to the specification of
    :func:`rdkit.Chem.rdmolops.GetAdjacencyMatrix`,
    The diagonal entries of the returned matrix are all-one.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.
        self_connection (bool): Add self connection or not.
            If True, diagonal element of adjacency matrix is filled with 1.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
            It is 2-dimensional array with shape (atoms1, atoms2), where
            atoms1 & atoms2 represent from and to of the edge respectively.
            If ``out_size`` is non-negative, the returned
            its size is equal to that value. Otherwise,
            it is equal to the number of atoms in the the molecule.
    """

    adj = rdmolops.GetAdjacencyMatrix(mol)
    s0, s1 = adj.shape
    if s0 != s1:
        raise ValueError('The adjacent matrix of the input molecule'
                         'has an invalid shape: ({}, {}). '
                         'It must be square.'.format(s0, s1))

    if self_connection:
        adj = adj + np.eye(s0)
    if out_size < 0:
        adj_array = adj.astype(np.float32)
    elif out_size >= s0:
        adj_array = np.zeros((out_size, out_size),
                                dtype=np.float32)
        adj_array[:s0, :s1] = adj
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of atoms in the input molecules (={}).'
            .format(out_size, s0))
    return adj_array


def construct_discrete_edge_matrix(mol, out_size=-1):
    """Returns the edge-type dependent adjacency matrix of the given molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
            It is 3-dimensional array with shape (edge_type, atoms1, atoms2),
            where edge_type represents the bond type,
            atoms1 & atoms2 represent from and to of the edge respectively.
            If ``out_size`` is non-negative, its size is equal to that value.
            Otherwise, it is equal to the number of atoms in the the molecule.
    """
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))
    adjs = np.zeros((4, size, size), dtype=np.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0
    return adjs


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='qm9',
                        choices=['qm9'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['relgcn'],)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    start_time = time.time()
    args = parse()
    data_name = args.data_name
    data_type = args.data_type
    print('args', vars(args))

    if data_name == 'qm9':
        max_atoms = 9
    else:
        raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


    if data_type == 'relgcn':
        preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
    else:
        raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

    data_dir = "."
    os.makedirs(data_dir, exist_ok=True)

    if data_name == 'qm9':
        print('Preprocessing qm9 data:')
        df_qm9 = pd.read_csv('qm9.csv', index_col=0)
        labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                  'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES1')
        result = parser.parse(df_qm9, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    else:
        raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

    NumpyTupleDataset.save(os.path.join(data_dir, '{}_{}.npz'.format(data_name, data_type)), dataset)
    print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )