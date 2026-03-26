import os
import pickle
import lmdb
import torch
import numpy as np

from loguru import logger
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from functools import partial
from torchdrug import data, transforms  


AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
    'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 
    'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

categories_names = ['ec1', 'ec2', 'ec3', 'ec4', 'species_id','genus_id']
nums_names = ['T_K', 'inv_T', 'log_T', 
       'pH_centered', 'pH_squared', 'pH',
       'num_mutations', 'mut_ratio',
       'min_rel_pos', 'max_rel_pos', 'mean_rel_pos', 'std_rel_pos','frac_mut_Nterm', 'frac_mut_Cterm',
       'num_charge_change','any_charge_change', 'net_charge_change', 'abs_net_charge_change',
       'num_polarity_change', 
       'num_hydro_increase', 'num_hydro_decrease','net_hydropathy_change','abs_net_hydropathy_change',
       'num_aromatic_change', 'any_aromatic_change', 
       'any_to_proline','any_from_proline', 'any_to_gly', 'any_from_gly', 
       ]

allowable_features = {
    'possible_symbol_list': ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"] + ['misc'],
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW', 
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_total_valence_list': [0, 1, 2, 3, 4, 5, 6, 7, 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
}

def parse_mutation(mutation_str):
    """
    解析突变信息，返回 mut_idx, mut_from_aa_idx, mut_to_aa_idx
    """
    mut_idx = []
    mut_from_aa_idx = []
    mut_to_aa_idx = []
    
    if mutation_str == "wildtype" or mutation_str == "WT":
        return torch.tensor([[-1]]), torch.tensor([[-1]]), torch.tensor([[-1]])
    
    mutations = mutation_str.split('/')
    
    for mutation in mutations:
        mutation = mutation.strip()
        aa_from, pos_and_aa_to = mutation[0], mutation[1:]  
        pos = int(pos_and_aa_to[:-1]) - 1 
        aa_to = pos_and_aa_to[-1]          
        
        mut_idx.append(pos)
        mut_from_aa_idx.append(AA_TO_IDX[aa_from])
        mut_to_aa_idx.append(AA_TO_IDX[aa_to])
    
    mut_idx = torch.tensor([mut_idx], dtype=torch.long)
    mut_from_aa_idx = torch.tensor([mut_from_aa_idx], dtype=torch.long)
    mut_to_aa_idx = torch.tensor([mut_to_aa_idx], dtype=torch.long)
    
    return mut_idx, mut_from_aa_idx, mut_to_aa_idx

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1
    
def lig_atom_featurizer(smi):
    # mol = read_mols('.',pep_noh, remove_hs=False,sanitize=True)
    mol = Chem.MolFromSmiles(smi)
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_symbol_list'], atom.GetSymbol()),
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            safe_index(allowable_features['possible_total_valence_list'],atom.GetTotalValence()),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)

def GetMACCSKeys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_str = fp.ToBitString()
    fp_array = np.array([int(i) for i in fp_str])
    return fp_array

def get_chiralcenters(smi):
    # mol = read_mols('.',pep_noh, remove_hs=False,sanitize=True)
    mol = Chem.MolFromSmiles(smi)
    try:
        chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
    except:
        chiralcenters = []
    chiral_arr = torch.zeros([mol.GetNumAtoms()])
    for (i, rs) in chiralcenters:
        if rs == 'R':
            chiral_arr[i] =1
        elif rs == 'S':
            chiral_arr[i] =2 
        else:
            chiral_arr[i] =3 
    return chiral_arr

feature_dims = [
            len(value) for key, value in allowable_features.items() # Atom_features_dim
        ] + [4] # Atom_charity_center_dim


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        self.emb_dim = emb_dim
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        self.final_layer = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        x_embedding = self.final_layer(x_embedding)
        return x_embedding
    
class TabularDataset_MultiTask(data.ProteinDataset):
    def __init__(self, dataset_df, enzyme_lmdb, C, N, protein_lmdbkey='SaprotLmdbKey', mol_lmdb=None, weight=None,
                 mutate_pred=False, use_fingerprint=False, use_atom_feat=False,
                 dataset_root='./data', seq_max_length=1024, structure_path='', gnn='gearnet', **kwargs):
        # lmdb path
        self.enzyme_lmdb_path = enzyme_lmdb
        self.mol_lmdb_path = mol_lmdb

        # processed structure path
        self.structure_path = structure_path

        # structure file
        self.structure_file = dataset_df['StructureFile'].tolist()
        self.processed_structure_path = os.path.join(dataset_root, "protein_processed")

        # lmdb key
        self.enzyme_lmdb_key = dataset_df[protein_lmdbkey].tolist()
        self.mol_lmdb_key = dataset_df['MolLmdbKey'].tolist()
        # self.compare_lmdb_key = dataset_df['Compare_LmdbKey'].tolist()

        # feature control
        self.use_fingerprint = use_fingerprint
        self.use_atom_feat = use_atom_feat
        self.mutate_pred = mutate_pred

        # dataframe
        self.smiles = dataset_df['SMILES'].tolist()
        self.weight = weight
        self.kcat_log = dataset_df['kcat_log10'].tolist()
        self.km_log = dataset_df['km_log10'].tolist()
        self.eff_log = dataset_df['eff_log10'].tolist()

        # lmdb_env
        self.enzyme_env = None
        self.mol_env = None

        # tabular features
        self.C = torch.tensor(C).int()
        self.N = torch.tensor(N).float()


        # transforms
        transforms_keys = ['graph'] 

        self.transforms_func = transforms.transform.Compose([
            transforms.ProteinView(view='residue', keys=transforms_keys),
            transforms.TruncateProtein(max_length=seq_max_length, random=False, keys=transforms_keys),
        ])
        
        self.kwargs = kwargs

    def open_lmdb(self):
        if self.enzyme_env is None:
            self.enzyme_env = lmdb.open(
                self.enzyme_lmdb_path,
                readonly=True,
                create=False,
                max_readers=2048,
                lock=False,
            )
        if self.mol_env is None and self.mol_lmdb_path is not None:
            self.mol_env = lmdb.open(
                self.mol_lmdb_path,
                readonly=True,
                create=False,
                max_readers=2048,
                lock=False,
            )

    def get_item(self, index):
        item = {}

        # Resolve file paths robustly
        fname = self.structure_file[index]
        stem, _ = os.path.splitext(fname)
        struc_file_path = os.path.join(self.structure_path, f"{stem}.pkl")

        # LMDB
        self.open_lmdb()
        enzyme_lmdbkey = str(self.enzyme_lmdb_key[index])

        # Protein
        with open(struc_file_path, 'rb') as f:
            protein = pickle.load(f)
        item['graph'] = protein
        item['kcat_log'] = torch.tensor(self.kcat_log[index]).view(-1)
        item['km_log'] = torch.tensor(self.km_log[index]).view(-1)
        item['eff_log'] = torch.tensor(self.eff_log[index]).view(-1)

        # Protein residue features -> dense
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        # Apply transforms
        if self.transforms_func:
            item = self.transforms_func(item)

        # Molecule features
        smiles = self.smiles[index]

        if self.use_atom_feat:
            atom_arr = lig_atom_featurizer(smiles)
            chiral_arr = get_chiralcenters(smiles)
            mol_arr = torch.cat([atom_arr, chiral_arr.unsqueeze(-1)], dim=1)
            item['mol_arr'] = mol_arr
            item['atom_num'] = mol_arr.size(0)
            item['SMILES'] = smiles

        if self.use_fingerprint:
            maccs_keys = GetMACCSKeys(smiles)
            item['MACCSKeys'] = torch.tensor(maccs_keys).float()

        # Enzyme embeddings
        with self.enzyme_env.begin(write=False) as txn:
            residue_embedding = pickle.loads(txn.get(enzyme_lmdbkey.encode()))

        # Molecule embeddings (optional if LMDB provided)
        if self.mol_env is not None and self.mol_lmdb_key is not None:
            keys = self.mol_lmdb_key[index]
            keys = sorted(keys) if isinstance(keys, (list, tuple)) else [keys]
            mol_repr_list = []
            with self.mol_env.begin(write=False) as txn:
                for key in keys:
                    mol_embedding = pickle.loads(txn.get(str(key).encode()))
                    mol_repr_list.append(mol_embedding)
            if mol_repr_list:
                item['mol_embedding'] = torch.cat(mol_repr_list, dim=0)

        # Aggregate enzyme embedding
        enzyme_embedding = torch.mean(residue_embedding, dim=0)
        item['residue_embedding'] = residue_embedding
        item['enzyme_embedding'] = enzyme_embedding
        item['num_feat'] = self.N[index]
        item['cat_feat'] = self.C[index]
        if max(item['cat_feat']).item() == 9223372036854775804:
            print()
        # Sample weight
        if self.weight is not None:
            item['weight'] = self.weight[index]

        return item

    def __len__(self):
        return len(self.enzyme_lmdb_key)
