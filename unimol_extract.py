import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import torch
import lmdb
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from unimol_tools import UniMolRepr

def main():
    model_size = '1.1B'
    clf = UniMolRepr(data_type='molecule', 
                    remove_hs=False,
                    model_name='unimolv2', 
                    model_size=model_size,
                    use_gpu=True 
                    )

    df = pd.read_csv(f'data/df_merge_tabular.csv')
    smiles_list = list(set(df['SMILES'].values.tolist()))
    smiles_list = list(sorted(list(set(smiles_list))))

    itosmiles = set()
    for s in smiles_list:
        itosmiles.add(s)
    itosmiles = sorted(list(itosmiles))
    smilestoi = {itosmiles[i]:i for i in range(len(itosmiles))}

    with open(f'data/Uni-Mol2/tabular_unimol_smile_dict.pk', 'wb') as f:
        pickle.dump([itosmiles,smilestoi],f)

    lmdb_path = f'data/Uni-Mol2/tabular_unimol_1.1B.lmdb'
    env = lmdb.open(lmdb_path, map_size=2199023255556)
    with env.begin(write=True) as txn:
        for idx,smi in tqdm(enumerate(itosmiles)):
            if smi == '[H+]' or smi=='[HH]':
                continue
            name = str(smilestoi[smi])
            tmp_list = [smi]
            unimol_repr = clf.get_repr(tmp_list, return_atomic_reprs=True)
            full_repr = torch.cat((torch.tensor(unimol_repr['cls_repr']), torch.tensor(unimol_repr['atomic_reprs']).squeeze(0)), dim=0)
            txn.put(name.encode(), pickle.dumps(full_repr))
    env.close()


if __name__ == '__main__':
    main()