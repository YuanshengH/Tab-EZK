import os 
import pickle
import argparse
import lmdb
import pandas as pd 
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from tqdm import tqdm 


df = pd.read_csv('./data/df_merge_tabular.csv')
df.drop_duplicates(subset=['LmdbKey'], inplace=True)
df.reset_index(drop=True, inplace=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('./model_cache/protT5', do_lower_case=False)
model = T5EncoderModel.from_pretrained("./model_cache/protT5").to(device)

lmdb_path = './data/protT5/protT5.lmdb'
env = lmdb.open(lmdb_path, map_size=9199023255556)
with env.begin(write=True) as txn:
    for index, row in tqdm(df.iterrows()):
        seq = row['Sequence']
        name = row['LmdbKey']

        # preprocess sequence
        sequence = re.sub(r"[UZOB]", "X", seq)
        sequence = [" ".join(list(sequence))]
        ids = tokenizer(sequence, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = embedding_repr.last_hidden_state[0,1:len(seq)+1].cpu()
        txn.put(name.encode(), pickle.dumps(emb))
env.close()
            