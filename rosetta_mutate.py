import os
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb, Pose
from pyrosetta.toolbox import mutate_residue
from pyrosetta.rosetta.protocols.relax import FastRelax
import multiprocessing as mp

# 初始化 PyRosetta
pyrosetta.init()

# 定义一个函数，用于处理单个蛋白质结构的突变和松弛
def process_structure(row, structure_dir, output_dir):
    uid = row['UniprotID']
    enzymetype = row['Mutation']
    af_name = f"AF-{uid}-F1-model_v4.pdb"
    
    if enzymetype != 'wildtype' and af_name in os.listdir(structure_dir):
        mutate_label = '_'.join(enzymetype.split('/'))
        output_pdb_file = os.path.join(output_dir, f"AF-{uid}-F1-model_v4_{mutate_label}.pdb")
        if os.path.exists(output_pdb_file):
            return
        mutated_site = enzymetype.split('/')
        pose = pose_from_pdb(os.path.join(structure_dir, af_name))
        
        for m in mutated_site:
            aa = m[-1]
            site = int(m[1:-1])
            mutate_residue(pose, site, aa)

        relax = FastRelax()
        relax.set_scorefxn(pyrosetta.get_fa_scorefxn())
        relax.max_iter(10) 
        
        print(f"Relaxing the structure for {uid}...")
        relax.apply(pose)
        print(f"Relaxation complete for {uid}.")

        pose.dump_pdb(output_pdb_file)
        print(f"Mutated and relaxed structure saved to {output_pdb_file}")

structure_dir = './data/All_Structure'
output_dir = './data/All_Structure'
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
df = pd.read_csv(f'./data/df_merge_tabular.csv')

exit_struc = os.listdir('./data/All_Structure')
df = df[~df['StructureFile'].isin(exit_struc)]
df.drop_duplicates(subset=['UniprotID', 'Mutation'], inplace=True)

num_processes = mp.cpu_count()  
with mp.Pool(processes=num_processes) as pool:
    pool.starmap(process_structure, [(row, structure_dir, output_dir) for _, row in df.iterrows()])