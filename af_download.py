import os
import subprocess
import pandas as pd

def cmd(command):
    subp = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    
    # Real-time output reading
    while True:
        output = subp.stdout.readline()
        if output == '' and subp.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Check for any errors
    err = subp.stderr.read()
    if err:
        print(err)
    
    # Final exit status check
    if subp.poll() != 0:
        print(f"{command} Failure!")

merge_csv = './data/df_merge_tabular.csv'
df_merge = pd.read_csv(merge_csv, usecols=['UniprotID'])

uniprot_ids = []
for val in df_merge['UniprotID'].dropna().astype(str).tolist():
    # 兼容可能的多 ID 形式，如 "P12345;Q8XXX" 或空白分隔
    parts = val.replace(';', ' ').split()
    for p in parts:
        p = p.strip()
        if p:
            uniprot_ids.append(p)

uniprot_ids = sorted(set(uniprot_ids))


work_dir = './data/All_Structure'
url_template = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"

existed_list = os.listdir(work_dir)
existed_list = [i[3:-16] for i in existed_list]
uid_list = [i for i in uniprot_ids if i not in existed_list]

url_list = [url_template.format(x.strip().replace(';', '')) for x in uid_list]
urls_file = './data/All_Structure/url.txt'
with open(urls_file, "w") as f:
    f.write("\n".join(url_list))

cmd_str = f"aria2c --max-tries=10 --retry-wait=2 -i {urls_file} -x 16 -d {work_dir} -l ./data/All_Structure/af_download.log"
cmd(cmd_str)
print()