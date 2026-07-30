# Tab-EZK: A Physics-Informed Deep Learning Engine for Enzyme Kinetics


This repository provides the implementation of **A Physics-Informed Deep
Learning Engine for Predicting Enzyme Kinetics Under Heterogeneous
Experimental Conditions**.

## Overview

Tab-EZK is a physics-informed, structure-aware multitask framework for
predicting enzyme turnover numbers ($k_{\mathrm{cat}}$), Michaelis constants
($K_{\mathrm{m}}$), and catalytic efficiencies
($k_{\mathrm{cat}}/K_{\mathrm{m}}$). It combines:

- a GearNet-based graph neural network for three-dimensional protein
  structures;
- ProtT5 protein representations and Uni-Mol2 molecular representations;
- a Transformer-based tabular encoder for experimental conditions such as pH
  and temperature; and
- a mixture-of-experts fusion module for joint kinetic prediction.

The supported evaluation entry point is `test.py`.

## System Requirements

- Linux x86-64
- An NVIDIA GPU
- An NVIDIA driver compatible with CUDA 12.1
- Conda or Mamba

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YuanshengH/Tab-EZK.git
cd Tab-EZK
```

### 2. Create the Conda environment

The primary and recommended environment configuration is
[`environment.yml`](environment.yml).

```bash
conda env create -f environment.yml
conda activate tab-ezk
```

The environment uses Python 3.10, PyTorch 2.1.0, and CUDA 12.1. PyTorch, PyG,
and their compiled CUDA extensions are pinned to mutually compatible builds.

You can verify the main installation with:

```bash
python -c "import torch, torchdrug, torch_geometric; print(torch.__version__, torch.version.cuda)"
```

## Data and Model Files

Download the base dataset and checkpoint together with the two feature
resources that are available as precomputed archives:

- [Download the Tab-EZK data and checkpoint bundle](https://drive.google.com/file/d/1-3whB6ivC85TJ-KK-C4nVt9W2rNq_lNn/view?usp=sharing)
- [Download the precomputed protein structure archive](https://drive.google.com/file/d/1eSdL2tk5kX26Ls0XGVN_5Qk5_NpMiCxP/view?usp=drive_link)
- [Download the precomputed Uni-Mol2 representations](https://drive.google.com/file/d/1qMMzYWCXgrQHwGtotqDa7ON1Imv-4x2K/view?usp=drive_link)

Only the raw protein structures and Uni-Mol2 representations are provided as
precomputed feature downloads. The processed protein graphs and ProtT5
representations must be generated locally before evaluation by following the
required steps under [Rebuilding Features](#rebuilding-features).

The precomputed structure archive contains both the downloaded AlphaFold2
reference structures and the PyRosetta-generated structures for point-mutant
proteins in the dataset. When using this archive, you do not need to run
`af_download.py` or `rosetta_mutate.py`.

Extract the downloaded files under the repository root. During feature
preparation and evaluation, the relevant files use the following layout:

```text
Tab-EZK/
├── ckpt/
│   └── model.pth
└── data/
    ├── df_merge_tabular.csv
    ├── All_Structure/
    │   └── *.pdb
    ├── seed_split/
    │   └── 42/
    │       ├── test_kcat.csv
    │       ├── test_km.csv
    │       └── test_kcat_km.csv
    ├── processed_proteins/
    │   └── *.pkl
    ├── protT5/
    │   └── protT5.lmdb/
    │       ├── data.mdb
    │       └── lock.mdb
    └── Uni-Mol2/
        ├── tabular_unimol_smile_dict.pk
        └── tabular_unimol_1.1B.lmdb/
            ├── data.mdb
            └── lock.mdb
```

`data/All_Structure/` contains the raw reference and mutant PDB files and is
needed only when rebuilding the processed protein graphs.

## Rebuilding Features

Feature preparation combines downloadable resources with two mandatory local
processing steps:

| Feature | Precomputed download | Required action |
| --- | --- | --- |
| Raw protein structures | Available | Download the structure archive, or regenerate it with AlphaFold2 and PyRosetta |
| Processed protein graphs | Not available | Run all cells in `data_process.ipynb` |
| ProtT5 representations | Not available | Run `protT5_extract.py` |
| Uni-Mol2 representations | Available | Download the Uni-Mol2 archive, or regenerate it with `unimol_extract.py` |

The processed protein graphs and ProtT5 representations are required even when
the two precomputed archives are used.

### Protein structures (download or regenerate)

The dataset contains both wild-type proteins and proteins with one or more
point mutations. The structures downloaded from AlphaFold2 correspond to the
UniProt reference sequence and are not mutation-specific. Therefore, rebuilding
the complete structure set requires two stages: downloading the reference
structures and then generating the mutant structures with PyRosetta.

The simplest option is to use the
[precomputed structure archive](https://drive.google.com/file/d/1eSdL2tk5kX26Ls0XGVN_5Qk5_NpMiCxP/view?usp=drive_link)
described above. It already includes the generated mutant structures.

To rebuild the structures from source, first create the output directory and
download the AlphaFold2 reference structures:

```bash
mkdir -p data/All_Structure
python af_download.py
```

`af_download.py` reads UniProt identifiers from `data/df_merge_tabular.csv` and
writes the downloaded PDB files to `data/All_Structure/`.

Next, install PyRosetta in a separate environment and run:

```bash
python rosetta_mutate.py
```

`rosetta_mutate.py` reads the `UniprotID` and `Mutation` fields from
`data/df_merge_tabular.csv`, introduces the specified amino-acid substitutions
into the corresponding AlphaFold2 reference structure, performs structural
relaxation, and saves the resulting mutant PDB files in
`data/All_Structure/`. PyRosetta is distributed separately under the Rosetta
software license and is intentionally not included in `environment.yml`.

### Process protein structures (required)

After both the reference and mutant PDB files are available,
`data_process.ipynb` converts the structures in `data/All_Structure/` into
pickled TorchDrug protein graphs under `data/processed_proteins/`. This step is
mandatory; the structure archive contains PDB files rather than the processed
TorchDrug graphs.

Create the output directory, open `data_process.ipynb` with the `tab-ezk`
environment as its kernel, and run all cells from top to bottom:

```bash
mkdir -p data/processed_proteins
```

After the notebook finishes, confirm that the generated `.pkl` files are
present under `data/processed_proteins/`.

### ProtT5 representations (required)

Download the
[ProtT5-XL-UniRef50 model](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)
and place it under `model_cache/protT5/`:

```text
model_cache/
└── protT5/
    ├── config.json
    ├── special_tokens_map.json
    ├── spiece.model
    ├── tokenizer_config.json
    └── model weight file(s), such as pytorch_model.bin or model.safetensors
```

No precomputed ProtT5 representation archive is provided. Create the output
directory and run the extraction script:

```bash
mkdir -p data/protT5
python protT5_extract.py
```

The script writes residue-level representations to
`data/protT5/protT5.lmdb/`.

### Uni-Mol2 representations (download or regenerate)

The recommended approach is to use the
[precomputed Uni-Mol2 archive](https://drive.google.com/file/d/1qMMzYWCXgrQHwGtotqDa7ON1Imv-4x2K/view?usp=drive_link).

To regenerate these representations, create a separate environment following
the [Uni-Mol Tools documentation](https://github.com/deepmodeling/unimol_tools)
and run:

```bash
python unimol_extract.py
```

`unimol_tools` is intentionally excluded from the main environment because its
current dependency stack requires newer PyTorch and NumPy versions. 

## Evaluation

After activating the environment and placing the data and checkpoint in the
expected locations, run:

```bash
conda activate tab-ezk
python test.py --ckpt ckpt/model.pth
```

For systems with limited GPU memory or fewer CPU cores, reduce the batch size
and number of data-loader workers:

```bash
python test.py --ckpt ckpt/model.pth --batch_size 16 --num_workers 4
```

The script evaluates $k_{\mathrm{cat}}$, $K_{\mathrm{m}}$, and
$k_{\mathrm{cat}}/K_{\mathrm{m}}$, and reports RMSE, MAE, $R^2$, and Pearson
correlation.
