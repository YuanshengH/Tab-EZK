# Tab-EZK: A Physics-Informed Deep Learning Engine for Enzyme Kinetics

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

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

The evaluation script currently selects CUDA device `0` and does not provide a
CPU-only execution path.

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

For the fastest setup, use the provided precomputed data and model checkpoint:

- [Download the Tab-EZK data and checkpoint bundle](https://drive.google.com/file/d/1o-i4cl2u5j6cL5RDbutAeoQTuZxpD6ND/view?usp=sharing)
- [Download the precomputed AlphaFold structures](https://drive.google.com/file/d/1eSdL2tk5kX26Ls0XGVN_5Qk5_NpMiCxP/view?usp=drive_link)
- [Download the precomputed Uni-Mol2 representations](https://drive.google.com/file/d/1qMMzYWCXgrQHwGtotqDa7ON1Imv-4x2K/view?usp=drive_link)

Extract the required files under the repository root. The evaluation code
expects the following layout:

```text
Tab-EZK/
├── ckpt/
│   └── model.pth
└── data/
    ├── df_merge_tabular.csv
    ├── seed_0420/
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

`data/All_Structure/` and `model_cache/protT5/` are needed only when rebuilding
the processed features from source.

## Rebuilding Features (Optional)

The precomputed files are recommended for reproducing the reported evaluation.
The following steps are required only if you want to regenerate individual
feature sets.

### AlphaFold structures

You can use the
[precomputed structure archive](https://drive.google.com/file/d/1eSdL2tk5kX26Ls0XGVN_5Qk5_NpMiCxP/view?usp=drive_link)
or run:

```bash
python af_down_data.py
```

The current downloader uses legacy hard-coded paths: it reads
`data/test_split/tabular/merge/df_merge_tabular.csv`, writes PDB files to
`data/AFDB/All_Structure/`, and writes its log under
`data/AFDB/AF_structure/`. Create these directories before running the script.
To use the downloaded files with `data_process.ipynb`, move or link them into
`data/All_Structure/`.

### Process protein structures

`data_process.ipynb` converts the PDB files in `data/All_Structure/` into
pickled TorchDrug protein graphs under `data/processed_proteins/`. Ensure that
both directories exist before running the notebook.

Jupyter is not part of the core evaluation environment. If needed, install a
Jupyter frontend separately and run the notebook with the `tab-ezk` environment
as its kernel.

### ProtT5 representations

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

Then run:

```bash
python protT5_extract.py
```

The script writes residue-level representations to
`data/protT5/protT5.lmdb/`.

### Uni-Mol2 representations

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

### PyRosetta mutation workflow

`rosetta_mutate.py` is an optional structure-mutation workflow. PyRosetta is
distributed separately under the Rosetta software license and should be
installed in a dedicated environment rather than added to the main Tab-EZK
environment.

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

