import torch
import functools
import wandb
import random
import math
import sklearn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import Counter
from scipy.ndimage import convolve1d
from torch.nn.utils.rnn import pad_sequence   
from argparse import Namespace
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torchdrug import data
from typing import List, Union
from collections.abc import Mapping, Sequence
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score
from transformers.tokenization_utils_base import BatchEncoding
# from data_loaders.rxn_dataloader import pad_atom_distance_matrix
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from torch_geometric.data import Batch

def setup_wandb(args, model):
    """封装所有 wandb 初始化逻辑"""
    run = wandb.init(
        project="enzyme-kinetics",
        name=args.wandb_name,
        config=vars(args),
        dir=args.ckpt_save_path,
        save_code=True,
        mode='offline'
    )
    
    wandb.save("./model_structure/tabular_model.py")
    wandb.save("./train_tabular_multitask.py")
    
    wandb.watch(model)

    metrics = {
        "train": ["loss", "kcat_loss", "km_loss", "eff_loss", "cons_loss", "MoE_aux_loss"],
        "valid": ["loss_total", "loss_kcat", "loss_km", "loss_eff", "loss_cons", "loss_MoE_aux"]
    }
    
    for prefix, names in metrics.items():
        step_metric = f"{prefix}_global_step"
        for name in names:
            wandb.define_metric(f"{prefix}/{name}", step_metric=step_metric)
            
    for stage in ["train", "valid"]:
        for task in ["kcat", "km", "eff"]:
            for metric in ["MSE", "rmse", "r2", "pearson", "spearman"]:
                wandb.define_metric(f"{stage}/{task}_{metric}", step_metric="valid_global_step")
                
    return run

def log_regression_table(y_true, y_pred, task_name, stage="valid"):
    table = wandb.Table(columns=["Target", "Prediction", "Error"])
    for t, p in zip(y_true, y_pred):
        table.add_data(t, p, abs(t - p))
    
    wandb.log({f"{stage}/{task_name}_predictions": table})
    
def normalize(
    X_train, X_valid, X_test, normalization: str, seed: int, noise: float = 1e-3):
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        if noise:
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
                X_train.shape
            )
    normalizer.fit(X_train)
    return {
        'train': normalizer.transform(X_train),
        'valid': normalizer.transform(X_valid),
        'test': normalizer.transform(X_test),
    }

def normalize_test(
    X_train, X_test_kcat, X_test_km, X_test_eff, normalization: str, seed: int, noise: float = 1e-3):
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        if noise:
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
                X_train.shape
            )
    normalizer.fit(X_train)
    return {
        'train': normalizer.transform(X_train),
        'test_kcat': normalizer.transform(X_test_kcat),
        'test_km': normalizer.transform(X_test_km),
        'test_eff': normalizer.transform(X_test_eff),
    }


def get_categories(X_cat):
    return (
        None
        if X_cat is None
        else [
            len(set(X_cat[:, i].cpu().tolist())) 
            for i in range(X_cat.shape[1])
        ]
    )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evidential_loss_new(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    # lam = 0.2
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg         # torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss

def enzyme_rxn_collate(batch, key=None):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container

    residue_feature: (Num_residue, dim)
    enzyme_feature: (batch_size, dim)
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.untyped_storage()._new_shared(numel)
            out = elem.new(storage)
        if elem.dim() == 1:      
            return torch.stack(batch)
        elif elem.shape[-1] == 1024 and elem.dim()==2:      # protT5
            return torch.cat(batch, 0,)
        elif elem.shape[-1] == 19 and elem.dim()==2:
            return torch.cat(batch, 0,)
        elif elem.shape[-1] == 2560 and elem.dim()==2:
            
            return torch.cat(batch, 0,)
        elif elem.shape[-1] == 1536 and key=='mol_embedding':    # Uni-Mol v2
            return pad_sequence(batch, padding_value=0, batch_first=True)
        elif elem.shape[-1] == 1536 and key!='mol_embedding':   
            return torch.cat(batch, 0,)
        elif elem.shape[-1] == 167:    
            return torch.cat(batch, 0,)
        
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int): # atom num
        return batch
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: enzyme_rxn_collate([d[key] for d in batch], key=key) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in list of batch should be of equal size")
        return [enzyme_rxn_collate(samples) for samples in zip(*batch)]

    elif elem is None:
        return None
    
    elif isinstance(elem, np.ndarray):
        return np.stack(batch, 0)
    raise TypeError("Can't collate data with type `%s`" % type(elem))

def tabular_collate_extract(batch):
    item = enzyme_rxn_collate(batch)

    kcat_mask = ~torch.isnan(item["kcat_log"])
    km_mask = ~torch.isnan(item["km_log"])
    eff_mask = ~torch.isnan(item["eff_log"])

    item['kcat_mask'] = kcat_mask
    item['km_mask'] = km_mask
    item['eff_mask'] = eff_mask
    assert isinstance(item, dict)
    return item

def enzyme_rxn_collate_extract_finetune(batch):
    item = [x[0] for x in batch if x]
    esm_batch_token = pad_sequence([x[1][0].clone().detach() for x in batch], padding_value=1, batch_first=True)

    batch_data = enzyme_rxn_collate(item)
    assert isinstance(batch_data, dict)
    batch_data['esm_batch_token'] = esm_batch_token

    return batch_data


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    
    # elif isinstance(obj, dgl.DGLGraph):
    #     return obj.to(*args, **kwargs)
    elif isinstance(obj, BatchEncoding):
        return obj.to(*args, **kwargs)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj
    elif obj is None:
        return obj
    raise TypeError("Can't transfer object type `%s`" % type(obj))

