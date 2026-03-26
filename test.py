import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sklearn
import torch
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from loguru import logger
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tabular_dataset import TabularDataset_MultiTask
from utils import tabular_collate_extract, cuda, same_seed, normalize_test, get_categories
from model_structure.tabular_model import KineticModel_Tabular_MultiTask


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


def main(args):
    logger.add(os.path.join(os.path.dirname(args.ckpt), 'test.log'))
    logger.info(args)
    same_seed(args.seed)
    device = torch.device('cuda')

    df = pd.read_csv(f'./data/test_split/tabular/merge/df_merge_tabular.csv')
    df = df.replace('-', np.nan)

    train_df, _ = train_test_split(df, train_size=0.8, random_state=args.seed)
    test_kcat_df = pd.read_csv(f'./data/test_split/tabular/merge/seed/{args.seed}/test_kcat.csv')
    test_km_df = pd.read_csv(f'./data/test_split/tabular/merge/seed/{args.seed}/test_km.csv')
    test_eff_df = pd.read_csv(f'./data/test_split/tabular/merge/seed/{args.seed}/test_kcat_km.csv')

    test_kcat_df = test_kcat_df.replace('-', np.nan)
    test_km_df = test_km_df.replace('-', np.nan)
    test_eff_df = test_eff_df.replace('-', np.nan)
    test_kcat_df['km_log10'], test_kcat_df['eff_log10'] = 1, 1
    test_km_df['kcat_log10'], test_km_df['eff_log10'] = 1, 1
    test_eff_df['kcat_log10'], test_eff_df['km_log10'] = 1, 1

    train_N = np.array(train_df[nums_names].values).astype(float)
    train_C = np.array(train_df[categories_names].values).astype(str)

    test_kcat_N = np.array(test_kcat_df[nums_names].values).astype(float)
    test_kcat_C = np.array(test_kcat_df[categories_names].values).astype(str)
    test_km_N = np.array(test_km_df[nums_names].values).astype(float)
    test_km_C = np.array(test_km_df[categories_names].values).astype(str)
    test_eff_N = np.array(test_eff_df[nums_names].values).astype(float)
    test_eff_C = np.array(test_eff_df[categories_names].values).astype(str)

    args.numerical_features = len(nums_names)
    train_num_nan_masks = np.isnan(train_N)
    test_kcat_num_nan_masks = np.isnan(test_kcat_N)
    test_km_num_nan_masks = np.isnan(test_km_N)
    test_eff_num_nan_masks = np.isnan(test_eff_N)

    if train_num_nan_masks.any():
        num_new_values = np.nanmean(train_N, axis=0)

        train_num_nan_indices = np.where(train_num_nan_masks)
        train_N[train_num_nan_indices] = np.take(num_new_values, train_num_nan_indices[1])
        
        test_kcat_num_nan_indices = np.where(test_kcat_num_nan_masks)
        test_kcat_N[test_kcat_num_nan_indices] = np.take(num_new_values, test_kcat_num_nan_indices[1])

        test_km_num_nan_indices = np.where(test_km_num_nan_masks)
        test_km_N[test_km_num_nan_indices] = np.take(num_new_values, test_km_num_nan_indices[1])

        test_eff_num_nan_indices = np.where(test_eff_num_nan_masks)
        test_eff_N[test_eff_num_nan_indices] = np.take(num_new_values, test_eff_num_nan_indices[1])

    if args.normalization:
        N = normalize_test(train_N, test_kcat_N, test_km_N, test_eff_N, args.normalization, args.seed)

    unknown_value = np.iinfo('int64').max - 3 
    encoder = sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=unknown_value, dtype='int64').fit(train_C)
    train_C = encoder.transform(train_C)
    test_kcat_C = encoder.transform(test_kcat_C)
    test_km_C = encoder.transform(test_km_C)
    test_eff_C = encoder.transform(test_eff_C)
        
    args.categorical_features = get_categories(torch.tensor(train_C))


    test_kcat_df['kcat_log10'] = test_kcat_df['kcat'].apply(lambda x: np.log10(x))
    test_km_df['km_log10'] = test_km_df['km'].apply(lambda x: np.log10(x))
    test_eff_df['eff_log10'] = test_eff_df['kcat_km'].apply(lambda x: np.log10(x))

    with open(f'./data/Uni-Mol2/tabular_unimol_smile_dict.pk', 'rb') as f:
        itosmiles, smilestoi = pickle.load(f)
    test_kcat_df['MolLmdbKey'] = test_kcat_df['SMILES'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    test_km_df['MolLmdbKey'] = test_km_df['SMILES'].apply(lambda x: [smilestoi[i] for i in x.split('.')])
    test_eff_df['MolLmdbKey'] = test_eff_df['SMILES'].apply(lambda x: [smilestoi[i] for i in x.split('.')])

    enzyme_lmdb_path = f'./data/ProtT5/protT5.lmdb'
    args.enzyme_input_dim = 1024
    args.mol_lmdb = f'./data/Uni-Mol2/tabular_unimol_1.1B.lmdb'

    kcat_testset = TabularDataset_MultiTask(dataset_df=test_kcat_df, enzyme_lmdb=enzyme_lmdb_path,  
                        mol_lmdb=args.mol_lmdb, structure_path='./data/AFDB/processed_proteins', 
                        mutate_pred=args.mutate_pred,  use_fingerprint=args.use_fingerprint, use_atom_feat=args.use_atom_feature,
                        protein_lmdbkey='LmdbKey', N=N['test_kcat'], C=test_kcat_C)
    kcat_test_loader = DataLoader(kcat_testset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=tabular_collate_extract,)
    
    km_testset = TabularDataset_MultiTask(dataset_df=test_km_df, enzyme_lmdb=enzyme_lmdb_path,  
                        mol_lmdb=args.mol_lmdb, structure_path='./data/AFDB/processed_proteins', 
                        mutate_pred=args.mutate_pred,  use_fingerprint=args.use_fingerprint, use_atom_feat=args.use_atom_feature,
                        protein_lmdbkey='LmdbKey', N=N['test_km'], C=test_km_C)
    km_test_loader = DataLoader(km_testset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=tabular_collate_extract,)
    
    eff_testset = TabularDataset_MultiTask(dataset_df=test_eff_df, enzyme_lmdb=enzyme_lmdb_path,  
                        mol_lmdb=args.mol_lmdb, structure_path='./data/AFDB/processed_proteins', 
                        mutate_pred=args.mutate_pred,  use_fingerprint=args.use_fingerprint, use_atom_feat=args.use_atom_feature,
                        protein_lmdbkey='LmdbKey', N=N['test_eff'], C=test_eff_C)
    eff_test_loader = DataLoader(eff_testset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=tabular_collate_extract,)

    fusion_model = KineticModel_Tabular_MultiTask(args)

    if args.ckpt:
        ckpt_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
        state_dict = ckpt_dict['model']
        fusion_model.load_state_dict(state_dict)

    fusion_model.to(device)
    fusion_model.eval()
        
    # kcat test
    kcat_test_means_list = []
    kcat_test_lambdas_list = []
    kcat_test_alphas_list = []
    kcat_test_betas_list = []
    kcat_test_targets_list = []
    with torch.no_grad():
        for batch in tqdm(kcat_test_loader):
            batch = cuda(batch, device=device)
            yk = batch["kcat_log"].float().squeeze(1).to(device)
            (nig_k, nig_m, nig_eff, moe_aux_loss) = fusion_model(batch)

            mu_k, lam_k, alpha_k, beta_k = nig_k

            kcat_test_means_list.extend(mu_k.data.cpu().tolist())
            kcat_test_lambdas_list.extend(lam_k.data.cpu().tolist())
            kcat_test_alphas_list.extend(alpha_k.data.cpu().tolist())
            kcat_test_betas_list.extend(beta_k.data.cpu().tolist())
            kcat_test_targets_list.extend(yk.data.cpu().tolist())

    # evidential distribution
    kcat_test_out, kcat_test_metric = evidential_metrics(kcat_test_means_list, kcat_test_lambdas_list, kcat_test_alphas_list, kcat_test_betas_list, kcat_test_targets_list)
    kcat_test_out.update({
        'SMILES': test_kcat_df['SMILES'].tolist(),
        'LmdbKey': test_kcat_df['LmdbKey'].tolist(),
        'Target':np.array(kcat_test_targets_list),
    })
    
    rmse, mae, r2, pcc = kcat_test_metric["RMSE"], kcat_test_metric["MAE"], kcat_test_metric["R2"], kcat_test_metric["PearsonR"]
    print(f"kcat test r2: {r2},  test MAE: {mae}, test_PCC: {pcc}, test rmse: {rmse}")
    logger.info(f"kcat test r2: {r2}, test MAE: {mae}, test_PCC: {pcc}, test rmse: {rmse}")
    kcat_result_df = pd.DataFrame(kcat_test_out)

    # km test
    km_test_means_list = []
    km_test_lambdas_list = []
    km_test_alphas_list = []
    km_test_betas_list = []
    km_test_targets_list = []
    with torch.no_grad():
        for batch in tqdm(km_test_loader):
            batch = cuda(batch, device=device)
            ym = batch["km_log"].float().squeeze(1).to(device)
            (nig_k, nig_m, nig_eff, moe_aux_loss) = fusion_model(batch)

            mu_m, lam_m, alpha_m, beta_m = nig_m

            km_test_means_list.extend(mu_m.data.cpu().tolist())
            km_test_lambdas_list.extend(lam_m.data.cpu().tolist())
            km_test_alphas_list.extend(alpha_m.data.cpu().tolist())
            km_test_betas_list.extend(beta_m.data.cpu().tolist())
            km_test_targets_list.extend(ym.data.cpu().tolist())

    # evidential distribution
    km_test_out, km_test_metric = evidential_metrics(km_test_means_list, km_test_lambdas_list, km_test_alphas_list, km_test_betas_list, km_test_targets_list)
    km_test_out.update({
        'SMILES': test_km_df['SMILES'].tolist(),
        'LmdbKey': test_km_df['LmdbKey'].tolist(),
        'Target':np.array(km_test_targets_list),
    })
    rmse, mae, r2, pcc = km_test_metric["RMSE"], km_test_metric["MAE"], km_test_metric["R2"], km_test_metric["PearsonR"]
    print(f"km test r2: {r2},  test MAE: {mae}, test_PCC: {pcc}, test rmse: {rmse}")
    logger.info(f"km test r2: {r2}, test MAE: {mae}, test_PCC: {pcc}, test rmse: {rmse}")
    km_result_df = pd.DataFrame(km_test_out)

    # eff test
    eff_test_means_list = []
    eff_test_lambdas_list = []
    eff_test_alphas_list = []
    eff_test_betas_list = []
    eff_test_targets_list = []
    with torch.no_grad():
        for batch in tqdm(eff_test_loader):
            batch = cuda(batch, device=device)
            yeff = batch["eff_log"].float().squeeze(1).to(device)
            (nig_k, nig_m, nig_eff, moe_aux_loss) = fusion_model(batch)

            mu_eff, lam_eff, alpha_eff, beta_eff = nig_eff

            eff_test_means_list.extend(mu_eff.data.cpu().tolist())
            eff_test_lambdas_list.extend(lam_eff.data.cpu().tolist())
            eff_test_alphas_list.extend(alpha_eff.data.cpu().tolist())
            eff_test_betas_list.extend(beta_eff.data.cpu().tolist())
            eff_test_targets_list.extend(yeff.data.cpu().tolist())
    
    # evidential distribution
    eff_test_out, eff_test_metric = evidential_metrics(eff_test_means_list, eff_test_lambdas_list, eff_test_alphas_list, eff_test_betas_list, eff_test_targets_list)
    eff_test_out.update({
        'SMILES': test_eff_df['SMILES'].tolist(),
        'LmdbKey': test_eff_df['LmdbKey'].tolist(),
        'Target':np.array(eff_test_targets_list),
    })
    rmse, mae, r2, pcc = eff_test_metric["RMSE"], eff_test_metric["MAE"], eff_test_metric["R2"], eff_test_metric["PearsonR"]
    print(f"eff test r2: {r2},  test MAE: {mae}, test_PCC: {pcc}, test rmse: {rmse}")
    logger.info(f"eff test r2: {r2}, test MAE: {mae}, test_PCC: {pcc}, test rmse: {rmse}")

    eff_result_df = pd.DataFrame(eff_test_out)

def evidential_metrics(
    means, lambdas, alphas, betas, targets=None,
    clip_eps=1e-8, use_entropy=True,
):

    mu = np.asarray(means, dtype=np.float64).reshape(-1)
    lam = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alphas, dtype=np.float64).reshape(-1)
    beta  = np.asarray(betas,  dtype=np.float64).reshape(-1)
    v = lam

    inverse_evidence = 1.0 / ((alpha - 1.0) * lam)
    Confidence = inverse_evidence     # 不确定性
    Confidence = Confidence.clip(min=clip_eps)
    Confidence = np.sqrt(Confidence)

    ale_var = beta / (alpha - 1.0)        # Aleatoric
    epi_var = beta / ((alpha - 1.0) * v)            # Epistemic
    ale_var = np.clip(ale_var, clip_eps, None)
    epi_var = np.clip(epi_var, clip_eps, None)
    total_var_nig = ale_var + epi_var          

    ale_std = np.sqrt(ale_var)
    epi_std = np.sqrt(epi_var)    
    total_std = np.sqrt(np.clip(total_var_nig, clip_eps, None))

    if use_entropy:
        def gaussian_entropy(std):
            return 0.5 * np.log(2 * np.pi * np.e * std**2)
        entropy = gaussian_entropy(epi_std)
    else: 
        entropy = None          # nats

    out = {
        "Predict": mu,
        "Inverse_Evidence": Confidence,
        "AleatoricVar": ale_var,
        "EpistemicVar": epi_var,
        "TotalVar": total_var_nig,          
        "TotalStd": total_std,
        "AleatoricStd": ale_std,
        "EpistemicStd": epi_std,
        "entropy": entropy,     
    }

    if targets is not None:
        metric = {}
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        mae = np.mean(np.abs(y - mu))
        rmse = np.sqrt(mean_squared_error(y, mu))
        r2 = r2_score(y, mu)
        pr, _ = pearsonr(y, mu)
        metric = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "PearsonR": pr
        }

    return out, metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--normalization', default='quantile', choices=[None, 'standard', 'quantile'], type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt', default='./ckpt/tabular_multitask_cross_val/42/aaa_save_weight_epoch50_dim_512_cons0.1_head4_lam1_0.3—33-no_aug-best/model.pth', help="train from checkpoint")

    # Encoder setting
    parser.add_argument('--mutate_pred', action='store_true', default=True)
    parser.add_argument('--use_atom_feature', action='store_true', default=True)
    parser.add_argument('--use_fingerprint', action='store_true', default=True)

    # path
    parser.add_argument('--mol_lmdb', default=None)

    # GearNetdo
    parser.add_argument('--enzyme_input_dim', default=1024, type=int)

    # substrate
    parser.add_argument('--atom_feat_dim', default=128, type=int)
    parser.add_argument('--fingerprint_dim', default=64, type=int)
    parser.add_argument('--unimol_dim', default=1536, type=int)

    # attn
    parser.add_argument('--bridge_dim', default=512, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--num_cross_attn', default=1, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    
    # MoE
    parser.add_argument('--moe_top_k', default=3, type=int)
    parser.add_argument('--moe_mode', default='concat', type=str)
    parser.add_argument('--moe_num_experts', default=3, type=int)

    args = parser.parse_args()
    main(args)
