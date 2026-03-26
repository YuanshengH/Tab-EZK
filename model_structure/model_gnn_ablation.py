import os
import yaml
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Sequence

from dataset import allowable_features
from model_structure.FusionGearNet import EnzymeFusionNetwork, PocketNetwork
from model_structure.Attn_module import (
    GlobalMultiHeadAttention,
    GELU,
    SelfMultiHeadAttention
)
from model_structure.rxn_attn_model import (
    ReactionMGMTurnNet as RXNAttnNetwork,
)
from model_structure.MoE import MoE, MoE_old, KineticsMoE, KineticsMoEChannel

feature_dims = [
            len(value) for key, value in allowable_features.items() # Atom_features_dim
        ] + [4] 

def pack_residue_feats(packedgraph, residue_feats):
    num_residues = packedgraph.num_residues.tolist()
    edit_feats = torch.split(residue_feats, num_residues)
    masks = [
        torch.ones(num_residue, dtype=torch.uint8, device=residue_feats.device)
        for num_residue in num_residues
    ]
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    return padded_feats, masks

def read_model_state(model_save_path):
    model_state_fname = os.path.join(model_save_path, 'model.pth')
    args_fname = os.path.join(model_save_path, 'args.yml')
    # eval_results_fname = os.path.join(model_save_path, 'eval_results.csv')

    model_state = torch.load(model_state_fname,
                             map_location=torch.device('cpu'))
    keys = list(model_state.keys())
    if 'module.' in keys[0]:
        model_state = {k.replace('module.', ''): v for k,v in model_state.items()}
    args = yaml.load(open(args_fname, "r"), Loader=yaml.FullLoader)

    return model_state, args

def load_pretrain_model_state(model, pretrained_state, load_active_net=True):
    model_state = model.state_dict()
    pretrained_state_filter = {}
    extra_layers = []
    different_shape_layers = []
    need_train_layers = []
    for name, parameter in pretrained_state.items():
        if name in model_state and parameter.size() == model_state[name].size():
            pretrained_state_filter[name] = parameter
        elif name not in model_state:
            extra_layers.append(name)
        elif parameter.size() != model_state[name].size():
            different_shape_layers.append(name)
        else:
            pass
    if not load_active_net:
        for name, parameter in model_state.items():
            if 'active_net' in name:
                del pretrained_state_filter[name]
    
    for name, parameter in model_state.items():
        if name not in pretrained_state_filter:
            need_train_layers.append(name)

    model_state.update(pretrained_state_filter)
    model.load_state_dict(model_state)
    
    print('Extra layers:', extra_layers)
    print('Different shape layers:', different_shape_layers)
    print('Need to train layers:', need_train_layers)
    return model

def add_mean_dim(atom_feature_list):
    new_feats = []
    for feat in atom_feature_list:
        feat_mean = feat.mean(dim=0, keepdim=True)
        new_feat = torch.cat([feat_mean, feat], dim=0)
        new_feats.append(new_feat)
    return new_feats

def make_proj(in_dim, out_dim):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

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
    
class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)
        
class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden
    
class EvaluationDropout(nn.Dropout):
    def __init__(self, *args, **kwargs):
        super(EvaluationDropout, self).__init__(*args, **kwargs)
        self.inference_mode = False

    def set_inference_mode(self, val : bool):
        self.inference_mode = val

    def forward(self, input):
        if self.inference_mode:
            return nn.functional.dropout(input, p = 0)
        else:
            return nn.functional.dropout(input, p = self.p)

class AggregateLayer(nn.Module):
    def __init__(self, dropout=0.1, hidden_size=512, d_model=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dp  = nn.Dropout(dropout)

    def forward(self, context, mask):
        """
        context: [B, L, D]
        mask   : [B, L] (1=valid, 0=pad)
        """
        B, L, D = context.shape
        logits = self.fc2(self.tanh(self.fc1(context)))          # [B, L, 1]
        mask   = mask.bool()
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        has_any = mask.any(dim=1)                                 # [B]
        attn    = torch.zeros_like(logits)                        # [B, L, 1]
        if has_any.any():
            attn_valid = torch.softmax(logits[has_any], dim=1)    # [Bv, L, 1]
            attn[has_any] = attn_valid

        out = torch.bmm(context.transpose(1, 2), attn).squeeze(2) # [B, D]
        return self.dp(out)


class NodeAttn(nn.Module):
    def __init__(self, hidden_dim, dropout, num_heads=8):
        super(NodeAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y, x_mask, y_mask):
        """
        x -> y ==> y^
        """
        masked_x = x * x_mask.unsqueeze(-1)
        x_sum_mask = x_mask.sum(dim=1, keepdim=True).float()
        x_sum_mask = x_sum_mask.clamp(min=1e-6)
        mean_x = masked_x.sum(dim=1) / x_sum_mask

        q = self.q(mean_x)  # (B, D)
        k = self.k(y)       # (B, N, D)
        v = self.v(y)       # (B, N, D)

        q = q.view(q.shape[0], 1, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, num_heads, 1, head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, num_heads, N, head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, num_heads, N, head_dim)

        score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, 1, N)
        score = score - (1.0 - y_mask.unsqueeze(1).unsqueeze(2).float()) * 1e9  # (B, num_heads, 1, N)

        score = F.softmax(score, dim=-1)
        score = self.attn_dropout(score)

        attn_output = torch.matmul(score, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.shape[0], -1, self.num_heads * self.head_dim)  # (B, 1, num_heads * head_dim)

        out = attn_output.squeeze(1)
        out = self.out(out)
        return out
    
class NodeAttenAgg(nn.Module):
    def __init__(self, hidden_dim, att_layer, dropout):
        super().__init__()
        self.seq_m = nn.ModuleList(NodeAttn(hidden_dim, dropout) for _ in range(att_layer))

    def forward(self, x, y, x_mask, y_mask):
        out_list = []
        for layer in self.seq_m:
            out = layer(x, y, x_mask, y_mask)
            out_list.append(out)
        return torch.stack(out_list, dim=1)

class LayerAttn(nn.Module):
    def __init__(self, hidden_dim, dropout, num_heads=8):
        super(LayerAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, y):
        """
        x -> y ==> y^
        """
        mean_x = x.mean(dim=1)

        q = self.q(mean_x)  # (B, D)
        k = self.k(y)       # (B, N, D)
        v = self.v(y)       # (B, N, D)

        q = q.view(q.shape[0], 1, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, num_heads, 1, head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, num_heads, N, head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, num_heads, N, head_dim)

        score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, 1, N)
        score = F.softmax(score, dim=-1)
        score = self.attn_dropout(score)

        attn_output = torch.matmul(score, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.shape[0], -1, self.num_heads * self.head_dim)  # (B, 1, num_heads * head_dim)
        out = self.out(attn_output.squeeze(1))
        out = attn_output.squeeze(1)

        out = self.layer_norm(out)
        return out
    
class LayerAttenAgg(nn.Module):
    def __init__(self, hidden_dim, att_layer, dropout):
        super().__init__()
        self.seq_m = nn.ModuleList(LayerAttn(hidden_dim, dropout) for _ in range(att_layer))
        self.seq_o = nn.Sequential(
        nn.Linear(hidden_dim*att_layer, 4*hidden_dim*att_layer),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(4*hidden_dim*att_layer, hidden_dim)
        )

    def forward(self, x, y):
        return self.seq_o(torch.cat([m(x, y) for m in self.seq_m], dim=-1))

class FPNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        # self.lin3 = nn.Linear(out_dim, out_dim)
        # self.lin4 = nn.Linear(out_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))
        # x = x + F.relu(self.lin3(x))
        # x = x + F.relu(self.lin4(x))
        return x

class SiteAggregator(nn.Module):
    """
    对多突变位点的 token 先做轻量 self-attn，再做门控池化（学会“看哪个位点更关键”）。
    """
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.out = nn.Linear(d_model, d_model)
            
    def forward(self, site_tokens: torch.Tensor, site_mask: torch.Tensor):
        """
        site_tokens: [B, M, D]
        site_mask:   [B, M]  (1=有效, 0=padding)
        """
        if site_tokens.numel() == 0:
            return site_tokens.new_zeros(site_tokens.size(0), site_tokens.size(-1))

        # 检查哪些样本有至少1个有效位点
        has_any = site_mask.any(dim=1)  # [B]

        key_padding_mask = (~site_mask.bool())  # True=pad
        x = site_tokens.clone()

        # 只对有有效位点的样本跑 self-attn，其余返回0向量
        out = torch.zeros(site_tokens.size(0), site_tokens.size(2), device=site_tokens.device)

        if has_any.any():
            # MultiheadAttention expects [B, M, D]
            attn_out, _ = self.self_attn(
                x[has_any],
                x[has_any],
                x[has_any],
                key_padding_mask=key_padding_mask[has_any]
            )

            logits = self.gate(attn_out).squeeze(-1)                        # [Bv, M]
            logits = logits.masked_fill(key_padding_mask[has_any], float('-inf'))
            weights = torch.softmax(logits, dim=1)                          # [Bv, M]
            weights = weights.unsqueeze(-1)                                 # [Bv, M, 1]

            agg = (attn_out * weights).sum(dim=1)                           # [Bv, D]
            out[has_any] = self.out(agg)

        return out


class GatedFusion3(nn.Module):
    """
    将 (位点向量, 邻域向量, 全局向量) 以门控方式融合，而不是简单 concat。
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj_site   = nn.Linear(d_in, d_out)
        self.proj_neigh  = nn.Linear(d_in, d_out)
        self.proj_global = nn.Linear(d_in, d_out)
        self.gate = nn.Sequential(
            nn.Linear(d_out * 3, d_out),
            nn.GELU(),
            nn.Linear(d_out, 3)   # 对三路做权重
        )

    def forward(self, h_site, h_neigh, h_global):
        s = self.proj_site(h_site)
        n = self.proj_neigh(h_neigh)
        g = self.proj_global(h_global)
        cat = torch.cat([s, n, g], dim=-1)
        alpha = torch.softmax(self.gate(cat), dim=-1)  # [B, 3]
        fused = alpha[:, 0:1] * s + alpha[:, 1:2] * n + alpha[:, 2:3] * g
        return fused, alpha  # 返回融合向量与权重，便于可解释性



class KineticModel_GearNet(nn.Module):
    def __init__(self, args):
        super(KineticModel_GearNet, self).__init__()
        self.args = args
        # GearNet encoder
        self.enzyme_encoder = EnzymeFusionNetwork(input_dim=args.enzyme_input_dim)

        # cross attention
        self.interaction_net_1 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        self.interaction_net_2 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        # substrate module
        if args.use_atom_feature:
            self.atom_encoder = AtomEncoder(emb_dim=args.atom_feat_dim)
            self.atom_aggregator = AggregateLayer(d_model=args.atom_feat_dim, hidden_size=args.atom_feat_dim)

        self.use_fingerprint = args.use_fingerprint
        if self.use_fingerprint:
            self.fingerprint_model = FPNet(167, args.bridge_dim)

        # dimension reduce
        self.enzyme_bridge_model = MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_dims=[args.bridge_dim])
        self.mol_bridge_model = MultiLayerPerceptron(input_dim=args.unimol_dim, hidden_dims=[args.bridge_dim])

        # pocket module
        if args.use_pocket:
            self.pocket_encoder = PocketNetwork(input_dim=args.pocket_input_dim, hidden_dims=args.gearnet_hidden)
            self.aggragator_pocket = AggregateLayer(d_model=args.pocket_dim, hidden_size=args.pocket_dim)
            self.pocket_bridge_model =  MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.pocket_input_dim, hidden_dims=[args.pocket_dim])

        # delta module
        self.enable_mutate_pred = args.mutate_pred
        if self.enable_mutate_pred:

            self.max_rel_dist = getattr(args, "max_rel_dist", 64)  # 距离桶上限
            self.use_blosum = getattr(args, "use_blosum", False)   # 可选：用 BLOSUM
            self.mut_flag_emb = nn.Embedding(2, args.enzyme_input_dim)                # 0/1
            self.rel_dist_emb = nn.Embedding(self.max_rel_dist + 1, args.enzyme_input_dim)  # 0..64，64表示≥64）
            self.sub_pair_emb = nn.Embedding(20 * 20, args.enzyme_input_dim)

            if self.use_blosum:
                self.blosum_bins = getattr(args, "blosum_bins", 16)
                self.blosum_emb = nn.Embedding(self.blosum_bins, args.enzyme_input_dim)

            self._tau_raw = nn.Parameter(torch.tensor(1.5))

            self.mut_fuse = nn.Sequential(
                nn.LayerNorm(args.enzyme_input_dim * 3),  # 位点向量 + 邻域池化 + 全局池化
                nn.Linear(args.enzyme_input_dim * 3, args.bridge_dim),
                nn.GELU(),
            )

            self.site_agg = SiteAggregator(d_model=args.enzyme_input_dim, heads=4, dropout=args.dropout)
            self.site_fusion = GatedFusion3(d_in=args.enzyme_input_dim, d_out=args.bridge_dim)

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)
        # self.aggragator_gearnet = AggregateLayer(d_model=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_size=sum(args.gearnet_hidden)+args.enzyme_input_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # EC number embedding
        self.ec_embedding = nn.Embedding(
            num_embeddings=args.num_ec_classes, 
            embedding_dim=args.ec_emb_dim
        )

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + int(args.use_pocket) + 2 + int(self.use_fingerprint)
                    + int(self.enable_mutate_pred) + 1)  # 按你的 moe_channels 顺序数数
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "bridge_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  # 仅占位；forward 中按实际长度处理
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "esm": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "pocket": nn.Linear(args.pocket_dim, args.bridge_dim) if args.use_pocket else nn.Identity(),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "mut": make_proj(args.bridge_dim, args.bridge_dim),
            "ec": make_proj(args.ec_emb_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
        })

        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                 # 或 "per-channel"，取决于你想要的语义
                    expert_hidden=expert_hidden,
                    dropout=args.dropout,
                    gate_temperature=getattr(args, "moe_gate_tau", 1.0),
                    noisy_gate=getattr(args, "moe_noisy", False),
                    aux_loss_type=getattr(args, "moe_aux_type", "kl"),
                )
        elif args.moe_mode == 'per-channel':
            self.moe = KineticsMoEChannel(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    top_k=top_k,
                    dropout=args.dropout,
                )

        self.output_block = nn.Linear(args.bridge_dim, 4)
    
    def forward(self, input):
        moe_channels = []

        # Uni-Mol Embedding
        mol_embedding = input['mol_embedding']
        mol_mask = mol_embedding.sum(dim=2) != 0
        mol_mask = mol_mask.to(torch.int)

        unimol_raw = self.aggragator_unimol(mol_embedding, mol_mask)
        unimol_vec = self.to_bridge["unimol"](unimol_raw)
        moe_channels.append(unimol_vec)

        # Mol Embedding Compute
        if self.args.use_atom_feature:
            atom_feature = self.atom_encoder(input['mol_arr'])
            split_atom_feature = torch.split(atom_feature, input['atom_num'])
            split_atom_feature = add_mean_dim(split_atom_feature)
            padding_atom_feature = pad_sequence(split_atom_feature, batch_first=True, padding_value=0)
            atom_embedding = self.atom_aggregator(padding_atom_feature, mol_mask)
            moe_channels.append(self.to_bridge["atom"](atom_embedding)) 

        mol_embedding = self.mol_bridge_model(mol_embedding)

        # ESM
        esm_node_feature, esm_mask = pack_residue_feats(
            input["graph"], input['residue_embedding']
        )
        esm_raw = self.aggragator_esm(esm_node_feature, esm_mask)              # [B, enzyme_input_dim]
        esm_vec = self.to_bridge["esm"](esm_raw)                               # [B, bridge_dim]
        moe_channels.append(esm_vec)

        # GearNet
        enzyme_output = self.enzyme_encoder(input['graph'], input['residue_embedding'], input['enzyme_embedding'])
        gearnet_embedding, gearnet_mask = pack_residue_feats(
            input["graph"], enzyme_output["node_feature"]
        )
        # gearnet_embedding = self.aggragator_gearnet(gearnet_embedding, gearnet_mask)
        # moe_channels.append(gearnet_embedding)
        # moe_channels.append(enzyme_output["graph_feature"])

        protein_node_feature = self.enzyme_bridge_model(enzyme_output["node_feature"])     
        protein_node_feature, protein_mask = pack_residue_feats(
            input["graph"], protein_node_feature
        )

        if self.args.use_pocket:
            pocket_output = self.pocket_encoder(input['pocket'], input['pocket_embedding'])
            pocket_node_feature = self.pocket_bridge_model(pocket_output)
            pocket_node_feature, pocket_mask = pack_residue_feats(
                input["pocket"], pocket_node_feature
            )
            pocket_feature = self.aggragator_pocket(pocket_node_feature, pocket_mask)   # [B, pocket_dim]
            moe_channels.append(self.to_bridge["pocket"](pocket_feature)) 

        # 将蛋白质和小分子的token的表征计算cross-attn，再将蛋白质和小分子的信息拼接
        protein_node_feature_2 = self.interaction_net_1(
            src=protein_node_feature,
            tgt=mol_embedding,
            src_mask=protein_mask,
            tgt_mask=mol_mask,
        )

        mol_node_embedding_2 = self.interaction_net_2(
            src=mol_embedding,
            tgt=protein_node_feature,
            src_mask=mol_mask,
            tgt_mask=protein_mask,
        )  

        protein_feature = self.aggragator_pro(protein_node_feature_2, protein_mask)
        mol_feature = self.aggragator_mol(mol_node_embedding_2, mol_mask)
        protein_feature_2 = self.to_bridge["protein"](protein_feature)
        mol_feature_2 = self.to_bridge["ligand"](mol_feature)
        moe_channels.append(protein_feature_2) 
        moe_channels.append(mol_feature_2)     

        if self.args.use_fingerprint:
            mol_fingerprint = self.fingerprint_model(input['MACCSKeys'])
            moe_channels.append(self.to_bridge["fp"](mol_fingerprint))

        if self.enable_mutate_pred:
            protein1, protein1_mask = pack_residue_feats(input["graph"], input['residue_embedding'])
            protein2, _ = pack_residue_feats(input["graph"], input['compare_embedding'])
            delta_embedding = protein1 - protein2

            B, L, D = delta_embedding.shape
            device = delta_embedding.device

            pos = torch.arange(L, device=device)[None, :].expand(B, L)
            mut_idx = input["mut_idx"]            
            valid_mut = (mut_idx >= 0)            
            S = torch.zeros(B, L, dtype=torch.long, device=device)
            S.scatter_(1, mut_idx.clamp_min(0), valid_mut.long())
            S = S.bool()        # 突变位点为True, 其余False

            # 相对距离（到最近一个突变位点）
            big = torch.full((B, L), 10_000, device=device, dtype=torch.long)
            dist = big
            for j in range(mut_idx.size(1)):
                mj = mut_idx[:, j].clamp_min(0)                               # (B,)
                d = (pos - mj[:, None]).abs()
                d = torch.where(valid_mut[:, j][:, None], d, big)            # 无效位点忽略
                dist = torch.minimum(dist, d)
            max_bucket = self.max_rel_dist
            dist = dist.clamp(max=max_bucket)                                 # (B,L)

            S_f = S.float()  # [B,L]
            emb_flag = self.mut_flag_emb(torch.ones_like(S, dtype=torch.long)) * S_f.unsqueeze(-1)
            has_mut = valid_mut.any(dim=1, keepdim=True)  # [B,1]
            emb_reld = self.rel_dist_emb(dist) * has_mut.float().unsqueeze(-1)                               # (B,L,D)

            # 替换对嵌入只加在位点
            pair_emb = torch.zeros_like(delta_embedding)
            if "mut_from_aa_idx" in input:
                pair_id = input["mut_from_aa_idx"] * 20 + input["mut_to_aa_idx"]   # (B,M)
                for j in range(mut_idx.size(1)):
                    idxj  = mut_idx[:, j]                       # (B,)
                    maskj = valid_mut[:, j]                     # (B,)
                    # 先选出有效样本的 batch 索引 & 位点
                    b_idx = torch.arange(B, device=device)[maskj]     # (#v,)
                    posj  = idxj[maskj]                                # (#v,)
                    # 边界过滤
                    ok = (posj >= 0) & (posj < L)
                    if ok.any():
                        pij = self.sub_pair_emb(pair_id[:, j][maskj])[ok]   # (#ok, D)
                        pair_emb[b_idx[ok], posj[ok]] += pij

            delta_full = delta_embedding + emb_flag + emb_reld + pair_emb

            B, L, D = delta_full.shape
            idx_safe = mut_idx.clamp(min=0, max=L-1)                     # [B, M]

            site_tokens = torch.gather(                                  # gather 位点向量：从序列维度 L 上按位点索引取 token, 只取突变位点的embedding
                delta_full, dim=1,
                index=idx_safe.unsqueeze(-1).expand(-1, -1, D)           # [B, M, D]
            )
            site_tokens = site_tokens * valid_mut.unsqueeze(-1).float()     # 对无效位点位置置零

            # 位点级自注意力 + 门控池化
            eps = 1e-8
            h_site_attn = self.site_agg(site_tokens, valid_mut)          # [B, D]

            # 邻域权重（序列或 3D 距离；WT 时退化为全 0）
            tau = torch.nn.functional.softplus(self._tau_raw) + 1e-3
            w = torch.exp(-dist.float() / tau) * protein1_mask.float()                 # (B,L)
            w = w / (w.sum(1, keepdim=True) + eps)
            h_neigh = torch.bmm(w.unsqueeze(1), delta_full).squeeze(1)        # (B,D)

            # 全局池化
            m = protein1_mask.float()
            h_global = (delta_full * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + eps)

            h_fused, fusion_weights = self.site_fusion(h_site_attn, h_neigh, h_global)                                   # (B,1280)
            # mutate_pred = self.mutate_head(h_fused)        
            h_fused_2 = self.to_bridge["mut"](h_fused)                 
            moe_channels.append(h_fused_2)

            # query_protein = self.query_protein.expand(protein1.shape[0], -1, -1)
            # mutate_mask = protein1_mask * (delta_embedding.sum(dim=2) != 0).to(torch.int)
            # delta_embedding = self.pooler_protein(query_protein, delta_embedding, delta_embedding, (1-mutate_mask).bool())[0]
            # delta_embedding = torch.nan_to_num(delta_embedding, nan=0.0)

            # 1
            # delta_embedding = self.proj_protein(delta_embedding)
            # mutate_pred = self.mutate_pred(delta_embedding.mean(dim=1))

            #2 
            # mutate_pred = self.mutate_pred(delta_embedding.sum(dim=1))
            # moe_channels.append(delta_embedding.sum(dim=1))

            #3
            # mutate_embedding = self.proj_protein(delta_embedding.sum(dim=1))
            # mutate_pred = self.mutate_pred(mutate_embedding)
            # moe_channels.append(delta_embedding.sum(dim=1))

        # EC embedding
        ec_embedding = self.ec_embedding(input['EC_idx'].squeeze(-1))
        ec_embedding_2 = self.to_bridge["ec"](ec_embedding)
        moe_channels.append(ec_embedding_2)
    
        # MoE
        router_parts = [protein_feature, mol_feature]
        if h_fused is not None:
            router_parts.append(h_fused)
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
            topk_idx, gates = moe_gateinfo
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        # evidential regression learning 
        output = self.output_block(moe_out)
        min_val = 1e-4

        # Split the outputs into the four distribution parameters
        means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
        lambdas = torch.nn.Softplus()(loglambdas) + min_val
        alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = torch.nn.Softplus()(logbetas) + min_val

        if self.enable_mutate_pred:
            return means, lambdas, alphas, betas, moe_aux_loss
        else:
            return means, lambdas, alphas, betas, moe_aux_loss

class KineticModel_GVP(nn.Module):
    def __init__(self, args):
        super(KineticModel_GVP, self).__init__()
        self.args = args
        # GVP encoder
        self.enzyme_encoder = EnzymeFusionNetwork(input_dim=args.enzyme_input_dim, structure_model='gvp')

        # cross attention
        self.interaction_net_1 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        self.interaction_net_2 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        # substrate module
        if args.use_atom_feature:
            self.atom_encoder = AtomEncoder(emb_dim=args.atom_feat_dim)
            self.atom_aggregator = AggregateLayer(d_model=args.atom_feat_dim, hidden_size=args.atom_feat_dim)

        self.use_fingerprint = args.use_fingerprint
        if self.use_fingerprint:
            self.fingerprint_model = FPNet(167, args.bridge_dim)

        # dimension reduce
        self.enzyme_bridge_model = MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden), hidden_dims=[args.bridge_dim])
        self.mol_bridge_model = MultiLayerPerceptron(input_dim=args.unimol_dim, hidden_dims=[args.bridge_dim])

        # pocket module
        if args.use_pocket:
            self.pocket_encoder = PocketNetwork(input_dim=args.pocket_input_dim, hidden_dims=args.gearnet_hidden)
            self.aggragator_pocket = AggregateLayer(d_model=args.pocket_dim, hidden_size=args.pocket_dim)
            self.pocket_bridge_model =  MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.pocket_input_dim, hidden_dims=[args.pocket_dim])

        # delta module
        self.enable_mutate_pred = args.mutate_pred
        if self.enable_mutate_pred:

            self.max_rel_dist = getattr(args, "max_rel_dist", 64)  # 距离桶上限
            self.use_blosum = getattr(args, "use_blosum", False)   # 可选：用 BLOSUM
            self.mut_flag_emb = nn.Embedding(2, args.enzyme_input_dim)                # 0/1
            self.rel_dist_emb = nn.Embedding(self.max_rel_dist + 1, args.enzyme_input_dim)  # 0..64，64表示≥64）
            self.sub_pair_emb = nn.Embedding(20 * 20, args.enzyme_input_dim)

            if self.use_blosum:
                self.blosum_bins = getattr(args, "blosum_bins", 16)
                self.blosum_emb = nn.Embedding(self.blosum_bins, args.enzyme_input_dim)

            self._tau_raw = nn.Parameter(torch.tensor(1.5))

            self.mut_fuse = nn.Sequential(
                nn.LayerNorm(args.enzyme_input_dim * 3),  # 位点向量 + 邻域池化 + 全局池化
                nn.Linear(args.enzyme_input_dim * 3, args.bridge_dim),
                nn.GELU(),
            )

            self.site_agg = SiteAggregator(d_model=args.enzyme_input_dim, heads=4, dropout=args.dropout)
            self.site_fusion = GatedFusion3(d_in=args.enzyme_input_dim, d_out=args.bridge_dim)

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)
        # self.aggragator_gearnet = AggregateLayer(d_model=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_size=sum(args.gearnet_hidden)+args.enzyme_input_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # EC number embedding
        self.ec_embedding = nn.Embedding(
            num_embeddings=args.num_ec_classes, 
            embedding_dim=args.ec_emb_dim
        )

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + int(args.use_pocket) + 2 + int(self.use_fingerprint)
                    + int(self.enable_mutate_pred) + 1)  # 按你的 moe_channels 顺序数数
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "router_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  # 仅占位；forward 中按实际长度处理
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "esm": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "pocket": nn.Linear(args.pocket_dim, args.bridge_dim) if args.use_pocket else nn.Identity(),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "mut": make_proj(args.bridge_dim, args.bridge_dim),
            "ec": make_proj(args.ec_emb_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
        })

        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                 # 或 "per-channel"，取决于你想要的语义
                    expert_hidden=expert_hidden,
                    dropout=args.dropout,
                    gate_temperature=getattr(args, "moe_gate_tau", 1.0),
                    noisy_gate=getattr(args, "moe_noisy", False),
                    aux_loss_type=getattr(args, "moe_aux_type", "kl"),
                )
        elif args.moe_mode == 'per-channel':
            self.moe = KineticsMoEChannel(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    top_k=top_k,
                    dropout=args.dropout,
                )

        self.output_block = nn.Linear(args.bridge_dim, 4)
    
    def forward(self, input):
        moe_channels = []

        # Uni-Mol Embedding
        mol_embedding = input['mol_embedding']
        mol_mask = mol_embedding.sum(dim=2) != 0
        mol_mask = mol_mask.to(torch.int)

        unimol_raw = self.aggragator_unimol(mol_embedding, mol_mask)
        unimol_vec = self.to_bridge["unimol"](unimol_raw)
        moe_channels.append(unimol_vec)

        # Mol Embedding Compute
        if self.args.use_atom_feature:
            atom_feature = self.atom_encoder(input['mol_arr'])
            split_atom_feature = torch.split(atom_feature, input['atom_num'])
            split_atom_feature = add_mean_dim(split_atom_feature)
            padding_atom_feature = pad_sequence(split_atom_feature, batch_first=True, padding_value=0)
            atom_embedding = self.atom_aggregator(padding_atom_feature, mol_mask)
            moe_channels.append(self.to_bridge["atom"](atom_embedding)) 

        mol_embedding = self.mol_bridge_model(mol_embedding)

        # ESM
        esm_node_feature, esm_mask = pack_residue_feats(
            input["graph"], input['residue_embedding']
        )
        esm_raw = self.aggragator_esm(esm_node_feature, esm_mask)              # [B, enzyme_input_dim]
        esm_vec = self.to_bridge["esm"](esm_raw)                               # [B, bridge_dim]
        moe_channels.append(esm_vec)

        # GearNet
        enzyme_output = self.enzyme_encoder(input['graph'], input['residue_embedding'], input['enzyme_embedding'])
        gearnet_embedding, gearnet_mask = pack_residue_feats(
            input["graph"], enzyme_output["node_feature"]
        )
        # gearnet_embedding = self.aggragator_gearnet(gearnet_embedding, gearnet_mask)
        # moe_channels.append(gearnet_embedding)
        # moe_channels.append(enzyme_output["graph_feature"])

        protein_node_feature = self.enzyme_bridge_model(enzyme_output["node_feature"])     
        protein_node_feature, protein_mask = pack_residue_feats(
            input["graph"], protein_node_feature
        )

        if self.args.use_pocket:
            pocket_output = self.pocket_encoder(input['pocket'], input['pocket_embedding'])
            pocket_node_feature = self.pocket_bridge_model(pocket_output)
            pocket_node_feature, pocket_mask = pack_residue_feats(
                input["pocket"], pocket_node_feature
            )
            pocket_feature = self.aggragator_pocket(pocket_node_feature, pocket_mask)   # [B, pocket_dim]
            moe_channels.append(self.to_bridge["pocket"](pocket_feature)) 

        # 将蛋白质和小分子的token的表征计算cross-attn，再将蛋白质和小分子的信息拼接
        protein_node_feature_2 = self.interaction_net_1(
            src=protein_node_feature,
            tgt=mol_embedding,
            src_mask=protein_mask,
            tgt_mask=mol_mask,
        )

        mol_node_embedding_2 = self.interaction_net_2(
            src=mol_embedding,
            tgt=protein_node_feature,
            src_mask=mol_mask,
            tgt_mask=protein_mask,
        )  

        protein_feature = self.aggragator_pro(protein_node_feature_2, protein_mask)
        mol_feature = self.aggragator_mol(mol_node_embedding_2, mol_mask)
        protein_feature_2 = self.to_bridge["protein"](protein_feature)
        mol_feature_2 = self.to_bridge["ligand"](mol_feature)
        moe_channels.append(protein_feature_2) 
        moe_channels.append(mol_feature_2)     

        if self.args.use_fingerprint:
            mol_fingerprint = self.fingerprint_model(input['MACCSKeys'])
            moe_channels.append(self.to_bridge["fp"](mol_fingerprint))

        if self.enable_mutate_pred:
            protein1, protein1_mask = pack_residue_feats(input["graph"], input['residue_embedding'])
            protein2, _ = pack_residue_feats(input["graph"], input['compare_embedding'])
            delta_embedding = protein1 - protein2

            B, L, D = delta_embedding.shape
            device = delta_embedding.device

            pos = torch.arange(L, device=device)[None, :].expand(B, L)
            mut_idx = input["mut_idx"]            
            valid_mut = (mut_idx >= 0)            
            S = torch.zeros(B, L, dtype=torch.long, device=device)
            S.scatter_(1, mut_idx.clamp_min(0), valid_mut.long())
            S = S.bool()        # 突变位点为True, 其余False

            # 相对距离（到最近一个突变位点）
            big = torch.full((B, L), 10_000, device=device, dtype=torch.long)
            dist = big
            for j in range(mut_idx.size(1)):
                mj = mut_idx[:, j].clamp_min(0)                               # (B,)
                d = (pos - mj[:, None]).abs()
                d = torch.where(valid_mut[:, j][:, None], d, big)            # 无效位点忽略
                dist = torch.minimum(dist, d)
            max_bucket = self.max_rel_dist
            dist = dist.clamp(max=max_bucket)                                 # (B,L)

            S_f = S.float()  # [B,L]
            emb_flag = self.mut_flag_emb(torch.ones_like(S, dtype=torch.long)) * S_f.unsqueeze(-1)
            has_mut = valid_mut.any(dim=1, keepdim=True)  # [B,1]
            emb_reld = self.rel_dist_emb(dist) * has_mut.float().unsqueeze(-1)                               # (B,L,D)

            # 替换对嵌入只加在位点
            pair_emb = torch.zeros_like(delta_embedding)
            if "mut_from_aa_idx" in input:
                pair_id = input["mut_from_aa_idx"] * 20 + input["mut_to_aa_idx"]   # (B,M)
                for j in range(mut_idx.size(1)):
                    idxj  = mut_idx[:, j]                       # (B,)
                    maskj = valid_mut[:, j]                     # (B,)
                    # 先选出有效样本的 batch 索引 & 位点
                    b_idx = torch.arange(B, device=device)[maskj]     # (#v,)
                    posj  = idxj[maskj]                                # (#v,)
                    # 边界过滤
                    ok = (posj >= 0) & (posj < L)
                    if ok.any():
                        pij = self.sub_pair_emb(pair_id[:, j][maskj])[ok]   # (#ok, D)
                        pair_emb[b_idx[ok], posj[ok]] += pij

            delta_full = delta_embedding + emb_flag + emb_reld + pair_emb

            B, L, D = delta_full.shape
            idx_safe = mut_idx.clamp(min=0, max=L-1)                     # [B, M]

            site_tokens = torch.gather(                                  # gather 位点向量：从序列维度 L 上按位点索引取 token, 只取突变位点的embedding
                delta_full, dim=1,
                index=idx_safe.unsqueeze(-1).expand(-1, -1, D)           # [B, M, D]
            )
            site_tokens = site_tokens * valid_mut.unsqueeze(-1).float()     # 对无效位点位置置零

            # 位点级自注意力 + 门控池化
            eps = 1e-8
            h_site_attn = self.site_agg(site_tokens, valid_mut)          # [B, D]

            # 邻域权重（序列或 3D 距离；WT 时退化为全 0）
            tau = torch.nn.functional.softplus(self._tau_raw) + 1e-3
            w = torch.exp(-dist.float() / tau) * protein1_mask.float()                 # (B,L)
            w = w / (w.sum(1, keepdim=True) + eps)
            h_neigh = torch.bmm(w.unsqueeze(1), delta_full).squeeze(1)        # (B,D)

            # 全局池化
            m = protein1_mask.float()
            h_global = (delta_full * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + eps)

            h_fused, fusion_weights = self.site_fusion(h_site_attn, h_neigh, h_global)                                   # (B,1280)
            # mutate_pred = self.mutate_head(h_fused)        
            h_fused_2 = self.to_bridge["mut"](h_fused)                 
            moe_channels.append(h_fused_2)

            # query_protein = self.query_protein.expand(protein1.shape[0], -1, -1)
            # mutate_mask = protein1_mask * (delta_embedding.sum(dim=2) != 0).to(torch.int)
            # delta_embedding = self.pooler_protein(query_protein, delta_embedding, delta_embedding, (1-mutate_mask).bool())[0]
            # delta_embedding = torch.nan_to_num(delta_embedding, nan=0.0)

            # 1
            # delta_embedding = self.proj_protein(delta_embedding)
            # mutate_pred = self.mutate_pred(delta_embedding.mean(dim=1))

            #2 
            # mutate_pred = self.mutate_pred(delta_embedding.sum(dim=1))
            # moe_channels.append(delta_embedding.sum(dim=1))

            #3
            # mutate_embedding = self.proj_protein(delta_embedding.sum(dim=1))
            # mutate_pred = self.mutate_pred(mutate_embedding)
            # moe_channels.append(delta_embedding.sum(dim=1))

        # EC embedding
        ec_embedding = self.ec_embedding(input['EC_idx'].squeeze(-1))
        ec_embedding_2 = self.to_bridge["ec"](ec_embedding)
        moe_channels.append(ec_embedding_2)
    
        # MoE
        router_parts = [protein_feature, mol_feature]
        if h_fused is not None:
            router_parts.append(h_fused)
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
            topk_idx, gates = moe_gateinfo
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        # evidential regression learning 
        output = self.output_block(moe_out)
        min_val = 1e-4

        # Split the outputs into the four distribution parameters
        means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
        lambdas = torch.nn.Softplus()(loglambdas) + min_val
        alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = torch.nn.Softplus()(logbetas) + min_val

        if self.enable_mutate_pred:
            return means, lambdas, alphas, betas, moe_aux_loss
        else:
            return means, lambdas, alphas, betas, moe_aux_loss

class KineticModel_CDConv(nn.Module):
    def __init__(self, args):
        super(KineticModel_CDConv, self).__init__()
        self.args = args
        # CDConv encoder
        self.enzyme_encoder = EnzymeFusionNetwork(input_dim=args.enzyme_input_dim, structure_model='cdconv')

        # cross attention
        self.interaction_net_1 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        self.interaction_net_2 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        # substrate module
        if args.use_atom_feature:
            self.atom_encoder = AtomEncoder(emb_dim=args.atom_feat_dim)
            self.atom_aggregator = AggregateLayer(d_model=args.atom_feat_dim, hidden_size=args.atom_feat_dim)

        self.use_fingerprint = args.use_fingerprint
        if self.use_fingerprint:
            self.fingerprint_model = FPNet(167, args.bridge_dim)

        # dimension reduce
        self.enzyme_bridge_model = MultiLayerPerceptron(input_dim=3072, hidden_dims=[args.bridge_dim])
        self.mol_bridge_model = MultiLayerPerceptron(input_dim=args.unimol_dim, hidden_dims=[args.bridge_dim])

        # pocket module
        if args.use_pocket:
            self.pocket_encoder = PocketNetwork(input_dim=args.pocket_input_dim, hidden_dims=args.gearnet_hidden)
            self.aggragator_pocket = AggregateLayer(d_model=args.pocket_dim, hidden_size=args.pocket_dim)
            self.pocket_bridge_model =  MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.pocket_input_dim, hidden_dims=[args.pocket_dim])

        # delta module
        self.enable_mutate_pred = args.mutate_pred
        if self.enable_mutate_pred:

            self.max_rel_dist = getattr(args, "max_rel_dist", 64)  # 距离桶上限
            self.use_blosum = getattr(args, "use_blosum", False)   # 可选：用 BLOSUM
            self.mut_flag_emb = nn.Embedding(2, args.enzyme_input_dim)                # 0/1
            self.rel_dist_emb = nn.Embedding(self.max_rel_dist + 1, args.enzyme_input_dim)  # 0..64，64表示≥64）
            self.sub_pair_emb = nn.Embedding(20 * 20, args.enzyme_input_dim)

            if self.use_blosum:
                self.blosum_bins = getattr(args, "blosum_bins", 16)
                self.blosum_emb = nn.Embedding(self.blosum_bins, args.enzyme_input_dim)

            self._tau_raw = nn.Parameter(torch.tensor(1.5))

            self.mut_fuse = nn.Sequential(
                nn.LayerNorm(args.enzyme_input_dim * 3),  # 位点向量 + 邻域池化 + 全局池化
                nn.Linear(args.enzyme_input_dim * 3, args.bridge_dim),
                nn.GELU(),
            )

            self.site_agg = SiteAggregator(d_model=args.enzyme_input_dim, heads=4, dropout=args.dropout)
            self.site_fusion = GatedFusion3(d_in=args.enzyme_input_dim, d_out=args.bridge_dim)

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)
        # self.aggragator_gearnet = AggregateLayer(d_model=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_size=sum(args.gearnet_hidden)+args.enzyme_input_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # EC number embedding
        self.ec_embedding = nn.Embedding(
            num_embeddings=args.num_ec_classes, 
            embedding_dim=args.ec_emb_dim
        )

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + int(args.use_pocket) + 2 + int(self.use_fingerprint)
                    + int(self.enable_mutate_pred) + 1)  # 按你的 moe_channels 顺序数数
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "router_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  # 仅占位；forward 中按实际长度处理
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "esm": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "pocket": nn.Linear(args.pocket_dim, args.bridge_dim) if args.use_pocket else nn.Identity(),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "mut": make_proj(args.bridge_dim, args.bridge_dim),
            "ec": make_proj(args.ec_emb_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
        })

        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                 # 或 "per-channel"，取决于你想要的语义
                    expert_hidden=expert_hidden,
                    dropout=args.dropout,
                    gate_temperature=getattr(args, "moe_gate_tau", 1.0),
                    noisy_gate=getattr(args, "moe_noisy", False),
                    aux_loss_type=getattr(args, "moe_aux_type", "kl"),
                )
        elif args.moe_mode == 'per-channel':
            self.moe = KineticsMoEChannel(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    top_k=top_k,
                    dropout=args.dropout,
                )

        self.output_block = nn.Linear(args.bridge_dim, 4)
    
    def forward(self, input):
        moe_channels = []

        # Uni-Mol Embedding
        mol_embedding = input['mol_embedding']
        mol_mask = mol_embedding.sum(dim=2) != 0
        mol_mask = mol_mask.to(torch.int)

        unimol_raw = self.aggragator_unimol(mol_embedding, mol_mask)
        unimol_vec = self.to_bridge["unimol"](unimol_raw)
        moe_channels.append(unimol_vec)

        # Mol Embedding Compute
        if self.args.use_atom_feature:
            atom_feature = self.atom_encoder(input['mol_arr'])
            split_atom_feature = torch.split(atom_feature, input['atom_num'])
            split_atom_feature = add_mean_dim(split_atom_feature)
            padding_atom_feature = pad_sequence(split_atom_feature, batch_first=True, padding_value=0)
            atom_embedding = self.atom_aggregator(padding_atom_feature, mol_mask)
            moe_channels.append(self.to_bridge["atom"](atom_embedding)) 

        mol_embedding = self.mol_bridge_model(mol_embedding)

        # ESM
        esm_node_feature, esm_mask = pack_residue_feats(
            input["graph"], input['residue_embedding']
        )
        esm_raw = self.aggragator_esm(esm_node_feature, esm_mask)              # [B, enzyme_input_dim]
        esm_vec = self.to_bridge["esm"](esm_raw)                               # [B, bridge_dim]
        moe_channels.append(esm_vec)

        # GearNet
        enzyme_output = self.enzyme_encoder(input['graph'], input['residue_embedding'], input['enzyme_embedding'])
        gearnet_embedding, gearnet_mask = pack_residue_feats(
            input["graph"], enzyme_output["node_feature"]
        )
        # gearnet_embedding = self.aggragator_gearnet(gearnet_embedding, gearnet_mask)
        # moe_channels.append(gearnet_embedding)
        # moe_channels.append(enzyme_output["graph_feature"])

        protein_node_feature = self.enzyme_bridge_model(enzyme_output["node_feature"])     
        protein_node_feature, protein_mask = pack_residue_feats(
            input["graph"], protein_node_feature
        )

        if self.args.use_pocket:
            pocket_output = self.pocket_encoder(input['pocket'], input['pocket_embedding'])
            pocket_node_feature = self.pocket_bridge_model(pocket_output)
            pocket_node_feature, pocket_mask = pack_residue_feats(
                input["pocket"], pocket_node_feature
            )
            pocket_feature = self.aggragator_pocket(pocket_node_feature, pocket_mask)   # [B, pocket_dim]
            moe_channels.append(self.to_bridge["pocket"](pocket_feature)) 

        # 将蛋白质和小分子的token的表征计算cross-attn，再将蛋白质和小分子的信息拼接
        protein_node_feature_2 = self.interaction_net_1(
            src=protein_node_feature,
            tgt=mol_embedding,
            src_mask=protein_mask,
            tgt_mask=mol_mask,
        )

        mol_node_embedding_2 = self.interaction_net_2(
            src=mol_embedding,
            tgt=protein_node_feature,
            src_mask=mol_mask,
            tgt_mask=protein_mask,
        )  

        protein_feature = self.aggragator_pro(protein_node_feature_2, protein_mask)
        mol_feature = self.aggragator_mol(mol_node_embedding_2, mol_mask)
        protein_feature_2 = self.to_bridge["protein"](protein_feature)
        mol_feature_2 = self.to_bridge["ligand"](mol_feature)
        moe_channels.append(protein_feature_2) 
        moe_channels.append(mol_feature_2)     

        if self.args.use_fingerprint:
            mol_fingerprint = self.fingerprint_model(input['MACCSKeys'])
            moe_channels.append(self.to_bridge["fp"](mol_fingerprint))

        if self.enable_mutate_pred:
            protein1, protein1_mask = pack_residue_feats(input["graph"], input['residue_embedding'])
            protein2, _ = pack_residue_feats(input["graph"], input['compare_embedding'])
            delta_embedding = protein1 - protein2

            B, L, D = delta_embedding.shape
            device = delta_embedding.device

            pos = torch.arange(L, device=device)[None, :].expand(B, L)
            mut_idx = input["mut_idx"]            
            valid_mut = (mut_idx >= 0)            
            S = torch.zeros(B, L, dtype=torch.long, device=device)
            S.scatter_(1, mut_idx.clamp_min(0), valid_mut.long())
            S = S.bool()        # 突变位点为True, 其余False

            # 相对距离（到最近一个突变位点）
            big = torch.full((B, L), 10_000, device=device, dtype=torch.long)
            dist = big
            for j in range(mut_idx.size(1)):
                mj = mut_idx[:, j].clamp_min(0)                               # (B,)
                d = (pos - mj[:, None]).abs()
                d = torch.where(valid_mut[:, j][:, None], d, big)            # 无效位点忽略
                dist = torch.minimum(dist, d)
            max_bucket = self.max_rel_dist
            dist = dist.clamp(max=max_bucket)                                 # (B,L)

            S_f = S.float()  # [B,L]
            emb_flag = self.mut_flag_emb(torch.ones_like(S, dtype=torch.long)) * S_f.unsqueeze(-1)
            has_mut = valid_mut.any(dim=1, keepdim=True)  # [B,1]
            emb_reld = self.rel_dist_emb(dist) * has_mut.float().unsqueeze(-1)                               # (B,L,D)

            # 替换对嵌入只加在位点
            pair_emb = torch.zeros_like(delta_embedding)
            if "mut_from_aa_idx" in input:
                pair_id = input["mut_from_aa_idx"] * 20 + input["mut_to_aa_idx"]   # (B,M)
                for j in range(mut_idx.size(1)):
                    idxj  = mut_idx[:, j]                       # (B,)
                    maskj = valid_mut[:, j]                     # (B,)
                    # 先选出有效样本的 batch 索引 & 位点
                    b_idx = torch.arange(B, device=device)[maskj]     # (#v,)
                    posj  = idxj[maskj]                                # (#v,)
                    # 边界过滤
                    ok = (posj >= 0) & (posj < L)
                    if ok.any():
                        pij = self.sub_pair_emb(pair_id[:, j][maskj])[ok]   # (#ok, D)
                        pair_emb[b_idx[ok], posj[ok]] += pij

            delta_full = delta_embedding + emb_flag + emb_reld + pair_emb

            B, L, D = delta_full.shape
            idx_safe = mut_idx.clamp(min=0, max=L-1)                     # [B, M]

            site_tokens = torch.gather(                                  # gather 位点向量：从序列维度 L 上按位点索引取 token, 只取突变位点的embedding
                delta_full, dim=1,
                index=idx_safe.unsqueeze(-1).expand(-1, -1, D)           # [B, M, D]
            )
            site_tokens = site_tokens * valid_mut.unsqueeze(-1).float()     # 对无效位点位置置零

            # 位点级自注意力 + 门控池化
            eps = 1e-8
            h_site_attn = self.site_agg(site_tokens, valid_mut)          # [B, D]

            # 邻域权重（序列或 3D 距离；WT 时退化为全 0）
            tau = torch.nn.functional.softplus(self._tau_raw) + 1e-3
            w = torch.exp(-dist.float() / tau) * protein1_mask.float()                 # (B,L)
            w = w / (w.sum(1, keepdim=True) + eps)
            h_neigh = torch.bmm(w.unsqueeze(1), delta_full).squeeze(1)        # (B,D)

            # 全局池化
            m = protein1_mask.float()
            h_global = (delta_full * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + eps)

            h_fused, fusion_weights = self.site_fusion(h_site_attn, h_neigh, h_global)                                   # (B,1280)
            # mutate_pred = self.mutate_head(h_fused)        
            h_fused_2 = self.to_bridge["mut"](h_fused)                 
            moe_channels.append(h_fused_2)

            # query_protein = self.query_protein.expand(protein1.shape[0], -1, -1)
            # mutate_mask = protein1_mask * (delta_embedding.sum(dim=2) != 0).to(torch.int)
            # delta_embedding = self.pooler_protein(query_protein, delta_embedding, delta_embedding, (1-mutate_mask).bool())[0]
            # delta_embedding = torch.nan_to_num(delta_embedding, nan=0.0)

            # 1
            # delta_embedding = self.proj_protein(delta_embedding)
            # mutate_pred = self.mutate_pred(delta_embedding.mean(dim=1))

            #2 
            # mutate_pred = self.mutate_pred(delta_embedding.sum(dim=1))
            # moe_channels.append(delta_embedding.sum(dim=1))

            #3
            # mutate_embedding = self.proj_protein(delta_embedding.sum(dim=1))
            # mutate_pred = self.mutate_pred(mutate_embedding)
            # moe_channels.append(delta_embedding.sum(dim=1))

        # EC embedding
        ec_embedding = self.ec_embedding(input['EC_idx'].squeeze(-1))
        ec_embedding_2 = self.to_bridge["ec"](ec_embedding)
        moe_channels.append(ec_embedding_2)
    
        # MoE
        router_parts = [protein_feature, mol_feature]
        if h_fused is not None:
            router_parts.append(h_fused)
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
            topk_idx, gates = moe_gateinfo
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        # evidential regression learning 
        output = self.output_block(moe_out)
        min_val = 1e-4

        # Split the outputs into the four distribution parameters
        means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
        lambdas = torch.nn.Softplus()(loglambdas) + min_val
        alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = torch.nn.Softplus()(logbetas) + min_val

        if self.enable_mutate_pred:
            return means, lambdas, alphas, betas, moe_aux_loss
        else:
            return means, lambdas, alphas, betas, moe_aux_loss

class KineticModel_CDConv_Condition(nn.Module):
    def __init__(self, args):
        super(KineticModel_CDConv_Condition, self).__init__()
        self.args = args
        # CDConv encoder
        self.enzyme_encoder = EnzymeFusionNetwork(input_dim=args.enzyme_input_dim, structure_model='cdconv')

        # cross attention
        self.interaction_net_1 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        self.interaction_net_2 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        # substrate module
        if args.use_atom_feature:
            self.atom_encoder = AtomEncoder(emb_dim=args.atom_feat_dim)
            self.atom_aggregator = AggregateLayer(d_model=args.atom_feat_dim, hidden_size=args.atom_feat_dim)

        self.use_fingerprint = args.use_fingerprint
        if self.use_fingerprint:
            self.fingerprint_model = FPNet(167, args.bridge_dim)

        # dimension reduce
        self.enzyme_bridge_model = MultiLayerPerceptron(input_dim=3072, hidden_dims=[args.bridge_dim])
        self.mol_bridge_model = MultiLayerPerceptron(input_dim=args.unimol_dim, hidden_dims=[args.bridge_dim])

        # pocket module
        if args.use_pocket:
            self.pocket_encoder = PocketNetwork(input_dim=args.pocket_input_dim, hidden_dims=args.gearnet_hidden)
            self.aggragator_pocket = AggregateLayer(d_model=args.pocket_dim, hidden_size=args.pocket_dim)
            self.pocket_bridge_model =  MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.pocket_input_dim, hidden_dims=[args.pocket_dim])

        # delta module
        self.enable_mutate_pred = args.mutate_pred
        if self.enable_mutate_pred:

            self.max_rel_dist = getattr(args, "max_rel_dist", 64)  # 距离桶上限
            self.use_blosum = getattr(args, "use_blosum", False)   # 可选：用 BLOSUM
            self.mut_flag_emb = nn.Embedding(2, args.enzyme_input_dim)                # 0/1
            self.rel_dist_emb = nn.Embedding(self.max_rel_dist + 1, args.enzyme_input_dim)  # 0..64，64表示≥64）
            self.sub_pair_emb = nn.Embedding(20 * 20, args.enzyme_input_dim)

            if self.use_blosum:
                self.blosum_bins = getattr(args, "blosum_bins", 16)
                self.blosum_emb = nn.Embedding(self.blosum_bins, args.enzyme_input_dim)

            self._tau_raw = nn.Parameter(torch.tensor(1.5))

            self.mut_fuse = nn.Sequential(
                nn.LayerNorm(args.enzyme_input_dim * 3),  # 位点向量 + 邻域池化 + 全局池化
                nn.Linear(args.enzyme_input_dim * 3, args.bridge_dim),
                nn.GELU(),
            )

            self.site_agg = SiteAggregator(d_model=args.enzyme_input_dim, heads=4, dropout=args.dropout)
            self.site_fusion = GatedFusion3(d_in=args.enzyme_input_dim, d_out=args.bridge_dim)

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)
        # self.aggragator_gearnet = AggregateLayer(d_model=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_size=sum(args.gearnet_hidden)+args.enzyme_input_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # EC number embedding
        self.ec_embedding = nn.Embedding(
            num_embeddings=args.num_ec_classes, 
            embedding_dim=args.ec_emb_dim
        )

        # Condition process
        self.ConditionNet = AutoDisEmbedding(num_dense_feature=2, num_channels=args.channels, embedding_dim=args.bridge_dim//2)

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + int(args.use_pocket) + 2 + int(self.use_fingerprint)
                    + int(self.enable_mutate_pred) + 1 + 1)  # 按你的 moe_channels 顺序数数
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "router_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  # 仅占位；forward 中按实际长度处理
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "esm": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "pocket": nn.Linear(args.pocket_dim, args.bridge_dim) if args.use_pocket else nn.Identity(),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "mut": make_proj(args.bridge_dim, args.bridge_dim),
            "ec": make_proj(args.ec_emb_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
            "Cond": make_proj(args.ec_emb_dim, args.bridge_dim),
        })

        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                 # 或 "per-channel"，取决于你想要的语义
                    expert_hidden=expert_hidden,
                    dropout=args.dropout,
                    gate_temperature=getattr(args, "moe_gate_tau", 1.0),
                    noisy_gate=getattr(args, "moe_noisy", False),
                    aux_loss_type=getattr(args, "moe_aux_type", "kl"),
                )
        elif args.moe_mode == 'per-channel':
            self.moe = KineticsMoEChannel(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    top_k=top_k,
                    dropout=args.dropout,
                )

        self.output_block = nn.Linear(args.bridge_dim, 4)
    
    def forward(self, input):
        moe_channels = []

        # Uni-Mol Embedding
        mol_embedding = input['mol_embedding']
        mol_mask = mol_embedding.sum(dim=2) != 0
        mol_mask = mol_mask.to(torch.int)

        unimol_raw = self.aggragator_unimol(mol_embedding, mol_mask)
        unimol_vec = self.to_bridge["unimol"](unimol_raw)
        moe_channels.append(unimol_vec)

        # Mol Embedding Compute
        if self.args.use_atom_feature:
            atom_feature = self.atom_encoder(input['mol_arr'])
            split_atom_feature = torch.split(atom_feature, input['atom_num'])
            split_atom_feature = add_mean_dim(split_atom_feature)
            padding_atom_feature = pad_sequence(split_atom_feature, batch_first=True, padding_value=0)
            atom_embedding = self.atom_aggregator(padding_atom_feature, mol_mask)
            moe_channels.append(self.to_bridge["atom"](atom_embedding)) 

        mol_embedding = self.mol_bridge_model(mol_embedding)

        # ESM
        esm_node_feature, esm_mask = pack_residue_feats(
            input["graph"], input['residue_embedding']
        )
        esm_raw = self.aggragator_esm(esm_node_feature, esm_mask)              # [B, enzyme_input_dim]
        esm_vec = self.to_bridge["esm"](esm_raw)                               # [B, bridge_dim]
        moe_channels.append(esm_vec)

        # GearNet
        enzyme_output = self.enzyme_encoder(input['graph'], input['residue_embedding'], input['enzyme_embedding'])
        gearnet_embedding, gearnet_mask = pack_residue_feats(
            input["graph"], enzyme_output["node_feature"]
        )
        # gearnet_embedding = self.aggragator_gearnet(gearnet_embedding, gearnet_mask)
        # moe_channels.append(gearnet_embedding)
        # moe_channels.append(enzyme_output["graph_feature"])

        protein_node_feature = self.enzyme_bridge_model(enzyme_output["node_feature"])     
        protein_node_feature, protein_mask = pack_residue_feats(
            input["graph"], protein_node_feature
        )

        if self.args.use_pocket:
            pocket_output = self.pocket_encoder(input['pocket'], input['pocket_embedding'])
            pocket_node_feature = self.pocket_bridge_model(pocket_output)
            pocket_node_feature, pocket_mask = pack_residue_feats(
                input["pocket"], pocket_node_feature
            )
            pocket_feature = self.aggragator_pocket(pocket_node_feature, pocket_mask)   # [B, pocket_dim]
            moe_channels.append(self.to_bridge["pocket"](pocket_feature)) 

        # 将蛋白质和小分子的token的表征计算cross-attn，再将蛋白质和小分子的信息拼接
        protein_node_feature_2 = self.interaction_net_1(
            src=protein_node_feature,
            tgt=mol_embedding,
            src_mask=protein_mask,
            tgt_mask=mol_mask,
        )

        mol_node_embedding_2 = self.interaction_net_2(
            src=mol_embedding,
            tgt=protein_node_feature,
            src_mask=mol_mask,
            tgt_mask=protein_mask,
        )  

        protein_feature = self.aggragator_pro(protein_node_feature_2, protein_mask)
        mol_feature = self.aggragator_mol(mol_node_embedding_2, mol_mask)
        protein_feature_2 = self.to_bridge["protein"](protein_feature)
        mol_feature_2 = self.to_bridge["ligand"](mol_feature)
        moe_channels.append(protein_feature_2) 
        moe_channels.append(mol_feature_2)     

        if self.args.use_fingerprint:
            mol_fingerprint = self.fingerprint_model(input['MACCSKeys'])
            moe_channels.append(self.to_bridge["fp"](mol_fingerprint))

        if self.enable_mutate_pred:
            protein1, protein1_mask = pack_residue_feats(input["graph"], input['residue_embedding'])
            protein2, _ = pack_residue_feats(input["graph"], input['compare_embedding'])
            delta_embedding = protein1 - protein2

            B, L, D = delta_embedding.shape
            device = delta_embedding.device

            pos = torch.arange(L, device=device)[None, :].expand(B, L)
            mut_idx = input["mut_idx"]            
            valid_mut = (mut_idx >= 0)            
            S = torch.zeros(B, L, dtype=torch.long, device=device)
            S.scatter_(1, mut_idx.clamp_min(0), valid_mut.long())
            S = S.bool()        # 突变位点为True, 其余False

            # 相对距离（到最近一个突变位点）
            big = torch.full((B, L), 10_000, device=device, dtype=torch.long)
            dist = big
            for j in range(mut_idx.size(1)):
                mj = mut_idx[:, j].clamp_min(0)                               # (B,)
                d = (pos - mj[:, None]).abs()
                d = torch.where(valid_mut[:, j][:, None], d, big)            # 无效位点忽略
                dist = torch.minimum(dist, d)
            max_bucket = self.max_rel_dist
            dist = dist.clamp(max=max_bucket)                                 # (B,L)

            S_f = S.float()  # [B,L]
            emb_flag = self.mut_flag_emb(torch.ones_like(S, dtype=torch.long)) * S_f.unsqueeze(-1)
            has_mut = valid_mut.any(dim=1, keepdim=True)  # [B,1]
            emb_reld = self.rel_dist_emb(dist) * has_mut.float().unsqueeze(-1)                               # (B,L,D)

            # 替换对嵌入只加在位点
            pair_emb = torch.zeros_like(delta_embedding)
            if "mut_from_aa_idx" in input:
                pair_id = input["mut_from_aa_idx"] * 20 + input["mut_to_aa_idx"]   # (B,M)
                for j in range(mut_idx.size(1)):
                    idxj  = mut_idx[:, j]                       # (B,)
                    maskj = valid_mut[:, j]                     # (B,)
                    # 先选出有效样本的 batch 索引 & 位点
                    b_idx = torch.arange(B, device=device)[maskj]     # (#v,)
                    posj  = idxj[maskj]                                # (#v,)
                    # 边界过滤
                    ok = (posj >= 0) & (posj < L)
                    if ok.any():
                        pij = self.sub_pair_emb(pair_id[:, j][maskj])[ok]   # (#ok, D)
                        pair_emb[b_idx[ok], posj[ok]] += pij

            delta_full = delta_embedding + emb_flag + emb_reld + pair_emb

            B, L, D = delta_full.shape
            idx_safe = mut_idx.clamp(min=0, max=L-1)                     # [B, M]

            site_tokens = torch.gather(                                  # gather 位点向量：从序列维度 L 上按位点索引取 token, 只取突变位点的embedding
                delta_full, dim=1,
                index=idx_safe.unsqueeze(-1).expand(-1, -1, D)           # [B, M, D]
            )
            site_tokens = site_tokens * valid_mut.unsqueeze(-1).float()     # 对无效位点位置置零

            # 位点级自注意力 + 门控池化
            eps = 1e-8
            h_site_attn = self.site_agg(site_tokens, valid_mut)          # [B, D]

            # 邻域权重（序列或 3D 距离；WT 时退化为全 0）
            tau = torch.nn.functional.softplus(self._tau_raw) + 1e-3
            w = torch.exp(-dist.float() / tau) * protein1_mask.float()                 # (B,L)
            w = w / (w.sum(1, keepdim=True) + eps)
            h_neigh = torch.bmm(w.unsqueeze(1), delta_full).squeeze(1)        # (B,D)

            # 全局池化
            m = protein1_mask.float()
            h_global = (delta_full * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + eps)

            h_fused, fusion_weights = self.site_fusion(h_site_attn, h_neigh, h_global)                                   # (B,1280)
            # mutate_pred = self.mutate_head(h_fused)        
            h_fused_2 = self.to_bridge["mut"](h_fused)                 
            moe_channels.append(h_fused_2)

            # query_protein = self.query_protein.expand(protein1.shape[0], -1, -1)
            # mutate_mask = protein1_mask * (delta_embedding.sum(dim=2) != 0).to(torch.int)
            # delta_embedding = self.pooler_protein(query_protein, delta_embedding, delta_embedding, (1-mutate_mask).bool())[0]
            # delta_embedding = torch.nan_to_num(delta_embedding, nan=0.0)

            # 1
            # delta_embedding = self.proj_protein(delta_embedding)
            # mutate_pred = self.mutate_pred(delta_embedding.mean(dim=1))

            #2 
            # mutate_pred = self.mutate_pred(delta_embedding.sum(dim=1))
            # moe_channels.append(delta_embedding.sum(dim=1))

            #3
            # mutate_embedding = self.proj_protein(delta_embedding.sum(dim=1))
            # mutate_pred = self.mutate_pred(mutate_embedding)
            # moe_channels.append(delta_embedding.sum(dim=1))


        # Condition process
        condition_embedding = self.ConditionNet(input['Condition'])
        condition_embedding_2 = self.to_bridge["Cond"](condition_embedding)
        moe_channels.append(condition_embedding_2)
        
        # EC embedding
        ec_embedding = self.ec_embedding(input['EC_idx'].squeeze(-1))
        ec_embedding_2 = self.to_bridge["ec"](ec_embedding)
        moe_channels.append(ec_embedding_2)
    
        # MoE
        router_parts = [protein_feature, mol_feature]
        if h_fused is not None:
            router_parts.append(h_fused)
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
            topk_idx, gates = moe_gateinfo
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        # evidential regression learning 
        output = self.output_block(moe_out)
        min_val = 1e-4

        # Split the outputs into the four distribution parameters
        means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
        lambdas = torch.nn.Softplus()(loglambdas) + min_val
        alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = torch.nn.Softplus()(logbetas) + min_val

        if self.enable_mutate_pred:
            return means, lambdas, alphas, betas, moe_aux_loss
        else:
            return means, lambdas, alphas, betas, moe_aux_loss
        
class KineticModel_Condition(nn.Module):
    def __init__(self, args):
        super(KineticModel_Condition, self).__init__()
        self.args = args
        # GearNet encoder
        self.enzyme_encoder = EnzymeFusionNetwork(input_dim=args.enzyme_input_dim)

        # cross attention
        self.interaction_net_1 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        self.interaction_net_2 = GlobalMultiHeadAttention(
            args.bridge_dim,
            heads=args.num_heads,
            n_layers=args.num_cross_attn,
            cross_attn_h_rate=1,
            dropout=args.dropout,
        )

        # substrate module
        if args.use_atom_feature:
            self.atom_encoder = AtomEncoder(emb_dim=args.atom_feat_dim)
            self.atom_aggregator = AggregateLayer(d_model=args.atom_feat_dim, hidden_size=args.atom_feat_dim)

        self.use_fingerprint = args.use_fingerprint
        if self.use_fingerprint:
            self.fingerprint_model = FPNet(167, args.bridge_dim)

        # dimension reduce
        self.enzyme_bridge_model = MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_dims=[args.bridge_dim])
        self.mol_bridge_model = MultiLayerPerceptron(input_dim=args.unimol_dim, hidden_dims=[args.bridge_dim])

        # pocket module
        if args.use_pocket:
            self.pocket_encoder = PocketNetwork(input_dim=args.pocket_input_dim, hidden_dims=args.gearnet_hidden)
            self.aggragator_pocket = AggregateLayer(d_model=args.pocket_dim, hidden_size=args.pocket_dim)
            self.pocket_bridge_model =  MultiLayerPerceptron(input_dim=sum(args.gearnet_hidden)+args.pocket_input_dim, hidden_dims=[args.pocket_dim])

        # delta module
        self.enable_mutate_pred = args.mutate_pred
        if self.enable_mutate_pred:
            # self.query_protein = nn.Parameter(torch.zeros(1, args.num_query_tokens, args.enzyme_input_dim))
            # nn.init.normal_(self.query_protein, 0, 0.02)
            # self.pooler_protein = nn.MultiheadAttention(
            #     embed_dim=args.enzyme_input_dim,
            #     num_heads=8,
            #     batch_first=True
            # )
            # self.proj_protein = nn.Linear(args.enzyme_input_dim, args.bridge_dim)
            # mutate_output_dim = 3 if args.mutate_pred_target == 'classification' else 1
            # 1
            # self.mutate_pred = nn.Sequential(
            #     nn.Linear(args.bridge_dim, args.bridge_dim),
            #     nn.GELU(),
            #     nn.Dropout(p=0.1),
            #     nn.Linear(args.bridge_dim, mutate_output_dim),
            # )
            # MoE_dims.append(args.bridge_dim)

            # 2
            # self.mutate_head = nn.Sequential(
            #     nn.Linear(args.bridge_dim, args.bridge_dim),
            #     nn.GELU(),
            #     nn.Dropout(p=0.1),
            #     nn.Linear(args.bridge_dim, mutate_output_dim),
            # )

            # ===== NEW: mutation-aware embeddings =====
            self.max_rel_dist = getattr(args, "max_rel_dist", 64)  # 距离桶上限
            self.use_blosum = getattr(args, "use_blosum", False)   # 可选：用 BLOSUM
            self.mut_flag_emb = nn.Embedding(2, args.enzyme_input_dim)                # 0/1
            self.rel_dist_emb = nn.Embedding(self.max_rel_dist + 1, args.enzyme_input_dim)  # 0..64，64表示≥64）
            self.sub_pair_emb = nn.Embedding(20 * 20, args.enzyme_input_dim)

            if self.use_blosum:
                self.blosum_bins = getattr(args, "blosum_bins", 16)
                self.blosum_emb = nn.Embedding(self.blosum_bins, args.enzyme_input_dim)

            self._tau_raw = nn.Parameter(torch.tensor(1.5))

            self.mut_fuse = nn.Sequential(
                nn.LayerNorm(args.enzyme_input_dim * 3),  # 位点向量 + 邻域池化 + 全局池化
                nn.Linear(args.enzyme_input_dim * 3, args.bridge_dim),
                nn.GELU(),
            )

            self.site_agg = SiteAggregator(d_model=args.enzyme_input_dim, heads=4, dropout=args.dropout)
            self.site_fusion = GatedFusion3(d_in=args.enzyme_input_dim, d_out=args.bridge_dim)

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)
        # self.aggragator_gearnet = AggregateLayer(d_model=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_size=sum(args.gearnet_hidden)+args.enzyme_input_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # Condition process
        self.ConditionNet = AutoDisEmbedding(num_dense_feature=2, num_channels=args.channels, embedding_dim=args.bridge_dim//2)

        # EC number embedding
        self.ec_embedding = nn.Embedding(
            num_embeddings=args.num_ec_classes, 
            embedding_dim=args.ec_emb_dim
        )

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + int(args.use_pocket) + 2 + int(self.use_fingerprint)
                    + int(self.enable_mutate_pred) + 1 + 1)  # 按你的 moe_channels 顺序数数
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "router_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  # 仅占位；forward 中按实际长度处理
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "esm": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "pocket": nn.Linear(args.pocket_dim, args.bridge_dim) if args.use_pocket else nn.Identity(),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "mut": make_proj(args.bridge_dim, args.bridge_dim),
            "ec": make_proj(args.ec_emb_dim, args.bridge_dim),
            "Cond": make_proj(args.ec_emb_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
        })

        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                 # 或 "per-channel"，取决于你想要的语义
                    expert_hidden=expert_hidden,
                    dropout=args.dropout,
                    gate_temperature=getattr(args, "moe_gate_tau", 1.0),
                    noisy_gate=getattr(args, "moe_noisy", False),
                    aux_loss_type=getattr(args, "moe_aux_type", "kl"),
                )
        elif args.moe_mode == 'per-channel':
            self.moe = KineticsMoEChannel(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    top_k=top_k,
                    dropout=args.dropout,
                )

        self.output_block = nn.Linear(args.bridge_dim, 4)
    
    def forward(self, input):
        moe_channels = []

        # Uni-Mol Embedding
        mol_embedding = input['mol_embedding']
        mol_mask = mol_embedding.sum(dim=2) != 0
        mol_mask = mol_mask.to(torch.int)

        unimol_raw = self.aggragator_unimol(mol_embedding, mol_mask)
        unimol_vec = self.to_bridge["unimol"](unimol_raw)
        moe_channels.append(unimol_vec)

        # Mol Embedding Compute
        if self.args.use_atom_feature:
            atom_feature = self.atom_encoder(input['mol_arr'])
            split_atom_feature = torch.split(atom_feature, input['atom_num'])
            split_atom_feature = add_mean_dim(split_atom_feature)
            padding_atom_feature = pad_sequence(split_atom_feature, batch_first=True, padding_value=0)
            atom_embedding = self.atom_aggregator(padding_atom_feature, mol_mask)
            moe_channels.append(self.to_bridge["atom"](atom_embedding)) 

        mol_embedding = self.mol_bridge_model(mol_embedding)

        # ESM
        esm_node_feature, esm_mask = pack_residue_feats(
            input["graph"], input['residue_embedding']
        )
        esm_raw = self.aggragator_esm(esm_node_feature, esm_mask)              # [B, enzyme_input_dim]
        esm_vec = self.to_bridge["esm"](esm_raw)                               # [B, bridge_dim]
        moe_channels.append(esm_vec)

        # GearNet
        enzyme_output = self.enzyme_encoder(input['graph'], input['residue_embedding'], input['enzyme_embedding'])
        gearnet_embedding, gearnet_mask = pack_residue_feats(
            input["graph"], enzyme_output["node_feature"]
        )
        # gearnet_embedding = self.aggragator_gearnet(gearnet_embedding, gearnet_mask)
        # moe_channels.append(gearnet_embedding)
        # moe_channels.append(enzyme_output["graph_feature"])

        protein_node_feature = self.enzyme_bridge_model(enzyme_output["node_feature"])     
        protein_node_feature, protein_mask = pack_residue_feats(
            input["graph"], protein_node_feature
        )

        if self.args.use_pocket:
            pocket_output = self.pocket_encoder(input['pocket'], input['pocket_embedding'])
            pocket_node_feature = self.pocket_bridge_model(pocket_output)
            pocket_node_feature, pocket_mask = pack_residue_feats(
                input["pocket"], pocket_node_feature
            )
            pocket_feature = self.aggragator_pocket(pocket_node_feature, pocket_mask)   # [B, pocket_dim]
            moe_channels.append(self.to_bridge["pocket"](pocket_feature)) 

        # 将蛋白质和小分子的token的表征计算cross-attn，再将蛋白质和小分子的信息拼接
        protein_node_feature_2 = self.interaction_net_1(
            src=protein_node_feature,
            tgt=mol_embedding,
            src_mask=protein_mask,
            tgt_mask=mol_mask,
        )

        mol_node_embedding_2 = self.interaction_net_2(
            src=mol_embedding,
            tgt=protein_node_feature,
            src_mask=mol_mask,
            tgt_mask=protein_mask,
        )  

        protein_feature = self.aggragator_pro(protein_node_feature_2, protein_mask)
        mol_feature = self.aggragator_mol(mol_node_embedding_2, mol_mask)
        protein_feature_2 = self.to_bridge["protein"](protein_feature)
        mol_feature_2 = self.to_bridge["ligand"](mol_feature)
        moe_channels.append(protein_feature_2) 
        moe_channels.append(mol_feature_2)     

        if self.args.use_fingerprint:
            mol_fingerprint = self.fingerprint_model(input['MACCSKeys'])
            moe_channels.append(self.to_bridge["fp"](mol_fingerprint))

        if self.enable_mutate_pred:
            protein1, protein1_mask = pack_residue_feats(input["graph"], input['residue_embedding'])
            protein2, _ = pack_residue_feats(input["graph"], input['compare_embedding'])
            delta_embedding = protein1 - protein2

            B, L, D = delta_embedding.shape
            device = delta_embedding.device

            pos = torch.arange(L, device=device)[None, :].expand(B, L)
            mut_idx = input["mut_idx"]            # (B,M), -1 表示无效位点
            valid_mut = (mut_idx >= 0)            # (B,M)
            S = torch.zeros(B, L, dtype=torch.long, device=device)
            S.scatter_(1, mut_idx.clamp_min(0), valid_mut.long())
            S = S.bool()        # 突变位点为True, 其余False

            # 相对距离（到最近一个突变位点）
            big = torch.full((B, L), 10_000, device=device, dtype=torch.long)
            dist = big
            for j in range(mut_idx.size(1)):
                mj = mut_idx[:, j].clamp_min(0)                               # (B,)
                d = (pos - mj[:, None]).abs()
                d = torch.where(valid_mut[:, j][:, None], d, big)            # 无效位点忽略
                dist = torch.minimum(dist, d)
            max_bucket = self.max_rel_dist
            dist = dist.clamp(max=max_bucket)                                 # (B,L)

            S_f = S.float()  # [B,L]
            emb_flag = self.mut_flag_emb(torch.ones_like(S, dtype=torch.long)) * S_f.unsqueeze(-1)
            has_mut = valid_mut.any(dim=1, keepdim=True)  # [B,1]
            emb_reld = self.rel_dist_emb(dist) * has_mut.float().unsqueeze(-1)                               # (B,L,D)

            # 替换对嵌入只加在位点
            pair_emb = torch.zeros_like(delta_embedding)
            if "mut_from_aa_idx" in input:
                pair_id = input["mut_from_aa_idx"] * 20 + input["mut_to_aa_idx"]   # (B,M)
                for j in range(mut_idx.size(1)):
                    idxj  = mut_idx[:, j]                       # (B,)
                    maskj = valid_mut[:, j]                     # (B,)
                    # 先选出有效样本的 batch 索引 & 位点
                    b_idx = torch.arange(B, device=device)[maskj]     # (#v,)
                    posj  = idxj[maskj]                                # (#v,)
                    # 边界过滤
                    ok = (posj >= 0) & (posj < L)
                    if ok.any():
                        pij = self.sub_pair_emb(pair_id[:, j][maskj])[ok]   # (#ok, D)
                        pair_emb[b_idx[ok], posj[ok]] += pij

            delta_full = delta_embedding + emb_flag + emb_reld + pair_emb

            B, L, D = delta_full.shape
            idx_safe = mut_idx.clamp(min=0, max=L-1)                     # [B, M]

            site_tokens = torch.gather(                                  # gather 位点向量：从序列维度 L 上按位点索引取 token
                delta_full, dim=1,
                index=idx_safe.unsqueeze(-1).expand(-1, -1, D)           # [B, M, D]
            )
            site_tokens = site_tokens * valid_mut.unsqueeze(-1).float()     # 对无效位点位置置零

            # 位点级自注意力 + 门控池化
            eps = 1e-8
            h_site_attn = self.site_agg(site_tokens, valid_mut)          # [B, D]

            # 邻域权重（序列或 3D 距离；WT 时退化为全 0）
            tau = torch.nn.functional.softplus(self._tau_raw) + 1e-3
            w = torch.exp(-dist.float() / tau) * protein1_mask.float()                 # (B,L)
            w = w / (w.sum(1, keepdim=True) + eps)
            h_neigh = torch.bmm(w.unsqueeze(1), delta_full).squeeze(1)        # (B,D)

            # 全局池化
            m = protein1_mask.float()
            h_global = (delta_full * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + eps)

            h_fused, fusion_weights = self.site_fusion(h_site_attn, h_neigh, h_global)                                   # (B,1280)
            # mutate_pred = self.mutate_head(h_fused)        
            h_fused_2 = self.to_bridge["mut"](h_fused)                 
            moe_channels.append(h_fused_2)

            # query_protein = self.query_protein.expand(protein1.shape[0], -1, -1)
            # mutate_mask = protein1_mask * (delta_embedding.sum(dim=2) != 0).to(torch.int)
            # delta_embedding = self.pooler_protein(query_protein, delta_embedding, delta_embedding, (1-mutate_mask).bool())[0]
            # delta_embedding = torch.nan_to_num(delta_embedding, nan=0.0)

            # 1
            # delta_embedding = self.proj_protein(delta_embedding)
            # mutate_pred = self.mutate_pred(delta_embedding.mean(dim=1))

            #2 
            # mutate_pred = self.mutate_pred(delta_embedding.sum(dim=1))
            # moe_channels.append(delta_embedding.sum(dim=1))

            #3
            # mutate_embedding = self.proj_protein(delta_embedding.sum(dim=1))
            # mutate_pred = self.mutate_pred(mutate_embedding)
            # moe_channels.append(delta_embedding.sum(dim=1))

        # Condition process
        condition_embedding = self.ConditionNet(input['Condition'])
        condition_embedding_2 = self.to_bridge["Cond"](condition_embedding)
        moe_channels.append(condition_embedding_2)

        # EC embedding
        ec_embedding = self.ec_embedding(input['EC_idx'].squeeze(-1))
        ec_embedding_2 = self.to_bridge["ec"](ec_embedding)
        moe_channels.append(ec_embedding_2)
    
        # MoE
        router_parts = [protein_feature, mol_feature]
        if h_fused is not None:
            router_parts.append(h_fused)
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        # evidential regression learning 
        output = self.output_block(moe_out)
        min_val = 1e-4

        # Split the outputs into the four distribution parameters
        means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
        lambdas = torch.nn.Softplus()(loglambdas) + min_val
        alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = torch.nn.Softplus()(logbetas) + min_val

        if self.enable_mutate_pred:
            return means, lambdas, alphas, betas, moe_aux_loss
        else:
            return means, lambdas, alphas, betas, moe_aux_loss

class AutoDisEmbedding(nn.Module):
    """An Embedding Learning Framework for Numerical Features in CTR Prediction.

    https://arxiv.org/pdf/2012.08986
    """

    def __init__(
        self,
        num_dense_feature: int,
        embedding_dim: int,
        num_channels: int,
        temperature: float = 0.1,
        keep_prob: float = 0.8,
        device = None,
    ) -> None:
        super().__init__()
        self.num_dense_feature = num_dense_feature
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.temperature = temperature

        self.meta_emb = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, embedding_dim, device=device)
        )

        # glorot normal initialization, std = sqrt(2 /(1+c))
        self.proj_w = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, device=device)
            * sqrt(2 / (1 + num_channels))
        )

        # glorot normal initialization, std = sqrt(2 /(c+c))
        self.proj_m = nn.Parameter(
            torch.randn(num_dense_feature, num_channels, num_channels, device=device)
            * sqrt(1 / num_channels)
        )
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dense_input):
        """Forward the module.

        Args:
            dense_input (Tensor): dense input feature, shape = [b, n],
                where b is batch_size, n is the number of dense features

        Returns:
            atde (Tensor): Tensor of autodis embedding.
        """
        hidden = self.leaky_relu(
            torch.einsum("nc,bn->bnc", self.proj_w, dense_input)
        )  # shape [b, n, c]
        x_bar = (
            torch.einsum("nij,bnj->bni", self.proj_m, hidden) + self.keep_prob * hidden
        )  # shape [b, n, c]
        x_hat = self.softmax(x_bar / self.temperature)  # shape = [b, n, c]
        emb = torch.einsum("ncd,bnc->bnd", self.meta_emb, x_hat)  # shape = [b, n, d]
        output = emb.reshape(
            (-1, self.num_dense_feature * self.embedding_dim)
        )  # shape = [b, n * d]
        return output
    