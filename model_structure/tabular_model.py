import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import typing as ty
import torch.nn.init as nn_init
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Sequence

from model_structure.FusionGearNet import EnzymeFusionNetwork
from model_structure.Attn_module import GlobalMultiHeadAttention
from model_structure.MoE import KineticsMoE, KineticsMoEChannel

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

feature_dims = [
            len(value) for key, value in allowable_features.items() # Atom_features_dim
        ] + [4] 

def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )


def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )

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

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="gelu", dropout=0):
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

class FPNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))

        return x


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        if max(x_cat[:,-2]) >3000:
            print()
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]     # 取 CLS
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

class KineticModel_Tabular_MultiTask(nn.Module):
    def __init__(self, args):
        super(KineticModel_Tabular_MultiTask, self).__init__()
        self.args = args
        # GearNet encoder
        self.enzyme_encoder = EnzymeFusionNetwork(input_dim=args.enzyme_input_dim)
        args.gearnet_hidden = [512,512,512]

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

        # delta module
        self.enable_mutate_pred = args.mutate_pred

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)
        self.aggragator_gearnet = AggregateLayer(d_model=sum(args.gearnet_hidden)+args.enzyme_input_dim, hidden_size=sum(args.gearnet_hidden)+args.enzyme_input_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + 2 + int(self.use_fingerprint) + 1
                    + int(self.enable_mutate_pred))  
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "bridge_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "protT5": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "tab": make_proj(args.bridge_dim, args.bridge_dim),
            "gearnet": make_proj(sum(args.gearnet_hidden)+args.enzyme_input_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
        })

        self.TabularNet = Transformer(d_numerical=args.numerical_features, categories=args.categorical_features, d_out=args.bridge_dim, token_bias=True, attention_dropout=args.dropout,
                                      d_token=args.bridge_dim, n_layers=3, n_heads=8, d_ffn_factor=2, ffn_dropout=args.dropout, residual_dropout=0.0, activation='reglu',
                                      initialization='kaiming', prenormalization=True, kv_compression=None, kv_compression_sharing=None)
        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                
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

        self.out_kcat = nn.Linear(args.bridge_dim, 4)
        self.out_km   = nn.Linear(args.bridge_dim, 4)
        self.out_eff  = nn.Linear(args.bridge_dim, 4)

    def _head_to_nig(self, out):
        min_val = 1e-5
        means, loglambdas, logalphas, logbetas = torch.chunk(out, 4, dim=-1)
        lambdas = F.softplus(loglambdas) + min_val
        alphas  = F.softplus(logalphas)  + min_val + 1.0  # alpha > 1
        betas   = F.softplus(logbetas)   + min_val
        return means.squeeze(-1), lambdas.squeeze(-1), alphas.squeeze(-1), betas.squeeze(-1)
    
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

        # ProtT5
        protT5_feature, residue_mask = pack_residue_feats(
            input["graph"], input['residue_embedding']
        )
        protT5_feat1 = self.aggragator_esm(protT5_feature, residue_mask)              # [B, enzyme_input_dim]
        protT5_vec = self.to_bridge["protT5"](protT5_feat1)                               # [B, bridge_dim]
        moe_channels.append(protT5_vec)

        # GearNet
        enzyme_output = self.enzyme_encoder(input['graph'], input['residue_embedding'], input['enzyme_embedding'])
        gearnet_embedding, _ = pack_residue_feats(
            input["graph"], enzyme_output["node_feature"]
        )
        gearnet_embedding = self.aggragator_gearnet(gearnet_embedding, residue_mask)
        gearnet_embedding = self.to_bridge["gearnet"](gearnet_embedding)
        moe_channels.append(gearnet_embedding)

        protein_node_feature = self.enzyme_bridge_model(enzyme_output["node_feature"])     
        protein_node_feature, _ = pack_residue_feats(
            input["graph"], protein_node_feature
        )

        protein_node_feature_2 = self.interaction_net_1(
            src=protein_node_feature,
            tgt=mol_embedding,
            src_mask=residue_mask,
            tgt_mask=mol_mask,
        )

        mol_node_embedding_2 = self.interaction_net_2(
            src=mol_embedding,
            tgt=protein_node_feature,
            src_mask=mol_mask,
            tgt_mask=residue_mask,
        )  

        protein_feature = self.aggragator_pro(protein_node_feature_2, residue_mask)
        mol_feature = self.aggragator_mol(mol_node_embedding_2, mol_mask)
        protein_feature_2 = self.to_bridge["protein"](protein_feature)
        mol_feature_2 = self.to_bridge["ligand"](mol_feature)
        moe_channels.append(protein_feature_2) 
        moe_channels.append(mol_feature_2)     

        if self.args.use_fingerprint:
            mol_fingerprint = self.fingerprint_model(input['MACCSKeys'])
            moe_channels.append(self.to_bridge["fp"](mol_fingerprint))

        # Tabular
        tabular_feat = self.TabularNet(input['num_feat'], input['cat_feat'])  # [B, bridge_dim]
        # copy_tab_feat = tabular_feat.clone()
        tabular_feat = self.to_bridge["tab"](tabular_feat)
        moe_channels.append(tabular_feat)
    
        # MoE
        router_parts = [protein_feature, mol_feature, tabular_feat]
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
            topk_idx, gates = moe_gateinfo
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        # Split the outputs into the four distribution parameters
        nig_k   = self._head_to_nig(self.out_kcat(moe_out))  # (μ_k, λ_k, α_k, β_k)
        nig_m   = self._head_to_nig(self.out_km(moe_out))
        nig_eff = self._head_to_nig(self.out_eff(moe_out))

        return nig_k, nig_m, nig_eff, moe_aux_loss,

class GearNet_Ablation_MultiTask(nn.Module):
    def __init__(self, args):
        super(GearNet_Ablation_MultiTask, self).__init__()
        self.args = args

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
        self.enzyme_bridge_model = MultiLayerPerceptron(input_dim=args.enzyme_input_dim, hidden_dims=[args.bridge_dim])
        self.mol_bridge_model = MultiLayerPerceptron(input_dim=args.unimol_dim, hidden_dims=[args.bridge_dim])

        # delta module
        self.enable_mutate_pred = args.mutate_pred

        # aggragator
        self.aggragator_esm = AggregateLayer(d_model=args.enzyme_input_dim, hidden_size=args.enzyme_input_dim)
        self.aggragator_unimol = AggregateLayer(d_model=args.unimol_dim, hidden_size=args.unimol_dim)

        self.aggragator_mol = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)
        self.aggragator_pro = AggregateLayer(d_model=args.bridge_dim, hidden_size=args.bridge_dim)

        # MoE config
        num_experts   = getattr(args, "moe_num_experts", 9)
        top_k         = getattr(args, "moe_top_k", 5)
        expert_hidden = getattr(args, "moe_hidden", None)

        n_channels = (1 + int(args.use_atom_feature) + 1 + 2 + int(self.use_fingerprint) 
                    + int(self.enable_mutate_pred)) 
        moe_input_dims = [args.bridge_dim] * n_channels

        router_dim = args.bridge_dim

        router_out_dim = getattr(args, "bridge_dim", 512)
        self.router_proj = nn.Sequential(
            nn.LayerNorm(args.bridge_dim * 3),  
            nn.Linear(args.bridge_dim * 3, router_out_dim),
            nn.GELU(),
        )

        self.to_bridge = nn.ModuleDict({
            "unimol": make_proj(args.unimol_dim, args.bridge_dim),
            "atom": make_proj(args.atom_feat_dim, args.bridge_dim) if args.use_atom_feature else nn.Identity(),
            "esm": make_proj(args.enzyme_input_dim, args.bridge_dim),
            "fp": make_proj(args.bridge_dim, args.bridge_dim),
            "tab": make_proj(args.bridge_dim, args.bridge_dim),
            "protein": nn.LayerNorm(args.bridge_dim),
            "ligand":  nn.LayerNorm(args.bridge_dim),
        })

        self.TabularNet = Transformer(d_numerical=args.numerical_features, categories=args.categorical_features, d_out=args.bridge_dim, token_bias=True, attention_dropout=args.dropout,
                                      d_token=args.bridge_dim, n_layers=3, n_heads=8, d_ffn_factor=2, ffn_dropout=args.dropout, residual_dropout=0.0, activation='reglu',
                                      initialization='kaiming', prenormalization=True, kv_compression=None, kv_compression_sharing=None)
        if args.moe_mode == 'concat':
            self.moe = KineticsMoE(
                    input_dims=moe_input_dims,
                    target_dim=args.bridge_dim,
                    router_dim=router_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    mode=getattr(args, "moe_mode", "concat"),                 
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

        self.loss_type = args.loss_type
        if self.loss_type == 'MSE':
            args.out_dim = 1
        else:
            args.out_dim = 4
            
        self.out_kcat = nn.Linear(args.bridge_dim, args.out_dim)
        self.out_km   = nn.Linear(args.bridge_dim, args.out_dim)
        self.out_eff  = nn.Linear(args.bridge_dim, args.out_dim)

    def _head_to_nig(self, out):
        min_val = 1e-5
        means, loglambdas, logalphas, logbetas = torch.chunk(out, 4, dim=-1)
        lambdas = F.softplus(loglambdas) + min_val
        alphas  = F.softplus(logalphas)  + min_val + 1.0  # alpha > 1
        betas   = F.softplus(logbetas)   + min_val
        return means.squeeze(-1), lambdas.squeeze(-1), alphas.squeeze(-1), betas.squeeze(-1)
    
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

        protein_node_feature = self.enzyme_bridge_model(esm_node_feature)     
        protein_mask = esm_mask

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

        # Tabular
        tabular_feat = self.TabularNet(input['num_feat'], input['cat_feat'])  # [B, bridge_dim]
        tabular_feat = self.to_bridge["tab"](tabular_feat)
        moe_channels.append(tabular_feat)
    
        # MoE
        router_parts = [protein_feature, mol_feature, tabular_feat]
        router_concat = torch.cat(router_parts, dim=-1)

        rc = router_concat
        router_feat = self.router_proj(rc)
        if self.args.moe_mode == 'concat':
            moe_out, moe_aux_loss, moe_gateinfo = self.moe(moe_channels, router_feat)
            topk_idx, gates = moe_gateinfo
        elif self.args.moe_mode == 'per-channel':
            moe_out, top_k_values = self.moe(moe_channels)
            moe_aux_loss = torch.std(top_k_values.sum(0))

        nig_k   = self._head_to_nig(self.out_kcat(moe_out))  # (μ_k, λ_k, α_k, β_k)
        nig_m   = self._head_to_nig(self.out_km(moe_out))
        nig_eff = self._head_to_nig(self.out_eff(moe_out))

        return nig_k, nig_m, nig_eff, moe_aux_loss