import torch
import math
import torch.nn.functional as F
from torch import nn
from typing import Tuple

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, position_embedding_type="rotary"):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model, d_model))
        self.gating = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.position_embedding_type = position_embedding_type   
        if position_embedding_type == 'rotary':
            self.rotary_emb = RotaryEmbedding(d_model // heads)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # Use Xavier initialization for linear layers' weights
                nn.init.xavier_uniform_(p)
        # For the gating layer's weights, use Xavier uniform initialization
        nn.init.xavier_uniform_(self.gating.weight)
        # Bias initialization for the gating layer
        nn.init.constant_(self.gating.bias, 0.)
        # For 'to_out' layer's weights, use Xavier initialization as well
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.constant_(self.to_out.bias, 0.)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        bs = src.size(0)

        q = self.q_linear(src)
        k = self.k_linear(tgt)
        v = self.v_linear(tgt)

        k1 = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)   # shape(bs, h, Num, d_k)
        q1 = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v1 = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)

        if self.position_embedding_type == "rotary":
            q1, k1 = self.rotary_emb(q1, k1)

        attn1 = torch.matmul(q1, k1.permute(0, 1, 3, 2))
        attn = attn1 / math.sqrt(self.d_k)

        if (src_mask is not None) and (tgt_mask is None):
            src_mask = src_mask.bool()
            src_mask = src_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            src_mask = src_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn_mask = src_mask
            attn[~attn_mask] = float(-9e9)

        elif (src_mask is not None) and (tgt_mask is not None):
            src_mask = src_mask.bool()
            tgt_mask = tgt_mask.bool()
            attn_mask = tgt_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn[~attn_mask] = float(-9e9)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout1(attn)
        output = torch.matmul(attn, v1)

        output = output.transpose(1, 2).contiguous().view(
            bs, -1, self.d_model).squeeze(-1)
        
        # gate self attention
        output = self.to_out(output * self.gating(src).tanh())
        return output, attn, attn_mask

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
# TODO: self attention 和 cross attention 交叉进行
class GlobalMultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 cross_attn_h_rate=1,
                 dropout=0.1,
                 self_attn=True, 
                 cross_attn=True):
        super(GlobalMultiHeadAttentionLayer, self).__init__()

        self.cross_attn_h_rate = cross_attn_h_rate

        self.self_attn = MultiHeadAttention(heads, d_model,
                                            dropout, position_embedding_type=None)
        self.cross_attn = MultiHeadAttention(heads, d_model,
                                             dropout, position_embedding_type=None)
        
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.linear2 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = GELU()
        self.sa = self_attn
        self.ca = cross_attn


    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src先做self-attn，再和tgt做cross-attn
        if self.sa:
            src_m_s, att_score_s, attn_mask_s = self._sa_block(         # 改成post norm
                src, src_mask)
            src = self.norm1(src + src_m_s)

        if self.ca:
            src_m_c, att_score_c, attn_mask_c = self._cra_block(
                src, tgt, src_mask, tgt_mask)
            src = self.norm2(src * self.cross_attn_h_rate + src_m_c)  # self.cross_attn_h_rate 控制交叉注意力机制自身的比例///改成post norm
            src = self.norm3(src + self._ff_block(src))     # 改成post norm

        # pre norm
        # if self.sa:
        #     src_m_s, att_score_s, attn_mask_s = self._sa_block(       
        #         self.norm1(src), src_mask)
        #     src = src + src_m_s

        # if self.ca:
        #     src_m_c, att_score_c, attn_mask_c = self._cra_block(
        #         self.norm2(src), tgt, src_mask, tgt_mask)
        #     src = src * self.cross_attn_h_rate + src_m_c
        #     src = src + self._ff_block(self.norm3(src))    

        return src

    def _sa_block(self, x, mask):
        m, att_score, attn_mask = self.self_attn(x, x, mask)
        return self.dropout1(m), att_score, attn_mask

    def _cra_block(self, src, tgt, src_mask, tgt_mask):
        src_m, att_score, attn_mask = self.cross_attn(src, tgt, src_mask, tgt_mask)
        return self.dropout2(src_m), att_score, attn_mask

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
    

class GlobalMultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 n_layers=1,
                 cross_attn_h_rate=0.1,
                 dropout=0.1):
        super(GlobalMultiHeadAttention, self).__init__()
        self.n_layers = n_layers

        layer_stack = []

        for _ in range(n_layers):
            layer_stack.append(
                GlobalMultiHeadAttentionLayer(
                    d_model=d_model,
                    heads=heads,
                    cross_attn_h_rate=cross_attn_h_rate,
                    dropout=dropout,))

        self.layers = nn.ModuleList(layer_stack)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        for n in range(self.n_layers):
            src = self.layers[
                n](src, tgt, src_mask, tgt_mask)

        return src

    

