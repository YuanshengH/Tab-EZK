import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Expert(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        h = hidden or max(out_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KineticsMoE(nn.Module):
    def __init__(
        self,
        input_dims: List[int],         
        target_dim: int,               
        router_dim: int,                
        num_experts: int,            
        top_k: int = 4,                
        mode: str = "concat",          
        expert_hidden: Optional[int] = None,
        dropout: float = 0.0,
        gate_temperature: float = 1.0,
        noisy_gate: bool = False,       
        aux_loss_type: str = "kl",      
    ):
        super().__init__()
        assert mode in ["concat", "per-channel"], "mode must be 'concat' or 'per-channel'"
        self.input_dims = input_dims
        self.target_dim = target_dim
        self.router_dim = router_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.mode = mode
        self.noisy_gate = noisy_gate
        self.gate_temperature = gate_temperature
        self.aux_loss_type = aux_loss_type

        if mode == "concat":
            in_dim = sum(input_dims)
            self.experts = nn.ModuleList([
                Expert(in_dim, target_dim, hidden=expert_hidden, dropout=dropout)
                for _ in range(num_experts)
            ])
        else:
            assert num_experts == len(input_dims), "per-channel 模式下 num_experts 必须等于通道数"
            self.experts = nn.ModuleList([
                Expert(input_dims[i], target_dim, hidden=expert_hidden, dropout=dropout)
                for i in range(num_experts)
            ])

        self.router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, max(256, router_dim)),
            nn.GELU(),
            nn.Linear(max(256, router_dim), num_experts)
        )

    @torch.no_grad()
    def _sample_noise(self, shape, device):
        u = torch.empty(shape, device=device).uniform_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(u))

    def _load_balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        E = probs.size(1)
        if self.aux_loss_type == "kl":
            target = torch.full_like(probs, 1.0 / E)
            loss = F.kl_div(probs.clamp_min(1e-8).log(), target, reduction="batchmean")
        else:
            loss = -(probs.clamp_min(1e-8).log() * probs).sum(dim=1).mean()
        return loss

    def forward(
        self,
        x_list: List[torch.Tensor],     
        router_feat: torch.Tensor,     
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = x_list[0].size(0)
        device = x_list[0].device

        logits = self.router(router_feat.float())  # [B, E]
        logits = logits / self.gate_temperature
        if self.training and self.noisy_gate:
            logits = logits + 0.5 * self._sample_noise(logits.shape, device)

        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [B,k], [B,k]
        gates = torch.softmax(topk_vals, dim=-1)                         # [B,k]

        probs_all = torch.softmax(logits, dim=-1)                        # [B,E]
        aux_loss = self._load_balance_loss(probs_all)

        y = torch.zeros(B, self.target_dim, device=device)

        if self.mode == "concat":
            xin = torch.cat(x_list, dim=-1)  # [B, sum(d)]
            for e in topk_idx.unique():      
                e = e.item()
                mask = (topk_idx == e).any(dim=1)  
                if mask.any():
                    ye = self.experts[e](xin[mask])         # [Be, target_dim]
                    gate_e = gates[mask][(topk_idx[mask] == e)].reshape(-1, 1)  # [Be,1]
                    y[mask] = y[mask] + gate_e * ye

        else:
            assert self.num_experts == len(x_list)
            for e in topk_idx.unique():
                e = e.item()
                mask = (topk_idx == e).any(dim=1)
                if mask.any():
                    xe = x_list[e][mask]                         # [Be, de]
                    ye = self.experts[e](xe)                     # [Be, target_dim]
                    gate_e = gates[mask][(topk_idx[mask] == e)].reshape(-1, 1)
                    y[mask] = y[mask] + gate_e * ye

        return y, aux_loss, (topk_idx, gates)

class KineticsMoEChannel(nn.Module):
    def __init__(self, input_dims, target_dim, top_k, dropout=0.0):
        super(KineticsMoEChannel, self).__init__()
        self.experts = []
        self.num_experts = len(input_dims)
        
        self.gating = nn.ModuleList([nn.Linear(input_dims[i], 1) for i in range(self.num_experts)])
        self.experts = nn.ModuleList([Expert(input_dims[i], target_dim) for i in range(self.num_experts)])

        # self.experts = nn.ModuleList([
        #     Expert(input_dims[i], target_dim, dropout=dropout)
        #     for i in range(len(input_dims))
        # ])

        self.target_dim = target_dim
        self.top_k = top_k
        
    def forward(self, x):
        batch_size = x[0].shape[0]
        indices = []
        for i in range(self.num_experts):
            indices.append(self.gating[i](x[i]))
        indices = torch.cat(indices, axis=1)
        top_k_values, top_k_indices = torch.topk(F.softmax(indices, dim=1), self.top_k)
        
        expert_outputs = torch.zeros(batch_size, self.top_k, self.target_dim).to(x[0].device)
        x_expanded = []
        for i in range(self.num_experts):
            x_expanded.append(x[i].unsqueeze(1).expand(-1, self.top_k, -1))  # [batch_size, input_dim] --> [batch_size, top_k, input_dim]
        expert_outputs = torch.zeros(batch_size, self.top_k, self.target_dim).to(x[0].device)

        for i in range(self.num_experts):
            mask = (top_k_indices == i).float().unsqueeze(-1)  # [batch_size, top_k, 1]
            selected_inputs = x_expanded[i] * mask  # [batch_size, top_k, input_dim]
            expert_outputs += self.experts[i](selected_inputs.view(-1, x[i].shape[1])).view(batch_size, self.top_k, self.target_dim) * mask

        gates_expanded = top_k_values.unsqueeze(-1).expand(-1, -1, self.target_dim)  # [batch_size, top_k, target_classes]

        x = (gates_expanded * expert_outputs).sum(1)
        
        return x, top_k_values 
    