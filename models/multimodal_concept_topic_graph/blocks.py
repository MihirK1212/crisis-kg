import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGATLayer(nn.Module):
    """
    Relational Graph Attention layer implementing the equations provided:

    h_i' = sigma( sum_{r in T_E} sum_{j in N_i^r} alpha_{ij}^r W_r h_j )
    alpha_{ij}^r = softmax_j( LeakyReLU( a_r^T [ W_r h_i || W_r h_j ] ) )

    Here, r indexes relation types (we use two relations: concept-concept and concept-topic).

    Inputs:
    - node_feats: Tensor [N, d]
    - edges: Dict[str, List[Tuple[int, int]]], mapping relation name -> list of (src, dst)
    """

    def __init__(self, in_dim: int, out_dim: int, relations: List[str]) -> None:
        super().__init__()
        self.relations = relations
        self.rel_to_W = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=False) for r in relations})
        self.rel_to_a = nn.ParameterDict({r: nn.Parameter(torch.empty(2 * out_dim)) for r in relations})
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for W in self.rel_to_W.values():
            nn.init.xavier_uniform_(W.weight)
        for a in self.rel_to_a.values():
            nn.init.xavier_uniform_(a.unsqueeze(0))

    def forward(self, node_feats: torch.Tensor, edges: Dict[str, List[Tuple[int, int]]]) -> torch.Tensor:
        device = node_feats.device
        N, _ = node_feats.shape
        out = torch.zeros((N, list(self.rel_to_W.values())[0].out_features), device=device)

        # Precompute transformed features per relation
        transformed: Dict[str, torch.Tensor] = {
            r: self.rel_to_W[r](node_feats) for r in self.relations
        }

        for r in self.relations:
            rel_edges = edges.get(r, [])
            if len(rel_edges) == 0:
                continue

            # Build neighbor lists for each destination node
            dst_to_srcs: Dict[int, List[int]] = {}
            for src, dst in rel_edges:
                dst_to_srcs.setdefault(dst, []).append(src)

            WrH = transformed[r]
            a_r = self.rel_to_a[r]

            for dst, src_list in dst_to_srcs.items():
                h_i = WrH[dst]  # [d]
                h_js = WrH[src_list]  # [num_n, d]
                # Compute attention scores per neighbor
                # score_j = a^T [h_i || h_j]
                h_i_tiled = h_i.unsqueeze(0).expand(h_js.shape[0], -1)
                cat = torch.cat([h_i_tiled, h_js], dim=-1)  # [num_n, 2d]
                e_ij = self.leakyrelu(cat @ a_r)  # [num_n]
                alpha = torch.softmax(e_ij, dim=0)  # softmax over neighbors j
                # Aggregate
                out[dst] = out[dst] + torch.sum(alpha.unsqueeze(-1) * h_js, dim=0)

        # Residual: nodes with no incoming edges of any relation preserve their input features
        out = out + node_feats
        return out


class CAGM(nn.Module):
    """
    Concept-Anchored Graph Modulation (CAGM)

    Implements the latent reasoning stream (Section 3.3 of the paper):
    - Token-to-Concept affinity A between concat([FV, FL]) and concept prototypes P
    - Stage I: concept-grounded pooling -> concept-instance reps C_inst
    - Stage II: graph-biased concept self-attention with A_cc bias, residual + FFN
    - Affinity-based back-projection to token space -> F_lat_v, F_lat_l

    Inputs:
    - FV: [N_v, d]
    - FL: [N_l, d]
    - P: [C, d] (concept prototypes from explicit stream)
    - adj_cc: [C, C] (binary adjacency for concept->concept edges)

    Returns:
    - Flatent_V: [N_v, d]
    - Flatent_L: [N_l, d]
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d = d_model
        # Projection for prototypes in affinity computation
        self.W_proj = nn.Linear(d_model, d_model, bias=False)

        # Reasoning over concept-instance representations
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Bias scaling for graph-guided attention
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

        # Output projection after reasoning
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Lightweight FFN for concept representations
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        # Cached tensors for visualization (populated on forward in eval)
        self.last_A: torch.Tensor | None = None            # [N_v+N_l, C]
        self.last_Attn: torch.Tensor | None = None         # [C, C]
        self.last_counts: Tuple[int, int, int] | None = None  # (N_v, N_l, C)

    def forward(
        self,
        FV: torch.Tensor,
        FL: torch.Tensor,
        P: torch.Tensor,
        adj_cc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = FL.device
        N_v = FV.shape[0]
        N_l = FL.shape[0]
        d = self.d

        # If no concept prototypes, return zeros to keep shapes consistent
        if P.numel() == 0:
            Flatent_V = torch.zeros((N_v, d), device=device)
            Flatent_L = torch.zeros((N_l, d), device=device)
            # Update caches for downstream visualization
            self.last_A = None
            self.last_Attn = None
            self.last_counts = (N_v, N_l, 0)
            return Flatent_V, Flatent_L

        # Token-to-Concept Affinity
        T = torch.cat([FV, FL], dim=0)  # [N_v + N_l, d]
        Pp = self.W_proj(P)  # [C, d]
        logits = T @ Pp.transpose(0, 1)  # [N, C]
        A = torch.softmax(logits, dim=1)  # [N, C]

        # Stage I: Concept-Grounded Token Pooling
        # concept-instance reps: A^T T
        C_inst = A.transpose(0, 1) @ T  # [C, d]

        # Stage II: Graph-Guided Relational Reasoning with bias from adj_cc
        Q = self.W_q(C_inst)
        K = self.W_k(C_inst)
        V = self.W_v(C_inst)
        attn_scores = (Q @ K.transpose(0, 1)) / math.sqrt(d)
        if adj_cc.numel() > 0:
            # Ensure adjacency bias matches shape
            B = adj_cc.float()
            if B.shape != attn_scores.shape:
                B = F.pad(B, (0, attn_scores.shape[1] - B.shape[1], 0, attn_scores.shape[0] - B.shape[0]))
                B = B[: attn_scores.shape[0], : attn_scores.shape[1]]
            attn_scores = attn_scores + self.bias_scale * B
        Attn = torch.softmax(attn_scores, dim=-1)
        C_upd = Attn @ V  # [C, d]
        # Residual + FFN
        C_upd = self.layer_norm_1(C_inst + self.out_proj(C_upd))
        C_upd = self.layer_norm_2(C_upd + self.ffn(C_upd))

        # Project back to token space using affinity
        T_latent = A @ C_upd  # [N, d]
        Flatent_V = T_latent[:N_v]
        Flatent_L = T_latent[N_v:]

        # Cache attention matrices for external visualization (store on CPU)
        try:
            self.last_A = A.detach().to("cpu")
            self.last_Attn = Attn.detach().to("cpu")
            self.last_counts = (N_v, N_l, P.shape[0])
        except Exception as e:
            raise e
            # Best-effort caching; never break forward pass
            self.last_A = None
            self.last_Attn = None
            self.last_counts = (N_v, N_l, P.shape[0])
        return Flatent_V, Flatent_L


class AKF(nn.Module):
    """
    Adaptive Knowledge Fusion (AKF) with multi-view consensus gating.

    Implements the adaptive knowledge fusion stage (Section 3.4 of the paper):
    - Explicit knowledge summary: K_S_exp = cross-attn(Q=FL, K/V=Fexp)
    - Multi-view consensus gating: Sp (perceptual), Sexp (explicit), Slat (latent)
    - Gated cross-modal attention where text tokens attend to visual patches with M_gate

    Inputs:
    - FV: [N_v, d]
    - FL: [N_l, d]
    - Fexp: [|V|, d]
    - Flatent_V: [N_v, d]
    - Flatent_L: [N_l, d]

    Returns:
    - Ffused: [N_l, d]
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d = d_model

        # For KS_exp
        self.Wq_ksexp = nn.Linear(d_model, d_model, bias=False)
        self.Wk_ksexp = nn.Linear(d_model, d_model, bias=False)
        self.Wv_ksexp = nn.Linear(d_model, d_model, bias=False)

        # Projections for cross-attention (text queries over visual keys/values)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Triple-pathway gating projections
        self.Wp1 = nn.Linear(d_model, d_model, bias=False)
        self.Wp2 = nn.Linear(d_model, d_model, bias=False)
        self.We1 = nn.Linear(d_model, d_model, bias=False)
        self.We2 = nn.Linear(d_model, d_model, bias=False)
        self.Wl1 = nn.Linear(d_model, d_model, bias=False)
        self.Wl2 = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        FV: torch.Tensor,
        FL: torch.Tensor,
        Fexp: torch.Tensor,
        Flatent_V: torch.Tensor,
        Flatent_L: torch.Tensor,
    ) -> torch.Tensor:
        d = self.d

        # KS_exp: cross-attention from FL to Fexp
        Qe = self.Wq_ksexp(FL)  # [N_l, d]
        Ke = self.Wk_ksexp(Fexp)  # [|V|, d]
        Ve = self.Wv_ksexp(Fexp)  # [|V|, d]
        scores_e = (Qe @ Ke.transpose(0, 1)) / math.sqrt(d)
        Attn_e = torch.softmax(scores_e, dim=-1)  # [N_l, |V|]
        KS_exp = Attn_e @ Ve  # [N_l, d]

        # Projections for final attention
        QL = self.W_Q(FL)
        KV = self.W_K(FV)
        VV = self.W_V(FV)

        # Triple-pathway gating
        Sp = self.Wp1(FL) @ self.Wp2(FV).transpose(0, 1)  # [N_l, N_v]
        Sexp = self.We1(FL) @ self.We2(KS_exp).transpose(0, 1)  # [N_l, N_l]
        # Align Sexp to [N_l, N_v] by averaging over text dimension and expanding if needed
        Sexp_aligned = Sexp.mean(dim=1, keepdim=True).expand(-1, FV.shape[0])  # [N_l, N_v]
        Slat = self.Wl1(Flatent_L) @ self.Wl2(Flatent_V).transpose(0, 1)  # [N_l, N_v]
        Msynergy = torch.sigmoid(Sp + Sexp_aligned + Slat)  # [N_l, N_v]

        # Knowledge-modulated cross-attention (text queries attend to visual tokens)
        logits = (QL @ KV.transpose(0, 1)) / math.sqrt(d)
        logits = logits * Msynergy
        Attn = torch.softmax(logits, dim=-1)
        Ffused = Attn @ VV  # [N_l, d]
        return Ffused
