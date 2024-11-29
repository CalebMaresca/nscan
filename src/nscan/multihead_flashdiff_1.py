"""
This implementation is based on the Differential Transformer paper and code:
https://github.com/microsoft/unilm/tree/master/Diff-Transformer
Authors: Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei
License: MIT License (MIT)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

# from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        encoder_out=None,
        rel_pos=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        
        # For cross-attention, use encoder_out length as source length
        if encoder_out is not None:
            src_len = encoder_out.size(1)
            # Project decoder input (x) to queries
            q = self.q_proj(x)
            # Project encoder output to keys and values
            k = self.k_proj(encoder_out)
            v = self.v_proj(encoder_out)
        else:
            # Original self-attention case
            src_len = tgt_len
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, 2 * self.head_dim)

        # if rel_pos is not None:
        #     q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        #     k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        attn1 = flash_attn_func(q1, k1, v, causal=False)
        attn2 = flash_attn_func(q2, k2, v, causal=False)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn
