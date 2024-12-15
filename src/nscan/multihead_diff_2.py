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


class MultiheadDiff2(nn.Module):
    """
    DiffAttn implemented with standard attention
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
    
    def standard_attention(self, q, k, v):
        """Standard scaled dot-product attention"""
        # q, k, v shapes: (batch_size, num_heads, seq_len, head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_probs = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_probs, v)
    
    def forward(
        self,
        x,
        encoder_out=None,
        rel_pos=None,
    ):
        batch_size, tgt_len, embed_dim = x.size()
        
        # For cross-attention, use encoder_out length as source length
        if encoder_out is not None:
            src_len = encoder_out.size(1)
            # Project decoder input (x) to queries
            q = self.q_proj(x)          # (batch_size, tgt_len, embed_dim)
            # Project encoder output to keys and values
            k = self.k_proj(encoder_out)  # (batch_size, src_len, embed_dim)
            v = self.v_proj(encoder_out)  # (batch_size, src_len, embed_dim)
        else:
            # Original self-attention case
            src_len = tgt_len
            q = self.q_proj(x)          # (batch_size, tgt_len, embed_dim)
            k = self.k_proj(x)          # (batch_size, src_len, embed_dim)
            v = self.v_proj(x)          # (batch_size, src_len, embed_dim)

        # We manipulate the Tensors twice to allow for applying positional embeddings
        # Since we aren't using positional embeddings, we could simplify this
        # Note: embed_dim = num_heads * head_dim * 2
        q = q.view(batch_size, tgt_len, 2 * self.num_heads, self.head_dim)  # (batch_size, tgt_len, 2*num_heads, head_dim)
        k = k.view(batch_size, src_len, 2 * self.num_heads, self.head_dim)  # (batch_size, src_len, 2*num_heads, head_dim)
        v = v.view(batch_size, src_len, self.num_heads, 2, self.head_dim)  # (batch_size, src_len, num_heads, 2, head_dim)

        # if rel_pos is not None:
        #     q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        #     k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        # Reshape for splitting into two attention mechanisms
        q = q.reshape(batch_size, tgt_len, self.num_heads, 2, self.head_dim)  # (batch_size, tgt_len, num_heads, 2, head_dim)
        k = k.reshape(batch_size, src_len, self.num_heads, 2, self.head_dim)  # (batch_size, src_len, num_heads, 2, head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]  # Each: (batch_size, tgt_len, num_heads, head_dim)
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]  # Each: (batch_size, src_len, num_heads, head_dim)
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]  # Each: (batch_size, src_len, num_heads, head_dim)

        # Transpose for attention computation
        q1 = q1.transpose(1, 2)  # (batch_size, num_heads, tgt_len, head_dim)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)
        k2 = k2.transpose(1, 2)
        v1 = v1.transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)
        v2 = v2.transpose(1, 2)

        attn11 = self.standard_attention(q1, k1, v1)  # (batch_size, num_heads, tgt_len, head_dim)
        attn12 = self.standard_attention(q1, k1, v2)
        attn1 = torch.cat([attn11, attn12], dim=-1)  # (batch_size, num_heads, tgt_len, 2*head_dim)
        
        attn21 = self.standard_attention(q2, k2, v1)  # (batch_size, num_heads, tgt_len, head_dim)
        attn22 = self.standard_attention(q2, k2, v2)
        attn2 = torch.cat([attn21, attn22], dim=-1)  # (batch_size, num_heads, tgt_len, 2*head_dim)

        # Transpose back
        attn1 = attn1.transpose(1, 2)  # (batch_size, tgt_len, num_heads, 2*head_dim)
        attn2 = attn2.transpose(1, 2)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2  # (batch_size, tgt_len, num_heads, 2*head_dim)

        attn = self.subln(attn)  # (batch_size, tgt_len, num_heads, 2*head_dim)
        attn = attn * (1 - self.lambda_init)  # (batch_size, tgt_len, num_heads, 2*head_dim)
        attn = attn.reshape(batch_size, tgt_len, self.num_heads * 2 * self.head_dim)  # (batch_size, tgt_len, embed_dim = num_heads * 2 * head_dim)
        
        attn = self.out_proj(attn)  # (batch_size, tgt_len, embed_dim)
        return attn


class MultiheadFlashDiff2(nn.Module):
    """
    DiffAttn implemented with FlashAttention, for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
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
        batch_size, tgt_len, embed_dim = x.size()
        
        # For cross-attention, use encoder_out length as source length
        if encoder_out is not None:
            src_len = encoder_out.size(1)
            # Project decoder input (x) to queries
            q = self.q_proj(x)          # (batch_size, tgt_len, embed_dim)
            # Project encoder output to keys and values
            k = self.k_proj(encoder_out)  # (batch_size, src_len, embed_dim)
            v = self.v_proj(encoder_out)  # (batch_size, src_len, embed_dim)
        else:
            # Original self-attention case
            src_len = tgt_len
            q = self.q_proj(x)          # (batch_size, tgt_len, embed_dim)
            k = self.k_proj(x)          # (batch_size, src_len, embed_dim)
            v = self.v_proj(x)          # (batch_size, src_len, embed_dim)

        # We manipulate the Tensors twice to allow for applying positional embeddings
        # Since we aren't using positional embeddings, we could simplify this
        # Note: embed_dim = num_heads * head_dim * 2
        q = q.view(batch_size, tgt_len, 2 * self.num_heads, self.head_dim)  # (batch_size, tgt_len, 2*num_heads, head_dim)
        k = k.view(batch_size, src_len, 2 * self.num_heads, self.head_dim)  # (batch_size, src_len, 2*num_heads, head_dim)
        v = v.view(batch_size, src_len, self.num_heads, 2, self.head_dim)  # (batch_size, src_len, num_heads, 2, head_dim)

        # if rel_pos is not None:
        #     q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        #     k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        # Reshape for splitting into two attention mechanisms
        q = q.reshape(batch_size, tgt_len, self.num_heads, 2, self.head_dim)  # (batch_size, tgt_len, num_heads, 2, head_dim)
        k = k.reshape(batch_size, src_len, self.num_heads, 2, self.head_dim)  # (batch_size, src_len, num_heads, 2, head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]  # Each: (batch_size, tgt_len, num_heads, head_dim)
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]  # Each: (batch_size, src_len, num_heads, head_dim)
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]  # Each: (batch_size, src_len, num_heads, head_dim)

        attn11 = flash_attn_func(q1, k1, v1, causal=False)  # (batch_size, tgt_len, num_heads, head_dim)
        attn12 = flash_attn_func(q1, k1, v2, causal=False)  # (batch_size, tgt_len, num_heads, head_dim)
        attn1 = torch.cat([attn11, attn12], dim=-1)  # (batch_size, tgt_len, num_heads, 2 * head_dim)
        
        attn21 = flash_attn_func(q2, k2, v1, causal=False)  # (batch_size, tgt_len, num_heads, head_dim)
        attn22 = flash_attn_func(q2, k2, v2, causal=False)  # (batch_size, tgt_len, num_heads, head_dim)
        attn2 = torch.cat([attn21, attn22], dim=-1)  # (batch_size, tgt_len, num_heads, 2 * head_dim)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2  # (batch_size, tgt_len, num_heads, 2*head_dim)

        attn = self.subln(attn)  # (batch_size, tgt_len, num_heads, 2*head_dim)
        attn = attn * (1 - self.lambda_init)  # (batch_size, tgt_len, num_heads, 2*head_dim)
        attn = attn.reshape(batch_size, tgt_len, self.num_heads * 2 * self.head_dim)  # (batch_size, tgt_len, embed_dim = num_heads * 2 * head_dim)
        
        attn = self.out_proj(attn)  # (batch_size, tgt_len, embed_dim)
        return attn