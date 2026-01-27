import torch
import torch.nn as nn
import math
from model.config import ModelConfig


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        q, k, v: (B, T, D)
        mask:    (T, T) or (B, T, T)
        return:  (B, T, D)
        """

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)

        return output

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # causal mask (registered as buffer, not parameter)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D)
        """
        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # apply causal mask
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)
    
class CachedCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x_t: torch.Tensor, kv_cache) -> torch.Tensor:
        """
        x_t: (B, 1, D)
        kv_cache: KVCache
        return: (B, 1, D)
        """
        B, T, D = x_t.shape
        assert T == 1, "Cached attention expects exactly one token"

        qkv = self.qkv_proj(x_t)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # append new K, V to cache
        kv_cache.append(k, v)

        # retrieve full cached K, V
        k_full = kv_cache.keys     # (B, H, T_total, Dh)
        v_full = kv_cache.values

        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v_full)

        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.out_proj(out)
