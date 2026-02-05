import torch
import torch.nn as nn

from model.config import ModelConfig
from model.block import TransformerBlock
from inference.cache import KVCache


class MiniGPTInferenceModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.reset_cache()

    def reset_cache(self):
        self.kv_caches = [KVCache() for _ in range(self.config.n_layers)]
        self.position = 0

    @torch.no_grad()
    def forward_step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, 1)
        returns logits: (B, vocab_size)
        """
        B, T = token_ids.shape
        assert T == 1, "Inference model expects exactly one token at a time"

        pos = torch.full((B, 1), self.position, device=token_ids.device)

        x = self.token_emb(token_ids) + self.pos_emb(pos)

        for block, cache in zip(self.blocks, self.kv_caches):
            x = block(x, cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        self.position += 1
        return logits.squeeze(1)
