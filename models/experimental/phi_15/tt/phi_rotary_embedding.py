import ttnn
import torch

from typing import Optional

from models.common.lightweightmodule import LightweightModule


class PhiRotaryEmbedding(LightweightModule):
    def __init__(
        self,
        device,
        config,
    ):
        self.device = device
        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta

        self.cos_cache, self.sin_cache = None, None

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device="cpu") / self.rotary_dim)
        )
        tt_inv_freq = ttnn.from_torch(
            inv_freq, dtype=ttnn.float32, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        self.inv_freq = ttnn.reshape(tt_inv_freq, [1, 1, 1, tt_inv_freq.shape[0]])

        t = ttnn.arange(end=self.max_position_embeddings, dtype=ttnn.float32, device=self.device)
        t = ttnn.reshape(t, [1, 1, t.shape[0], 1])
        freqs = ttnn.outer(t, self.inv_freq, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(t)

        emb = ttnn.concat([freqs, freqs], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.cos_cache = ttnn.cos(emb, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.sin_cache = ttnn.sin(emb, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.seq_len_cached = self.max_position_embeddings

        ttnn.deallocate(emb)

    def forward(self, x: ttnn.Tensor, token_idx: Optional[int] = None):
        batch_sz, num_heads, seq_len, rotary_dim = x.shape
        assert seq_len <= self.max_position_embeddings
        assert rotary_dim <= self.rotary_dim

        if seq_len > self.seq_len_cached:
            self._build_rope_cache(seq_len)

        return ttnn.experimental.rotary_embedding(
            x, self.cos_cache[:, :, :seq_len], self.sin_cache[:, :, :seq_len], token_idx
        )
