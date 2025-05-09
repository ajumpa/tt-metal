import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.phi_15.tt.phi_rotary_embedding import PhiRotaryEmbedding


class PhiAttention(LightweightModule):
    def __init__(self, device, config, parameters, layer_idx):
        super().__init__()

        self.fused_qkv_weight = ttnn.concat(
            [parameters.q_proj.weight, parameters.k_proj.weight, parameters.v_proj.weight], dim=-1
        )

        self.fused_qkv_bias = ttnn.concat(
            [parameters.q_proj.bias, parameters.k_proj.bias, parameters.v_proj.bias],
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        self.dense_weight = parameters.dense.weight
        self.dense_bias = ttnn.to_device(parameters.dense.bias, device, ttnn.L1_MEMORY_CONFIG)

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_dim = config.hidden_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.rotary_embedding = PhiRotaryEmbedding(device, config)
        self.layer_idx = layer_idx

    # Optimized multihead attention https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tutorials/ttnn_tutorials/003.html
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        past_key_values: ttnn.Tensor,
        use_cache: bool = False,
    ):
        batch, seq_len, hidden_size = hidden_states.shape

        assert hidden_size == self.hidden_dim

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        fused_qkv_output = ttnn.linear(
            hidden_states,
            self.fused_qkv_weight,
            bias=self.fused_qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )

        (
            query,
            key,
            value,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            transpose_key=False,
            num_heads=self.num_heads,
        )
        ttnn.deallocate(fused_qkv_output)

        query = self.rotary_embedding(query)
        key = self.rotary_embedding(key)
        key = ttnn.transpose(key, 2, 3)

        attention_scores = ttnn.matmul(
            query,
            key,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        attention_probs = ttnn.transformer.attention_softmax_(
            attention_scores, attention_mask=attention_mask, head_size=self.head_dim
        )

        context_layer = ttnn.matmul(
            attention_probs,
            value,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        ttnn.deallocate(attention_probs)

        context_layer_after_concatenate_heads = ttnn.transformer.concatenate_heads(
            context_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(context_layer)

        self_output = ttnn.linear(
            context_layer_after_concatenate_heads,
            self.dense_weight,
            bias=self.dense_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=batch, x=12),
        )
        ttnn.deallocate(context_layer_after_concatenate_heads)

        return self_output, past_key_values
