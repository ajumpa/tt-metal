import ttnn
from models.common.lightweightmodule import LightweightModule
from .phi_attention import PhiAttention
from .phi_mlp import PhiMLP


class PhiDecoder(LightweightModule):
    def __init__(
        self,
        device,
        attention_module: PhiAttention,
        mlp_module: PhiMLP,
        layernorm_weights: ttnn.Tensor,
        layernorm_bias: ttnn.Tensor,
        dim=2048,
        heads=32,
    ):
        super().__init__()

        self.device = device
        self.attention = attention_module
        self.feed_forwad = mlp_module
        self.layernorm_weights = layernorm_weights
        self.layernorm_bias = layernorm_bias

        self.dim = dim
        self.heads = heads

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        residual = hidden_states

        hidden_states = ttnn.layer_norm(hidden_states, self.layernorm_weights, bias=self.layernorm_bias)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = ttnn.layer_norm(hidden_states, self.layernorm_weights, bias=self.layernorm_bias)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states
