import ttnn
from models.common.lightweightmodule import LightweightModule

from .phi_attention import PhiAttention
from .phi_decoder import PhiDecoder
from .phi_mlp import PhiMLP

import torch
from transformers import PhiForCausalLM, AutoTokenizer


class PhiModel(LightweightModule):
    def __init__(
        self, device, weights_dict, state_dict, vocab_size=51200, n_layers=24, dim=2048, heads=32, rope_theta=10000.0
    ):
        super.__init__()

        self.device = device
        self.weights_dict = weights_dict
        self.state_dict = state_dict
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.heads = heads

        self.embedding_tokens_weights = weights_dict["model.embed_tokens.weight"]
        self.layers = []
        for i in range(n_layers):
            q_proj_weight = weights_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
            q_proj_bias = weights_dict[f"model.layers.{i}.self_attn.q_proj.bias"]

            k_proj_weight = weights_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
            k_proj_bias = weights_dict[f"model.layers.{i}.self_attn.k_proj.bias"]

            v_proj_weight = weights_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
            v_proj_bias = weights_dict[f"model.layers.{i}.self_attn.v_proj.bias"]

            dense_weight = weights_dict[f"model.layers.{i}.self_attn.dense.weight"]
            dense_bias = weights_dict[f"model.layers.{i}.self_attn.dense.bias"]

            linear1_weight = weights_dict[f"model.layers.{i}.mlp.fc1.weight"]
            linear1_bias = weights_dict[f"model.layers.{i}.mlp.fc1.bias"]

            linear2_weight = weights_dict[f"model.layers.{i}.mlp.fc2.weight"]
            linear2_bias = weights_dict[f"model.layers.{i}.mlp.fc2.bias"]

            layernorm_weight = weights_dict[f"model.layers.{i}.input_layernorm.weight"]
            layernorm_bias = weights_dict[f"model.layers.{i}.input_layernorm.bias"]

            attn = PhiAttention(
                device,
                q_proj_weight,
                q_proj_bias,
                k_proj_weight,
                k_proj_bias,
                v_proj_weight,
                v_proj_bias,
                dense_weight,
                dense_bias,
                dim=dim,
                heads=heads,
                rope_theta=rope_theta,
            )

            ffn = PhiMLP(device, linear1_weight, linear1_bias, linear2_weight, linear2_bias)

            block = PhiDecoder(device, attn, ffn, layernorm_weight, layernorm_bias, dim, heads)

            self.layers.append(block)

        self.final_layernorm_weights = weights_dict["model.final_layernorm.weight"]
        self.final_layernorm_bias = weights_dict["model.final_layernorm.bias"]

        self.lm_head_weights = weights_dict["lm_head.weight"]
        self.lm_head_bias = weights_dict["lm_head.bias"]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.embedding_tokens_weights)

        for layer in self.n_layers:
            x = layer(x)

        x = ttnn.layer_norm(x, epsilon=1e-05, weight=self.final_layernorm_weights, bias=self.final_layernorm_bias)

        x = ttnn.linear(x, self.lm_head_weights, bias=self.lm_head_bias)

        return x


torch.set_default_device("cpu")


def load_phi_model(device):
    model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

    weights = {
        name: ttnn.from_torch(param.data, dtype=ttnn.bfloat16, device=device)
        for name, param in model.named_parameters()
    }

    return model, weights, tokenizer


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    model, weights_dict, tokenizer = load_phi_model(device)

    config = model.config
    state_dict = model.state_dict()

    tt_model = PhiModel(
        device=device,
        weights_dict=weights_dict,
        state_dict=state_dict,
        vocab_size=config.vocab_size,
        n_layers=config.num_hidden_layers,
        dim=config.max_position_embeddings,
        heads=config.num_attention_heads,
        rope_theta=config.rope_theta,
    )

    x = torch.randint(0, config.vocab_size, 1, 256)
    x_tt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16)

    logits = tt_model(x_tt)

    print(logits.shape)

    ttnn.close_device(device)
