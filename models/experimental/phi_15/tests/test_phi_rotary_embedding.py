import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.phi_15.tt.phi_rotary_embedding import PhiRotaryEmbedding

import transformers
from transformers import PhiForCausalLM

from loguru import logger


@pytest.fixture(scope="module")
def torch_rope():
    model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", low_cpu_mem_usage=True).eval()

    rotary_dim = model.config.hidden_size // model.config.num_attention_heads
    torch_rope = transformers.models.phi.modeling_phi.PhiRotaryEmbedding(dim=rotary_dim).eval()

    return torch_rope


@pytest.mark.parametrize(
    "input_shape, expected_pcc",
    (
        (
            (32, 32, 128, 64),
            0.99,
        ),
    ),
)
def test_phi_rotary_embedding(
    input_shape,
    expected_pcc,
    torch_rope,
):
    torch.set_default_device("cpu")
    torch.manual_seed(0)

    torch_rope_config = transformers.PhiConfig.from_pretrained("microsoft/phi-1_5")

    batch, num_kv_heads, query_length, head_dim = input_shape

    torch_value_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)
    torch_query_layer = torch.rand(
        batch, torch_rope_config.num_attention_heads, query_length, head_dim, dtype=torch.float32
    )
    torch_key_layer = torch.rand(batch, num_kv_heads, query_length, head_dim, dtype=torch.float32)
    torch_cos, torch_sin = torch_rope.forward(torch_value_layer, seq_len=query_length)
    torch_query_embed, torch_key_embed = transformers.models.phi.modeling_phi.apply_rotary_pos_emb(
        torch_query_layer, torch_key_layer, torch_cos, torch_sin, None
    )

    ttnn_device = ttnn.open_device(device_id=0)

    ttnn_model = PhiRotaryEmbedding(ttnn_device, torch_rope_config)

    ttnn_query_layer = ttnn.from_torch(
        torch_query_layer, device=ttnn_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_key_layer = ttnn.from_torch(torch_key_layer, device=ttnn_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_query_embed = ttnn_model(ttnn_query_layer)
    ttnn_key_embed = ttnn_model(ttnn_key_layer)

    query_embed_pcc = assert_with_pcc(torch_query_embed, ttnn.to_torch(ttnn_query_embed), expected_pcc)
    key_embed_pcc = assert_with_pcc(torch_key_embed, ttnn.to_torch(ttnn_key_embed), expected_pcc)

    logger.success(f"Query Embeddings Passed: pcc: {query_embed_pcc}, expected: {expected_pcc}")
    logger.success(f"Key Embeddings Passed: pcc: {key_embed_pcc}, expected: {expected_pcc}")

    ttnn.close_device(ttnn_device)
