import ttnn
import pytest
import torch
import time

from models.experimental.phi_15.tt.phi_attention import PhiAttention

from ttnn.model_preprocessing import preprocess_model_parameters

import transformers
from transformers import PhiForCausalLM
from transformers.cache_utils import DynamicCache
from tests.ttnn.utils_for_testing import assert_with_pcc

from loguru import logger


def strip_state_dict_prefix(state_dict, prefix):
    return {k[len(prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def get_model_prefix(layer_index: int = 0):
    return f"model.layers.{layer_index}.self_attn"


@pytest.fixture(scope="module")
def torch_attention():
    model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", low_cpu_mem_usage=True).eval()

    state_dict = model.state_dict()
    model_config = transformers.PhiConfig.from_pretrained("microsoft/phi-1_5")

    state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())
    torch_attention = transformers.models.phi.modeling_phi.PhiAttention(model_config, layer_idx=0).eval()
    torch_attention.load_state_dict(state_dict)

    return torch_attention


@pytest.mark.parametrize(
    "batch, seq_len, gen_len, expected_pcc",
    (
        (1, 128, 2, 0.99),
        # (1, 128, 10, 0.99),
        # (1, 10, 128, 0.99),
        # (32, 128, 128, 0.99),
    ),
)
def test_phi_attention(batch, seq_len, gen_len, expected_pcc, torch_attention):
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    ttnn_device = ttnn.open_device(device_id=0)

    config = transformers.PhiConfig.from_pretrained("microsoft/phi-1_5")
    hidden_size = config.hidden_size

    # hidden_states, tt_hidden_states = create_attention_input(mode, ttnn.bfloat16, batch, seq_len, config.hidden_size, ttnn_device)
    hidden_states = torch.rand(batch, seq_len, hidden_size)
    attention_mask = torch.ones(batch, 1, seq_len, seq_len, dtype=torch.bool).tril()
    torch_position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    torch_past_key_values = DynamicCache()

    # Attention prefill
    torch_output, torch_attention_weights, torch_past_key_values = torch_attention(
        hidden_states,
        attention_mask=attention_mask,
        past_key_value=torch_past_key_values,
        position_ids=torch_position_ids,
        use_cache=True,
    )

    # TTNN attention
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_attention, device=ttnn_device, prefix=f"model.layers.self_attn"
    )

    tt_model = PhiAttention(
        device=ttnn_device,
        config=config,
        parameters=parameters,
        layer_idx=0,
    )

    tt_position_ids = ttnn.from_torch(
        torch_position_ids,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_position_ids = ttnn.from_torch(
        torch_position_ids,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN attention prefill
    start = time.time()
    tt_output, tt_layer_present = tt_model(
        tt_hidden_states,
        attention_mask=tt_attention_mask,
        position_ids=tt_position_ids,
        past_key_values=None,
        use_cache=True,
    )
    end = time.time()
    duration = end - start

    passed, pcc = assert_with_pcc(torch_output, ttnn.to_torch(tt_output).to(torch_output.dtype), expected_pcc)
    logger.success(
        f"Prefill : {passed} - Time: {duration}- Output: {tt_output.shape}  -  pcc: {pcc}, expected: {expected_pcc}"
    )

    # Attention decode
    """
    for i in range(gen_len):

        next_token_hidden = torch_output[:, -1:, :]
        past_key_values_length = torch_past_key_values.get_usable_length(1)
        attention_mask = torch.ones(batch, 1, 1, past_key_values_length+1, dtype=torch.bool).tril()
        torch_position_ids = torch.tensor([past_key_values_length]).unsqueeze(0)

        torch_output, torch_attention_weights, torch_past_key_values = torch_attention(
            next_token_hidden,
            attention_mask=attention_mask,
            past_key_value=torch_past_key_values,
            position_ids=torch_position_ids,
            output_attentions=True,
            use_cache=True,
        )

        print(f"Step {i}: Output shape = {torch_output.shape}")
        """

    ttnn.close_device(ttnn_device)
