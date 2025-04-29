import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.phi_15.tt.phi_mlp import PhiMLP

import transformers
from transformers import PhiForCausalLM

from loguru import logger


def strip_state_dict_prefix(state_dict, prefix):
    return {k[len(prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def get_model_prefix(lidx: int = 0):
    return f"model.layers.{lidx}.mlp"


@pytest.fixture(scope="module")
def torch_mlp():
    model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", low_cpu_mem_usage=True).eval()

    state_dict = model.state_dict()
    model_config = transformers.PhiConfig.from_pretrained("microsoft/phi-1_5")

    mlp_state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())
    torch_mlp = transformers.models.phi.modeling_phi.PhiMLP(model_config).eval()
    torch_mlp.load_state_dict(mlp_state_dict)

    return torch_mlp


@pytest.mark.parametrize(
    "batch, seq_len, expected_pcc",
    (
        (
            1,
            2048,
            0.99,
        ),
    ),
)
def test_phi_mlp(
    batch,
    seq_len,
    expected_pcc,
    torch_mlp,
):
    torch.set_default_device("cpu")
    torch.manual_seed(0)

    torch_mlp_config = transformers.PhiConfig.from_pretrained("microsoft/phi-1_5")

    input = torch.rand(batch, 1, seq_len, torch_mlp_config.hidden_size)
    torch_output = torch_mlp(input)

    ttnn_device = ttnn.open_device(device_id=0)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_mlp, device=ttnn_device, prefix=f"model.layers.mlp"
    )

    ttnn_model = PhiMLP(ttnn_device, torch_mlp_config, parameters)
    ttnn_input = ttnn.from_torch(
        input,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_output = ttnn_model(ttnn_input)

    passed, pcc = assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output).to(torch_output.dtype), expected_pcc)
    logger.success(f"Passed? : {passed} - Output: {ttnn_output.shape}  -  pcc: {pcc}, expected: {expected_pcc}")

    ttnn.close_device(ttnn_device)
