import ttnn
from models.common.lightweightmodule import LightweightModule


class PhiMLP(LightweightModule):
    def __init__(self, device, config, parameters):
        super().__init__()
        self.device = device
        self.config = config
        self.linear1_weight = parameters.fc1.weight
        self.linear1_bias = parameters.fc1.bias
        self.linear2_weight = parameters.fc2.weight
        self.linear2_bias = parameters.fc2.bias

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x1 = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias, activation="gelu")

        x2 = ttnn.linear(
            x1,
            self.linear2_weight,
            bias=self.linear2_bias,
        )
        ttnn.deallocate(x1)

        return x2
