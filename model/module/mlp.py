"""An implement of MLP."""

import logging
import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    def __init__(self, layers: List[int], need_relu_at_output: bool = False) -> None:
        assert len(layers) >= 2, f"len(layers) should ge then 2."
        super().__init__()
        self._layers = nn.Sequential()
        for i in range(1, len(layers)):
            self._layers.add_module(f"layer_{i}", nn.Linear(layers[i - 1], layers[i]))
            if i < len(layers) - 1 or need_relu_at_output:
                self._layers.add_module(f"relu_{i}", nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._layers(inputs)


def _test():
    mlp = MLP(layers=[8, 4, 2])
    inputs = torch.rand(10, 8)
    outputs = mlp(inputs)
    logging.info(f"inputs: {inputs.shape}\n{inputs}")
    logging.info(f"outputs: {outputs.shape}\n{outputs}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test()
