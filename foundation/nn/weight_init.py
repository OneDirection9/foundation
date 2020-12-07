from __future__ import absolute_import, division, print_function

from torch import nn

__all__ = [
    "constant_init",
    "normal_init",
    "uniform_init",
    "xavier_normal_init",
    "xavier_uniform_init",
    "kaiming_normal_init",
    "kaiming_uniform_init",
    "caffe2_xavier_init",
    "caffe2_msra_init",
]


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if module.weight is not None:
        # weight is None for nn.BatchNorm when instantiate with affine=False
        nn.init.constant_(module.weight, val)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0) -> None:
    nn.init.normal_(module.weight, mean, std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module: nn.Module, a: float = 0.0, b: float = 1.0, bias: float = 0) -> None:
    nn.init.uniform_(module.weight, a, b)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_normal_init(module: nn.Module, gain: float = 1.0, bias: float = 0) -> None:
    nn.init.xavier_normal_(module.weight, gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_uniform_init(module: nn.Module, gain: float = 1.0, bias: float = 0) -> None:
    nn.init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_normal_init(
    module: nn.Module,
    a: int = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
) -> None:
    nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_uniform_init(
    module: nn.Module,
    a: int = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
) -> None:
    nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module: nn.Module) -> None:
    """Initializes `module.weight` using the "XavierFill" implemented in Caffe2."""
    # Caffe2 implementation of XavierFill in fact corresponds to kaiming_uniform_ in PyTorch
    kaiming_uniform_init(module, a=1, mode="fan_in", nonlinearity="leaky_relu", bias=0)


def caffe2_msra_init(module: nn.Module) -> None:
    """Initializes `module.weight` using the "MSRAFill" implemented in Caffe2."""
    kaiming_normal_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0)
