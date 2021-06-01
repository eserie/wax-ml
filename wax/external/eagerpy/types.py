from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    # for static analyzers
    import jax  # noqa: F401
    import numpy  # noqa: F401
    import tensorflow  # noqa: F401
    import torch  # noqa: F401

Axes = Tuple[int, ...]
AxisAxes = Union[int, Axes]
Shape = Tuple[int, ...]
ShapeOrScalar = Union[Shape, int]

# tensorflow.Tensor, jax.numpy.ndarray and numpy.ndarray currently evaluate to Any
# we can therefore only provide additional type information for torch.Tensor
NativeTensor = Union[
    "torch.Tensor", "tensorflow.Tensor", "jax.numpy.ndarray", "numpy.ndarray"
]
