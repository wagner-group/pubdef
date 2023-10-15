"""Define commonly used types."""

from typing import Any

import torch
from jaxtyping import Float, Int64

BatchImages = Float[torch.Tensor, "batch channels height width"]
StackImages = Float[torch.Tensor, "batch stack channels height width"]

BatchLabels = Float[torch.Tensor, "batch classes"]
StackLabels = Float[torch.Tensor, "batch stack classes"]
BatchHardLabels = Int64[torch.Tensor, "batch"]
StackHardLabels = Int64[torch.Tensor, "batch stack"]

BatchLogits = Float[torch.Tensor, "batch classes"]
StackLogits = Float[torch.Tensor, "batch stack classes"]

# List of samples represented by dicts
SamplesList = list[dict[str, Any]]
