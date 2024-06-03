from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, ExponentialFamily, Gamma, Poisson, constraints
from torch.distributions.utils import (
    lazy_property,
    broadcast_all,
    logits_to_probs,
    probs_to_logits,
)

from ..utilities.math_tools import (
    log_probability_nb_positive,
    log_probability_zinb_positive,
    transform_counts_logits_to_mean_variance, 
    transform_mean_variance_to_counts_logits
)
