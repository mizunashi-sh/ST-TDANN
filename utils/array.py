from dataclasses import dataclass
from typing import Union, Tuple, List

import numpy as np
import scipy.stats as stats

@dataclass
class FlatIndices:
    chan_flat: np.ndarray
    x_flat: np.ndarray
    y_flat: np.ndarray

def get_flat_indices(dims):
    """
    dims should be a CDHW tuple
    """
    num_channels, num_depth, num_x, num_y = dims

    # chan flat goes like [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...]
    chan_flat = np.repeat(np.arange(num_channels), num_x * num_y)

    # x flat goes like [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, ...]
    x_flat = np.repeat(np.tile(np.arange(num_x), num_channels), num_y)

    # y flat goes like [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, ...]
    y_flat = np.tile(np.arange(num_y), num_channels * num_x)

    return FlatIndices(chan_flat, x_flat, y_flat)
