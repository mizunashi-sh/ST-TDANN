import numpy as np
from typing import Callable, Optional, Tuple, List
from typing_extensions import Literal
from .types import Dims
import scipy
from dataclasses import dataclass


@dataclass
class Window:
    indices: np.ndarray
    lims: List[List[float]]
    num_units: int


def collapse_and_trim_neighborhoods(
    nb_list,
    keep_fraction=0.95,
    min_nb_size: int = 25,
    keep_limit=None,
    target_shape=None,
):
    assert keep_fraction > 0.0 and keep_fraction <= 1.0

    neighborhood_sizes = np.array([len(n) for n in nb_list])
    if target_shape is not None:
        # sort neighborhoods in order from smallest to largest
        sort_ind = np.argsort(neighborhood_sizes)
        target_nbs = target_shape[0]
        target_units = target_shape[1]

        # take biggest neighborhoods to maximize chances of having target_units inside
        neighborhoods_to_keep = sort_ind[-target_nbs:]
        nb_list = np.array(nb_list)[neighborhoods_to_keep]

        n_list = list()
        for indices in nb_list:
            if len(indices) < target_units:
                continue
            indices_to_keep = np.random.choice(
                indices, size=(target_units,), replace=False
            )
            n_list.append(indices_to_keep)
    else:
        print(
            "Neighborhood sizes range from %d to %d, with a median of %d"
            % (
                np.min(neighborhood_sizes),
                np.max(neighborhood_sizes),
                np.median(neighborhood_sizes),
            )
        )

        # convert keep fraction to percentile cutoff
        percentile = (1 - keep_fraction) * 100
        n_size = max(
            int(scipy.stats.scoreatpercentile(neighborhood_sizes, percentile)),
            min_nb_size,
        )
        failing = np.less(neighborhood_sizes, n_size)
        print(
            "Minimum of %d excludes %.1f%% of neighborhoods"
            % (n_size, np.mean(failing) * 100.0)
        )

        # take final unit limit as minimum of n_size and keep_limit
        if keep_limit is None:
            final_limit = n_size
        else:
            final_limit = np.min([n_size, keep_limit])

        # do the trimming
        n_list = list()
        for indices in nb_list:
            if len(indices) < final_limit:
                continue
            indices_to_keep = np.random.choice(
                indices, size=(final_limit,), replace=False
            )
            n_list.append(indices_to_keep)

    print("Kept %d neighborhoods." % len(n_list))
    neighborhoods = np.stack(n_list)
    return neighborhoods


def precompute_neighborhoods(
    positions: np.ndarray, radius: float = 0.5, n_neighborhoods: int = 10
):
    start_xs = np.random.uniform(
        low=np.min(positions[:, 0]) + radius,
        high=np.max(positions[:, 0]) - radius,
        size=(n_neighborhoods,),
    )

    start_ys = np.random.uniform(
        low=np.min(positions[:, 1]) + radius,
        high=np.max(positions[:, 1]) - radius,
        size=(n_neighborhoods,),
    )

    neighborhoods = [
        indices_within_limits(
            positions,
            [[xstart - radius, xstart + radius], [ystart - radius, ystart + radius]],
        )
        for (xstart, ystart) in zip(start_xs, start_ys)
    ]
    return neighborhoods


def place_conv(
    dims: Dims,
    pos_lims: Tuple[float, ...],
    offset_pattern: Literal["random", "grid"],
    rf_overlap: float = 0.0,
    flatten: bool = True,
    return_rf_radius: bool = False,
):
    num_chan, num_depth, num_x, num_y = dims
    if len(pos_lims) == 2:
        rf_centers, rf_radius = compute_rf_centers(dims, rf_overlap, pos_lims)
        xx, yy = np.meshgrid(rf_centers, rf_centers)
    elif len(pos_lims) == 4:
        x_rf_centers, x_rf_radius = compute_rf_centers(dims, rf_overlap, pos_lims[:2])
        y_rf_centers, y_rf_radius = compute_rf_centers(dims, rf_overlap, pos_lims[2:])
        xx, yy = np.meshgrid(x_rf_centers, y_rf_centers)

        # not sure about this choice, but seems ok for now as long as aspect ratio is
        # defined in the interval [0, 1]
        rf_radius = max(x_rf_radius, y_rf_radius)

    # N x 2 array; x in first col, y in second
    anchors = np.stack((xx.ravel(), yy.ravel()), axis=1)

    # create a set of position offsets dictating how units corresponding to different
    # channels should be moved relative to the position of the anchor. There will be
    # as many offsets as there are anchors, and the size of each offset vector is
    # the number of channels in the layer
    if offset_pattern == "random":
        offsets = [
            np.random.uniform(size=(num_chan, 2), low=-rf_radius, high=rf_radius)
            for _ in range(len(anchors))
        ]
    elif offset_pattern == "grid":
        offsets = [
            grid_pattern(num_chan, extent=rf_radius) for _ in range(len(anchors))
        ]
    else:
        raise Exception("Offset pattern not recognized")

    position_list = []
    for offset, anchor in zip(offsets, anchors):
        position_list.append(anchor + offset)

    # stack all spatial anchors
    positions = np.stack(position_list)

    # move channel to first dimension, so now we have (channel, x * y, 2)
    positions = np.swapaxes(positions, 0, 1)

    # reshape to either (channel * x * y, 2) or (channel, x, y, 2)
    if flatten:
        positions = positions.reshape((-1, 2))
    else:
        positions = positions.reshape((num_chan, num_x, num_y, 2))

    if return_rf_radius:
        return positions, rf_radius

    return positions


def jitter_positions(pos: np.ndarray, jitter: float = 0):
    # add noise to each unit according to jitter
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=jitter, size=pos.shape)
    jittered = pos + noise

    # squish: divide the positions by the ratio of new range to old range
    jittered_squished = jittered / (np.ptp(jittered, axis=0) / np.ptp(pos, axis=0))

    # slide: start all positions at 0, then add back the original minimum (probably 0)
    return jittered_squished - np.min(jittered_squished, axis=0) + np.min(pos, axis=0)

def grid_pattern(n, extent: float = 1.0):
    """
    Returns grid that is close to square but missing
    some entries to match n
    """

    sq_ceil = int(np.ceil(np.sqrt(n)))

    nx = ny = sq_ceil

    xs = np.linspace(-extent, extent, nx)
    ys = np.linspace(-extent, extent, ny)

    xx, yy = np.meshgrid(xs, ys)
    xx = xx.ravel()[:n]
    yy = yy.ravel()[:n]

    perm = np.random.permutation(n)
    xx = xx[perm]
    yy = yy[perm]

    return np.stack((xx, yy), axis=1)

def compute_rf_centers(layer_dims, rf_overlap, pos_lims):
    assert len(layer_dims) > 1

    num_rfs = layer_dims[-1]
    map_width = np.ptp(pos_lims)

    rf_width = map_width / (num_rfs - (num_rfs * rf_overlap) + rf_overlap)
    rf_radius = rf_width / 2.0

    rf_centers = np.linspace(rf_radius, map_width - rf_radius, num_rfs)
    return rf_centers, rf_radius

def indices_within_limits(positions, limits: List[List[float]], unit_limit=None):
    """
    Inputs
        positions: N x 2 matrix of positions, where N is the number of units
        limits:  The ith list indexes the position limits for the ith column of the
            `positions` matrix
        unit_limit: maximum number of units to retain


    Returns
        indices: a 1-D array of indices where positions are within the limits
    """

    indices = np.where(
        (positions[:, 0] >= limits[0][0])
        & (positions[:, 0] <= limits[0][1])
        & (positions[:, 1] >= limits[1][0])
        & (positions[:, 1] <= limits[1][1])
    )[0]

    if isinstance(unit_limit, int) and len(indices) > unit_limit:
        indices = np.random.choice(indices, size=(unit_limit,), replace=False)

    return indices
