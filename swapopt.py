import torch
from dataclasses import dataclass, field
from typing import List
from model import SpatialR3d18
from losses import numpy_losses as losses
import numpy as np
from tqdm import tqdm
import copy
from typing import Any, Dict, Optional, List
from utils.spatial_units import precompute_neighborhoods, collapse_and_trim_neighborhoods
from positions import LayerPositions
from pathlib import Path
import math
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transforms
from transforms import ToCDHW
from dataset import VideoFolderDataset
from collections import OrderedDict
from utils.constants import RNG_SEED


@dataclass
class Metrics:
    num_swaps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)


def make_iterable(x):
    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x


def remove_spines(axes, to_remove=None):
    # if axes is a 2d array, flatten
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    # enforce inputs as list
    axes = make_iterable(axes)

    if to_remove == "all":
        to_remove = ["top", "right", "left", "bottom"]
    elif to_remove is None:
        to_remove = ["top", "right"]

    for ax in axes:
        for spine in to_remove:
            ax.spines[spine].set_visible(False)

def plot_metrics(swapper, save_path: Path):
    line_kwargs = {"lw": 1.5, "c": "k"}

    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    axes[0].plot(swapper.metrics.num_swaps, **line_kwargs)
    axes[1].plot(swapper.metrics.losses, **line_kwargs)

    axes[0].set_ylabel("Number of Swaps")
    axes[1].set_ylabel("Neighborhood Loss")

    remove_spines(axes)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")


def swap(positions, swap_ind):
    positions[[swap_ind[0], swap_ind[1]]] = positions[[swap_ind[1], swap_ind[0]]]


def swap_optimize_positions(
    features,
    positions,
    num_steps=100,
    seed=None,
    loss_params=None,
    disable_progress_bar: bool = True,
    return_metrics: bool = False,
):
    if loss_params is None:
        loss_params = {"loss_fn": losses.pearson_scl, "dist_scaling": 12.0}

    loss_params = copy.deepcopy(loss_params)
    loss_fn = loss_params.pop("loss_fn", losses.standard_scl)

    # set random seed, if provided
    rng = np.random.default_rng(seed=RNG_SEED)

    # compute unit x unit correlations
    correlations = np.corrcoef(features.T)

    # NaN correlations go to 0
    correlations = np.where(
        np.isnan(correlations), np.zeros_like(correlations), correlations
    )

    # start counting how many swaps occurred
    n_swaps = 0
    for i in tqdm(range(num_steps), desc="swapopt", disable=disable_progress_bar):
        # pick a random pair of units to swap
        swap_ind = rng.choice(np.arange(positions.shape[0]), size=(2,), replace=False)

        # compute loss before swapping
        previous_loss = losses.swapopt_spatial_loss_wrapper(
            loss_fn, correlations, positions, swap_ind, **loss_params
        )

        # swap
        swap(positions, swap_ind)

        # compute new loss after swapping
        current_loss = losses.swapopt_spatial_loss_wrapper(
            loss_fn, correlations, positions, swap_ind, **loss_params
        )

        if current_loss >= previous_loss:
            swap(positions, swap_ind)
        else:
            n_swaps += 1

    if return_metrics:
        valid_inds = np.std(features, axis=0) > 0
        nb_loss = losses.spatial_loss_wrapper(
            loss_fn, features[:, valid_inds], positions[valid_inds], **loss_params
        )
        return positions, n_swaps, nb_loss

    return positions


@dataclass
class Metrics:
    num_swaps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)


def swap_optimize_neighborhoods(
    features: np.ndarray,
    positions: np.ndarray,
    num_steps: int = 10_000,
    steps_per_neighborhood: int = 500,
    neighborhood_width: float = 6,  # mm
    max_units_per_neighborhood: int = 500,
    seed: Optional[int] = None,
    disable_progress_bar: bool = False,
    loss_params: Optional[Dict[str, Any]] = None,
    return_metrics: bool = False,
    log_every: int = 1,
):
    # safety checks
    assert neighborhood_width <= np.ptp(
        positions[:, 0]
    ) and neighborhood_width <= np.ptp(
        positions[:, 1]
    ), "neighborhood width cannot be larger than map limits in either dimension"

    assert (
        num_steps >= steps_per_neighborhood
    ), "cannot have fewer total steps than number of steps in a single neighborhood"

    metrics = Metrics()

    # set random seed, if provided
    rng = np.random.default_rng(seed=RNG_SEED)

    num_neighborhoods = int(np.ceil(num_steps / steps_per_neighborhood))

    for neighborhood_idx in tqdm(
        range(num_neighborhoods), desc="swapopt", disable=disable_progress_bar
    ):
        start_x = rng.uniform(
            low=np.min(positions[:, 0]),
            high=np.max(positions[:, 0]) - neighborhood_width,
        )
        start_y = rng.uniform(
            low=np.min(positions[:, 1]),
            high=np.max(positions[:, 1]) - neighborhood_width,
        )

        # find indices of units within the bounding box
        indices = np.where(
            (positions[:, 0] >= start_x)
            & (positions[:, 0] <= start_x + neighborhood_width)
            & (positions[:, 1] >= start_y)
            & (positions[:, 1] <= start_y + neighborhood_width)
        )[0]

        # don't bother if there aren't even 5 units
        if len(indices) < 5:
            continue

        # filter down to unit limit
        if len(indices) >= max_units_per_neighborhood:
            indices = rng.choice(
                indices, size=(max_units_per_neighborhood,), replace=False
            )

        # run swapopt without progress bar, restricted to neighborhood units
        log_this_iteration = neighborhood_idx % log_every == 0

        output = swap_optimize_positions(
            features[:, indices],
            np.copy(positions[indices, :]),
            num_steps=steps_per_neighborhood,
            seed=seed,
            loss_params=loss_params,
            disable_progress_bar=True,
            return_metrics=log_this_iteration,
        )

        if log_this_iteration:
            positions[indices], n_swaps, loss = output

            metrics.num_swaps.append(n_swaps)
            metrics.losses.append(loss)
        else:
            positions[indices] = output

    if return_metrics:
        return positions, metrics

    return positions

class Swapper:
    def __init__(
        self, config, features, layer: str
    ):
        self.config = config
        self.layer = layer
        self.features = features

        # determine initial and new save locations
        self.initial_dir = Path(self.config['position_dir'])

        # the new save dir _is_ model specific, because the positions now depend on the
        # features. So we need to drill down and make a new folder
        self.new_save_dir = (
            self.initial_dir.parent
            / f"{self.initial_dir.name}_swapped"
        )

        # check if positions already exist
        self.blocked = False
        if (self.new_save_dir / f"{self.layer}.pkl").exists():
            self.blocked = True

        self._load_positions()

    def unblock_overwrite(self):
        self.blocked = False

    def _load_positions(self):
        initial_position_path = self.initial_dir / f"{self.layer}.pkl"
        self.initial_positions: LayerPositions = LayerPositions.load(
            initial_position_path
        )

    def swap(self):
        if self.blocked:
            print(
                (
                    "Overwriting existing position file is blocked. Call "
                    ".unblock_overwrite() if you're really sure you want to continue"
                )
            )
            return

        self.new_positions, self.metrics = swap_optimize_neighborhoods(
            self.features,
            self.initial_positions.coordinates,
            neighborhood_width=self.initial_positions.neighborhood_width,
            return_metrics=True,
        )

        # need to recompute neighborhoods here!
        neighborhood_list = precompute_neighborhoods(
            self.new_positions,
            radius=self.initial_positions.neighborhood_width / 2,
            n_neighborhoods=20_000,
        )

        self.new_neighborhood_indices = collapse_and_trim_neighborhoods(
            neighborhood_list, keep_fraction=0.95, keep_limit=500, target_shape=None
        )

    def save_positions(self):
        if self.blocked:
            print(
                (
                    "Overwriting existing position file is blocked. Call "
                    ".unblock_overwrite() if you're really sure you want "
                    "to continue"
                )
            )
            return

        assert hasattr(self, "new_positions")

        new_pos = copy.deepcopy(self.initial_positions)
        new_pos.coordinates = self.new_positions
        new_pos.neighborhood_indices = self.new_neighborhood_indices
        new_pos.save(self.new_save_dir)


def get_hook(layer, layer_name, target_dict):
    def hook_function(_layer, _input, output, name=layer_name):
        out = output.detach().numpy()
        out = np.mean(out, axis=2)

        for i in range(out.shape[0]):
            target_dict[name].append(out[i])

    hook = layer.register_forward_hook(hook_function)
    return hook


def remove_prefix(original_dict, prefix):
    new_dict = OrderedDict()
    for key, value in original_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        new_dict[new_key] = value
    return new_dict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 2
transform = transforms.Compose([
    transforms.ToDtype(torch.float, scale=True),
    ToCDHW()
])

params = torch.load('epoch=399-step=14406.ckpt', map_location=device)
params = params['state_dict']
params = remove_prefix(params, 'backbone.')
def main(swapped_layer: str):
    configs = {
        'position_dir': './positions/model_positions_224_mt_modified'
    }

    model = SpatialR3d18(configs)
    model.load_state_dict(params)
    model.eval()

    dataset = VideoFolderDataset(
        root='datasets/drifting_gratings',
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    n_batches = math.ceil(len(dataset)/batch_size)

    model_layer_strings = [swapped_layer]
    layer_results = {
        k: [] for k in model_layer_strings
    }
    hooks = []

    for layer_name in model_layer_strings:
        layer_id, block_id = layer_name.split('.')
        layer = getattr(model._modules.get('base_model'), layer_id)[int(block_id)]
        hook = get_hook(layer, layer_name, target_dict=layer_results)
        hooks.append(hook)

    for batch_idx, batch in tqdm(enumerate(dataloader), total=n_batches, desc="batch"):
        if batch_idx == n_batches:
            break

        x, _ = batch
        x = x.to(device)
        model(x)

    layer = swapped_layer
    features = layer_results[layer]
    features = np.array(features)
    features = features.reshape((len(features), -1))

    swapper = Swapper(configs, features, layer)
    swapper.swap()
    swapper.save_positions()

    save_path = Path(f"plots/{layer}_metrics.png")
    plot_metrics(swapper, save_path)

if __name__=='__main__':
    layers = ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']
    for layer in layers:
        main(layer)
