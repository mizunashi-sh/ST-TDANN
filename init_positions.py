from dataclasses import dataclass
from utils.spatial_units import (
    collapse_and_trim_neighborhoods,
    jitter_positions,
    place_conv,
    precompute_neighborhoods,
)
from utils.types import VVSRegion, Dims
from utils.constants import TISSUE_SIZES, NEIGHBORHOOD_WIDTHS, BRAIN_MAPPING, OUTPUT_DIMS_FOR_224_INPUTS
from typing import Dict, List, Any
from positions import LayerPositions
from pathlib import Path

INITIAL_RF_OVERLAP = 0.1 
POS_VERSION = 2

@dataclass
class LayerPlacement:
    name: str
    tissue_size: float
    dims: Dims
    rf_overlap: float = INITIAL_RF_OVERLAP
    neighborhood_width: float = 0.5

def get_placement_configs(
    brain_mapping: Dict[str, VVSRegion],
    output_dims_for_224_inputs: Dict[str, Dims],
) -> List[LayerPlacement]:
    placement_configs: List[LayerPlacement] = [
        LayerPlacement(
            name=layer,
            tissue_size=TISSUE_SIZES[brain_area],
            dims=output_dims_for_224_inputs[layer],
            neighborhood_width=NEIGHBORHOOD_WIDTHS[brain_area],
        )
        for (layer, brain_area) in brain_mapping.items()
    ]

    initial_x = placement_configs[0].dims[-1]
    for cfg in placement_configs:
        downsampling_ratio = initial_x / cfg.dims[-1]
        cfg.rf_overlap = min(INITIAL_RF_OVERLAP * downsampling_ratio, 1.0)

    return placement_configs

def create_position_dict(cfg: LayerPlacement) -> Dict[str, Any]:
    positions, rf_radius = place_conv(
        dims=cfg.dims,
        pos_lims=(0, cfg.tissue_size),
        offset_pattern="random",
        rf_overlap=cfg.rf_overlap,
        return_rf_radius=True,
    )

    positions = jitter_positions(positions, jitter=0.3)

    neighborhood_list = precompute_neighborhoods(
        positions, radius=cfg.neighborhood_width / 2, n_neighborhoods=20_000
    )

    neighborhoods = collapse_and_trim_neighborhoods(
        neighborhood_list, keep_fraction=0.95, keep_limit=500, target_shape=None
    )

    return {
        "positions": positions,
        "neighborhoods": neighborhoods,
        "radius": cfg.neighborhood_width / 2,
    }

def init_positions(config):
    save_dir = config['initial_position_dir']
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"Saving to {save_dir}")

    placement_configs = get_placement_configs(
        BRAIN_MAPPING[config['model']['base_model_name']],
        OUTPUT_DIMS_FOR_224_INPUTS[config['model']['base_model_name']],
    )

    for cfg in placement_configs:
        position_dict = create_position_dict(cfg)
        layer_positions = LayerPositions(
            name=cfg.name,
            dims=cfg.dims,
            coordinates=position_dict["positions"],
            neighborhood_indices=position_dict["neighborhoods"],
            neighborhood_width=position_dict["radius"] * 2,
        )
        layer_positions.save(save_dir)

    version_file = save_dir / "version.txt"
    with open(version_file, "w") as stream:
        stream.write(str(POS_VERSION))

config = dict(
    initial_position_dir = Path('positions/model_positions_224_mt_modified/'),
    model = dict(
        base_model_name = 'r3d18'
    )
)
init_positions(config)
