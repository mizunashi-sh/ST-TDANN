from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Tuple, Union

import numpy as np
import torch
from utils.types import Dims, VVSRegion

from utils.array import FlatIndices, get_flat_indices

@dataclass
class LayerPositions:

    # name should be the name of the layer these positions are for
    name: str

    # dims are in CHW format if conv, or a 1-tuple if fc
    dims: Union[Dims, Tuple[int]]

    # coordinates should be an N x 2 matrix with the x-coordinates of each unit in the
    # first column and the y-coordinates in the second column
    coordinates: np.ndarray

    # coordinates should be an P x Q matrix. Each row is one neighborhood consisting
    # of P indices. For all p_i \in P, 0 <= p_i <= len(coordinates)
    neighborhood_indices: np.ndarray

    # neighborhood_width is the width, in mm, of the neighborhoods
    neighborhood_width: float

    def __post_init__(self):
        if len(self.dims)==1:
            assert np.prod(self.dims) == len(
                self
            ), "dims don't match number of units provided"
        elif len(self.dims)==4:
            assert self.dims[0]*self.dims[2]*self.dims[3] == len(
                self
            ), "dims don't match number of units provided"
        else:
            raise Exception(
                (
                    "Sorry, only FC (1-D shape) and conv3d (4-D shape) kernels are "
                    f"accepted, and dims was provided with {len(self.dims)} dimensions"
                )
            )
    

    @property
    def flat_indices(self) -> FlatIndices:
        if len(self.dims) == 1:
            return np.arange(self.dims[0])
        elif len(self.dims) == 4:
            return get_flat_indices(self.dims)
        else:
            raise Exception(
                (
                    "Sorry, only FC (1-D shape) and conv3d (4-D shape) kernels are "
                    f"accepted, and dims was provided with {len(self.dims)} dimensions"
                )
            )

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    def save_np(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.npz"
        np.savez(path, **vars(self))

    @classmethod
    def load(cls, path: Path) -> "LayerPositions":
        path = Path(path)
        assert path.exists(), path

        if path.suffix == ".pkl":
            with path.open("rb") as stream:
                return pickle.load(stream)
        elif path.suffix == ".npz":
            state = np.load(path)

            def _scalar(x):
                return x[()]

            return cls(
                name=_scalar(state["name"]),
                dims=state["dims"],
                coordinates=state["coordinates"],
                neighborhood_indices=state["neighborhood_indices"],
                neighborhood_width=_scalar(state["neighborhood_width"]),
            )
        raise ValueError("suffix must be npz or pkl")

    def __len__(self) -> int:
        return len(self.coordinates)


@dataclass
class NetworkPositions:
    layer_positions: Dict[str, LayerPositions]
    version: int

    @classmethod
    def load_from_dir(cls, load_dir: Path):
        load_dir = Path(load_dir)
        assert load_dir.is_dir(), load_dir
        layer_files = list(load_dir.glob("*.pkl")) + list(load_dir.glob("*.npz"))

        d = {}
        for layer_file in layer_files:
            layer_name = layer_file.stem
            d[layer_name] = LayerPositions.load(layer_file)

        version_path = load_dir / "version.txt"
        version = 1.0
        if version_path.is_file():
            with version_path.open("r") as stream:
                version = int(stream.readline())

        return cls(version=version, layer_positions=d)

    def to_torch(self):
        """
        Converts each array or float in the original layer positions to be a torch
        Tensor of the appropriate type
        """
        for pos in self.layer_positions.values():
            pos.coordinates = torch.from_numpy(pos.coordinates.astype(np.float32))
            pos.neighborhood_indices = torch.from_numpy(
                pos.neighborhood_indices.astype(int)
            )
            pos.neighborhood_width = torch.tensor(pos.neighborhood_width)
