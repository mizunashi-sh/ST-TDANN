from typing import Tuple
from typing_extensions import Literal

VVSRegion = Literal[
    "retina",
    "V1",
    "V2",
    "MT",
    "LIP"
]
Dims = Tuple[int, int, int, int]
