from .types import VVSRegion
from typing import Dict

RNG_SEED = 424

# estimate of the size of the retina sensitive to central eight degrees, given an
# estimate of roughly 300 microns per visual degree
RETINA_SIZE = 2.4  # mm

# estimate of how big, in mm, V1 in a single hemisphere is, from Benson et al
#  (sqrt 13.5cm^2 = 36.75mm)
V1_SIZE = 34.37  # mm

# V2 estimate also from Benson et al, slightly smaller than V1
V2_SIZE = 30.72  # mm

# human V4 estimates are hard to find, but this seems like a reasonable approximation
MT_SIZE = 7.07  # mm

# this is measured from the average size of the responsive VTC ROI in the NSD
LIP_SIZE = 7.48  # mm

TISSUE_SIZES: Dict[VVSRegion, float] = {
    "retina": RETINA_SIZE,
    "V1": V1_SIZE,
    "V2": V2_SIZE,
    "MT": MT_SIZE,
    "LIP": LIP_SIZE,
}

NEIGHBORHOOD_WIDTHS: Dict[VVSRegion, float] = {
    "retina": 0.0475,
    "V1": 2.7,
    "V2": 4.2,
    "MT": 2.1,
    "LIP": 2.7,
}

BRAIN_MAPPING = {
    "r3d18": {
        "layer1.0": "retina",
        "layer1.1": "retina",
        "layer2.0": "V1",
        "layer2.1": "V1",
        "layer3.0": "V2",
        "layer3.1": "MT",
        "layer4.0": "LIP",
        "layer4.1": "LIP",
    },
    "r3d50": {
        "layer1.0": "retina",
        "layer1.1": "retina",
        "layer1.2": "retina",
        "layer2.0": "V1",
        "layer2.1": "V1",
        "layer2.2": "V1",
        "layer2.3": "V2",
        "layer3.0": "MT",
        "layer3.1": "MT",
        "layer3.2": "MT",
        "layer3.3": "MT",
        "layer3.4": "MT",
        "layer3.5": "MT",
        "layer4.0": "LIP",
        "layer4.1": "LIP",
        "layer4.2": "LIP",
    },
}

OUTPUT_DIMS_FOR_224_INPUTS = {
    "r3d18": {
        "layer1.0": (64, 8, 56, 56),
        "layer1.1": (64, 8, 56, 56),
        "layer2.0": (128, 4, 28, 28),
        "layer2.1": (128, 4, 28, 28),
        "layer3.0": (256, 2, 14, 14),
        "layer3.1": (256, 2, 14, 14),
        "layer4.0": (512, 1, 7, 7),
        "layer4.1": (512, 1, 7, 7),
    },
    "r3d50": {
        "layer1.0": (256, 8, 56, 56),
        "layer1.1": (256, 8, 56, 56),
        "layer1.2": (256, 8, 56, 56),
        "layer2.0": (512, 8, 28, 28),
        "layer2.1": (512, 4, 28, 28),
        "layer2.2": (512, 4, 28, 28),
        "layer2.3": (512, 4, 28, 28),
        "layer3.0": (1024, 2, 14, 14),
        "layer3.1": (1024, 2, 14, 14),
        "layer3.2": (1024, 2, 14, 14),
        "layer3.3": (1024, 2, 14, 14),
        "layer3.4": (1024, 2, 14, 14),
        "layer3.5": (1024, 2, 14, 14),
        "layer4.0": (2048, 1, 7, 7),
        "layer4.1": (2048, 1, 7, 7),
        "layer4.2": (2048, 1, 7, 7),
    },
}