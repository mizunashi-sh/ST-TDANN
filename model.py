import torch
import torch.nn as nn
from torchvision.models.video.resnet import r3d_18, r2plus1d_18
from positions import NetworkPositions
from pathlib import Path

class SpatialR3d18(nn.Module):

    def __init__(self, configs):
        super(SpatialR3d18, self).__init__()

        self.positions = self._load_positions(configs['position_dir'])

        self.base_model = r3d_18()
        self.base_model.fc = nn.Identity()

        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions
        
    def forward(self, x):
        x = self.base_model.stem(x)
        x = self.maxpool(x)

        x_1_0 = self.base_model.layer1[0](x)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_2_0 = self.base_model.layer2[0](x_1_1)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_3_0 = self.base_model.layer3[0](x_2_1)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_4_0 = self.base_model.layer4[0](x_3_1)
        x_4_1 = self.base_model.layer4[1](x_4_0)

        x = self.base_model.avgpool(x_4_1)
        flat_outputs = torch.flatten(x, 1)

        spatial_outputs = {
            "layer1_0": (torch.mean(x_1_0, dim=2), self.positions["layer1.0"]),
            "layer1_1": (torch.mean(x_1_1, dim=2), self.positions["layer1.1"]),
            "layer2_0": (torch.mean(x_2_0, dim=2), self.positions["layer2.0"]),
            "layer2_1": (torch.mean(x_2_1, dim=2), self.positions["layer2.1"]),
            "layer3_0": (torch.mean(x_3_0, dim=2), self.positions["layer3.0"]),
            "layer3_1": (torch.mean(x_3_1, dim=2), self.positions["layer3.1"]),
            "layer4_0": (torch.mean(x_4_0, dim=2), self.positions["layer4.0"]),
            "layer4_1": (torch.mean(x_4_1, dim=2), self.positions["layer4.1"]),
        }

        return flat_outputs, spatial_outputs
    

class SpatialR2plus1d18(nn.Module):
    def __init__(self, configs):
        super(SpatialR2plus1d18, self).__init__()

        self.positions = self._load_positions(configs['position_dir'])

        self.base_model = r2plus1d_18()
        self.base_model.fc = nn.Identity()

        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))

    def _load_positions(self, position_dir):
        network_positions = NetworkPositions.load_from_dir(position_dir)
        network_positions.to_torch()
        return network_positions.layer_positions

    def forward(self, x):
        x = self.base_model.stem(x)
        x = self.maxpool(x)

        x_1_0 = self.base_model.layer1[0](x)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_2_0 = self.base_model.layer2[0](x_1_1)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_3_0 = self.base_model.layer3[0](x_2_1)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_4_0 = self.base_model.layer4[0](x_3_1)
        x_4_1 = self.base_model.layer4[1](x_4_0)

        x = self.base_model.avgpool(x_4_1)
        flat_outputs = torch.flatten(x, 1)

        spatial_outputs = {
            "layer1_0": (torch.mean(x_1_0, dim=2), self.positions["layer1.0"]),
            "layer1_1": (torch.mean(x_1_1, dim=2), self.positions["layer1.1"]),
            "layer2_0": (torch.mean(x_2_0, dim=2), self.positions["layer2.0"]),
            "layer2_1": (torch.mean(x_2_1, dim=2), self.positions["layer2.1"]),
            "layer3_0": (torch.mean(x_3_0, dim=2), self.positions["layer3.0"]),
            "layer3_1": (torch.mean(x_3_1, dim=2), self.positions["layer3.1"]),
            "layer4_0": (torch.mean(x_4_0, dim=2), self.positions["layer4.0"]),
            "layer4_1": (torch.mean(x_4_1, dim=2), self.positions["layer4.1"]),
        }

        return flat_outputs,spatial_outputs
