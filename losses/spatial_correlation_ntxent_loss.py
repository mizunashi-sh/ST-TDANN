import torch
import torch.nn as nn
from classy_vision.losses import ClassyLoss
from lightly.loss import NTXentLoss

def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    stddev[stddev == 0] = 1
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    c = torch.clamp(c, -1.0, 1.0)

    return c


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def pdist(positions: torch.Tensor, p: int = 2) -> torch.Tensor:
    return torch.norm(positions[:, None] - positions, dim=2, p=p)


def lower_tri(mat: torch.Tensor):
    N = mat.shape[0]
    return mat[torch.tril_indices(N, N, offset=-1).unbind()]


def spatial_loss_batch(
    base_loss_fn,
    features: torch.Tensor,
    positions: torch.Tensor,
    neighborhoods: torch.Tensor,
    n_runs: int = 1,
    agg_func=torch.mean,
):
    scl_list = torch.stack(
        [base_loss_fn(features, positions, neighborhoods) for _ in range(n_runs)]
    )

    if agg_func is None:
        return scl_list

    return agg_func(scl_list)


def spatial_correlation_loss(
    features: torch.Tensor, positions: torch.Tensor, neighborhoods: torch.Tensor
):
    indices = neighborhoods[torch.randint(len(neighborhoods), (1,))].squeeze()

    neighborhood_features = torch.flatten(features, start_dim=1)[:, indices]
    neighborhood_positions = positions[indices]

    distance_similarity = 1 / (pdist(neighborhood_positions) + 1)
    response_similarity = corrcoef(neighborhood_features.t())

    similarity_alignment = pearsonr(
        lower_tri(response_similarity), lower_tri(distance_similarity)
    )

    loss = (1 - similarity_alignment) / 2.0
    return loss

class SpatialCorrelationLossModule(nn.Module):
    def __init__(self, neighborhoods_per_batch, device):
        super(SpatialCorrelationLossModule, self).__init__()
        self.base_loss = spatial_correlation_loss
        self.neighborhoods_per_batch = neighborhoods_per_batch
        self.device = device

    def forward(self, features, positions, neighborhoods):
        return spatial_loss_batch(
            self.base_loss,
            features.to(self.device),
            positions.to(self.device),
            neighborhoods.to(self.device),
            n_runs=self.neighborhoods_per_batch,
        )

class SpatialCorrelationNTXent(ClassyLoss):
    """
    Config params:
        spatial_weight: amount by which to scale spatial loss relative to simclr loss
        ignore_index: sample should be ignored for loss, optional
        temperature: specify temperature for softmax. Default 1.0
    """

    def __init__(self, loss_config, device):
        super(SpatialCorrelationNTXent, self).__init__()
        self.loss_config = loss_config

        self._neighborhoods_per_batch = self.loss_config['neighborhoods_per_batch']
        self._layer_weights = self.loss_config['layer_weights']

        # initialize the loss modules
        self._temperature = self.loss_config['temperature']
        self._memory_bank_size = self.loss_config['memory_bank_size']
        self.criterion = NTXentLoss(
            self._temperature, self._memory_bank_size
        )

        self.device = device
        self.spatial_loss = SpatialCorrelationLossModule(self._neighborhoods_per_batch, self.device)

    def forward(self, output0, output1):
        flat_outputs0, spatial_outputs0 = output0
        flat_outputs1, spatial_outputs1 = output1

        loss = self.criterion(flat_outputs0, flat_outputs1)

        for layer, layer_output in spatial_outputs0.items():
            features, pos = layer_output
            layer_loss = self.spatial_loss(
                features, pos.coordinates, pos.neighborhood_indices
            )
            loss += self._layer_weights[layer] * layer_loss

        return loss
