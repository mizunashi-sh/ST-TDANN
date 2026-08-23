import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
from lightly.models.modules.heads import MoCoProjectionHead
from dataset import ContrastiveUCF101

from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,  
)
from model import SpatialR3d18
from losses.spatial_correlation_ntxent_loss import SpatialCorrelationNTXent
from transforms import ContrastiveTransform, ToCDHW
from typing import Dict
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

max_epochs = 400
warmup_epochs = 10       
warmup_start_lr = 0.001 
base_lr = 0.03

batch_size = 64
num_workers = 32

class MoCoModel(pl.LightningModule):

    def __init__(self, configs: Dict):
        super().__init__()

        self.backbone = SpatialR3d18(configs=configs['model'])
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.projection_head = MoCoProjectionHead(512, 2048, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = SpatialCorrelationNTXent(loss_config=configs['loss'], device=self.device)
    
    def training_step(self, batch, batch_idx):
        x_q, x_k = batch

        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        q, q_spatial = self.backbone(x_q)
        q = q.flatten(start_dim=1)
        q = self.projection_head(q)

        k, shuffle = batch_shuffle(x_k)
        k, k_spatial = self.backbone_momentum(k)
        k = k.flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion((q, q_spatial), (k, k_spatial))
        self.log("train_loss_ssl", loss)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optim, 
            start_factor=warmup_start_lr/base_lr,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=max_epochs-warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim,
            schedulers=[scheduler_warmup, scheduler_cos],
            milestones=[warmup_epochs]
        )

        return {
            'optimizer': optim,
            'lr_scheduler': scheduler
        }


transform = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.UniformTemporalSubsample(16),
    transforms.RandomCrop((224, 224)),
    ToCDHW()
])

noise_transform = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.UniformTemporalSubsample(16),
    transforms.RandomCrop((224, 224)),
    #transforms.GaussianNoise(),
    ToCDHW()
])

configs = {
    'model': {
        'position_dir': '/home/guzhaotian/corticalmaps/positions/model_positions_224_mt_modified'
    },
    'loss': {
        'neighborhoods_per_batch': 1,
        'layer_weights': {
            "layer1_0": 0.0,
            "layer1_1": 0.0,
            "layer2_0": 0.0,
            "layer2_1": 0.0,
            "layer3_0": 0.0,
            "layer3_1": 0.0,
            "layer4_0": 0.0,
            "layer4_1": 0.0,
        },
        'temperature': 0.1,
        'memory_bank_size': 16384
    }
}

def main():
    dataset_train = ContrastiveUCF101(root='./datasets/UCF-101', annotation_path='./datasets/splits/UCF101_Action_detection_splits', frames_per_clip=32, step_between_clips=16, num_workers=num_workers, transform=ContrastiveTransform(transform, noise_transform))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = MoCoModel(configs)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=2,
        accelerator="gpu",
        precision=32,
        strategy='ddp',
        accumulate_grad_batches=4,
    )
    trainer.fit(model, dataloader_train)

if __name__=='__main__':
    main()
