import sys
import torch
import numpy as np
import torchio as tio
import torch.nn as nn
import pytorch_lightning as pl


sys.path.append(f"/home/doronser/workspace/")
from shared_utils.models import BaseModel  # noqa: E402

sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402

from monai.networks.blocks import Convolution




class SegModel(BaseModel):
    """Pytorch Lightning module to use for BraTS18 segmentation

    -=BraTS18 Label Mapping=-
    WT = Whole Tumor = NCR/NET+ED+ET
    TC = Tumor Core = NCR/NET+ET
    ET = Enhancing Tumor
    """
    def __init__(self, net: nn.Module, criterion=DiceLoss(classes=4), optimizer_params=None, scheduler_params=None):
        super().__init__(net, criterion, optimizer_params, scheduler_params)
        self.softmax = nn.Softmax(dim=-1)

    def prepare_batch(self, batch):
        x = torch.cat([batch['t1'][tio.DATA], batch['t1ce'][tio.DATA],
                       batch['t2'][tio.DATA], batch['flair'][tio.DATA]], dim=1)

        y = batch['seg'][tio.DATA]

        if x.size()[-1] != 160:  # pad to even dimension
            x = torch.cat([x, torch.zeros([*x.size()[:-1], 5], device=x.device)], dim=-1)
            y = torch.cat([y, torch.zeros([*y.size()[:-1], 5], device=y.device)], dim=-1)

        mask0 = torch.where(y == 0, 1, 0)
        mask1 = torch.where(y == 1, 1, 0)
        mask2 = torch.where(y == 2, 1, 0)
        mask4 = torch.where(y == 4, 1, 0)
        y = torch.cat([mask0, mask1, mask2, mask4], dim=1).to(torch.float32)

        return x, y

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        seg = self.net(x)
        return seg, y

    def training_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss, per_ch_score = self.criterion(scores, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice', 100*(1-loss))
        self.log('train_dice_BG', per_ch_score[0])
        self.log('train_dice_NCR/NET', per_ch_score[1])
        self.log('train_dice_ED', per_ch_score[2])
        self.log('train_dice_ET', per_ch_score[3])
        return loss

    def validation_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss, per_ch_score = self.criterion(scores, labels)
        self.log("val_loss", loss, batch_size=labels.shape[0], on_epoch=True, on_step=False)
        self.log('val_dice', 100 * (1 - loss))
        self.log('val_dice_BG', per_ch_score[0])
        self.log('val_dice_NCR/NET', per_ch_score[1])
        self.log('val_dice_ED', per_ch_score[2])
        self.log('val_dice_ET', per_ch_score[3])
        return loss


# TODO complete
class Unet3D(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = self.build_decoder()

    def build_decoder(self):
        pass

    def forward(self, x):
        # TODO check
        x, skip = self.encoder(x)
        out = self.decoder(x, skip)
        return out


class Encoder3D(nn.Module):
    def __init__(self, depth: int = 4, in_channels: int = 4):
        super(Encoder3D, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.encoder, self.out_list = self.build_encoder()

    def build_encoder(self):
        blocks = []
        in_c = self.in_channels
        out_c = self.in_channels
        out_list = []
        for i in range(self.depth):
            if i != 0:
                out_list.append(out_c)
            blocks.append(Convolution(
                spatial_dims=3,
                in_channels=in_c,
                out_channels=out_c,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                dropout=0.1,
                norm=("instance", {"normalized_shape": out_c}),
            ))
            in_c = out_c
            out_c *= 2

        return blocks, out_list

    def forward(self, x):
        skip = []
        for b in self.encoder:
            x = b(x)
            skip.append(x)
        skip.reverse()
        return x, skip



