import sys
import torch
import numpy as np
import torchio as tio
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


sys.path.append(f"/home/doronser/workspace/")
from shared_utils.models import BaseModel  # noqa: E402

sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402
from monai.networks.blocks import Convolution  # noqa: E402
from monai.networks.nets import UNet as Monai_UNet  # noqa: E402


class ClassifierModel(BaseModel):
    def __init__(self, net: nn.Module, criterion=nn.CrossEntropyLoss(), optimizer_params=None, scheduler_params=None):
        super().__init__(net, criterion, optimizer_params, scheduler_params)
        self.acc = Accuracy(task="multiclass", num_classes=self.net.num_classes, top_k=1)

    def prepare_batch(self, batch):
        x = torch.cat([batch['t1'][tio.DATA], batch['t1ce'][tio.DATA],
                       batch['t2'][tio.DATA], batch['flair'][tio.DATA]], dim=1)

        if x.size()[-1] != 160:  # pad to even dimension
            x = torch.cat([x, torch.zeros([*x.size()[:-1], 5], device=x.device)], dim=-1)

        y = batch['label']
        return x, y

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        scores = self.net(x)
        return scores, y

    def training_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss = self.criterion(scores, labels)
        acc = self.acc(scores, labels)
        self.log_dict(dict(train_loss=loss, train_acc=acc), batch_size=labels.size()[0], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss = self.criterion(scores, labels)
        acc = self.acc(scores, labels)
        self.log_dict(dict(val_loss=loss, val_acc=acc), batch_size=labels.size()[0], on_epoch=True, on_step=False)
        return loss


class SegModel(BaseModel):
    """Pytorch Lightning module to use for BraTS18 segmentation

    -=BraTS18 Label Mapping=-
    WT = Whole Tumor = NCR/NET+ED+ET
    TC = Tumor Core = NCR/NET+ET
    ET = Enhancing Tumor
    """
    def __init__(self, net: nn.Module, criterion=DiceLoss(classes=4), optimizer_params=None, scheduler_params=None):
        super().__init__(net, criterion, optimizer_params, scheduler_params)

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


class EncoderClassifier(nn.Module):
    def __init__(self, depth: int = 5, in_channels: int = 4, num_classes: int = 10, dropout: int = 0.3):
        super().__init__()
        self.flat_size = 144000
        self.dropout = dropout
        self.num_classes = num_classes
        self.encoder = Encoder3D(depth=depth, in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.flat_size, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=64, out_features=self.num_classes)
        )

    def forward(self, x):
        x, _ = self.encoder(x)
        x = x.flatten(1)
        return self.classifier(x)


class Unet3D(nn.Module):
    def __init__(self, depth: int = 5, in_channels: int = 4):
        super().__init__()
        self.encoder = Encoder3D(depth=depth, in_channels=in_channels)
        self.decoder = Decoder3D(depth=depth, in_channels=self.encoder.out_size, out_channels=in_channels,
                                 skip_sizes=sorted(self.encoder.skip_sizes, reverse=True))
        self.unet = nn.Sequential(self.encoder, self.decoder)

    def __repr__(self):
        return self.unet.__repr__()

    def forward(self, x):
        return self.unet(x)


class Decoder3D(nn.Module):
    def __init__(self, depth: int = 5, in_channels: int = 64, out_channels: int = 4, skip_sizes: list = ()):
        super(Decoder3D, self).__init__()
        self.depth = depth
        self.skip_sizes = skip_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.decoder = self.build_decoder()

    def __repr__(self):
        return self.decoder.__repr__()

    def build_decoder(self):
        blocks = []
        in_c = self.in_channels
        out_c = self.skip_sizes[0] // 2
        for i in range(self.depth - 2):
            in_c += self.skip_sizes[i]
            blocks.append(Convolution(
                spatial_dims=3,
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1,
                strides=2,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                dropout=0.1,
                norm="instance",
                output_padding=1,
                is_transposed=True
            ))
            in_c = out_c
            if out_c != self.out_channels:
                out_c //= 2
        # last block is only convTranspose3D
        blocks.append(Convolution(spatial_dims=3,
                                  in_channels=in_c+self.skip_sizes[-1],
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  strides=2,
                                  padding=1,
                                  output_padding=1,
                                  is_transposed=True,
                                  act=None, norm=None, dropout=None
                                  ))
        return nn.ModuleList(blocks)

    def forward(self, x):
        out, skip = x
        for dec_block, skip in zip(self.decoder, skip):
            out = torch.cat([out, skip], dim=1)
            out = dec_block(out)
        return out


class Encoder3D(nn.Module):
    def __init__(self, depth: int = 5, in_channels: int = 4):
        """U-Net 3D encoder.
        Has *depth* downsampling blocks, where number channels is doubled for every block starting block #2.
        Each block is a monai convolution block: conv3D -> instance-norm -> dropout -> PReLU activation

        :param depth: determines the number of downsampling blocks
        :param in_channels: number of input channels
        """
        super(Encoder3D, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.encoder, out_list = self.build_encoder()
        self.skip_sizes = out_list[:-1]
        self.out_size = out_list[-1]

    def __repr__(self):
        return self.encoder.__repr__()

    def build_encoder(self):
        blocks = []
        in_c = self.in_channels
        out_c = self.in_channels
        out_list = []
        for i in range(self.depth):
            out_list.append(out_c)
            blocks.append(Convolution(
                spatial_dims=3,
                kernel_size=3,
                in_channels=in_c,
                out_channels=out_c,
                padding=1,
                strides=2 if i<self.depth-1 else 1,
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                dropout=0.1,
                norm="instance",
            ))
            in_c = out_c
            out_c *= 2

        return nn.ModuleList(blocks), out_list

    def forward(self, x):
        skip = []
        for b in self.encoder:
            x = b(x)
            skip.append(x)
        skip.pop()  # remove last output
        skip.reverse()
        return x, skip



