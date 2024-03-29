import os
import sys
import torch
import torchio as tio
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


sys.path.append(f"/home/{os.getlogin()}/workspace/")
from .utils import linear_warmup_decay  # noqa: E402
from .losses import BarlowTwinsLoss  # noqa: E402

sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402
from monai.networks.blocks import Convolution  # noqa: E402


class BaseModel(pl.LightningModule):
    """Template pytorch-lightning wrapper for vanilla pytorch nn.Modules"""
    def __init__(self, net: nn.Module, criterion, optimizer_params=None, scheduler_params=None):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def configure_optimizers(self):
        """Configures an optimizer and scheduler based on dict params"""
        DEFAULT_OPTIMIZER = dict(name='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

        if self.optimizer_params is None:
            self.optimizer_params = DEFAULT_OPTIMIZER
        optimizer_class = getattr(torch.optim, self.optimizer_params.pop('name'))
        optimizer = optimizer_class(self.parameters(), **self.optimizer_params)

        if self.scheduler_params is None:
            return optimizer
        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_params.pop('name'))
            scheduler = scheduler_class(optimizer, **self.scheduler_params)
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def prepare_batch(self, batch):
        raise NotImplementedError

    def infer_batch(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss = self.criterion(scores, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels = self.infer_batch(batch)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss, batch_size=labels.shape[0], on_epoch=True, on_step=False)
        return loss


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
        self.log_dict(dict(train_loss=loss, train_acc=acc), batch_size=labels.size()[0], prog_bar=True)
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
        self.log('train_loss', loss, prog_bar=True, batch_size=labels.shape[0])
        self.log('train_dice', 100*(1-loss), batch_size=labels.shape[0])
        self.log('train_dice_BG', per_ch_score[0], batch_size=labels.shape[0])
        self.log('train_dice_NCR/NET', per_ch_score[1], batch_size=labels.shape[0])
        self.log('train_dice_ED', per_ch_score[2], batch_size=labels.shape[0])
        self.log('train_dice_ET', per_ch_score[3], batch_size=labels.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        scores, labels = self.infer_batch(batch)
        loss, per_ch_score = self.criterion(scores, labels)
        self.log("val_loss", loss, batch_size=labels.shape[0], on_epoch=True, on_step=False)
        self.log('val_dice', 100 * (1 - loss), batch_size=labels.shape[0])
        self.log('val_dice_BG', per_ch_score[0], batch_size=labels.shape[0])
        self.log('val_dice_NCR/NET', per_ch_score[1], batch_size=labels.shape[0])
        self.log('val_dice_ED', per_ch_score[2], batch_size=labels.shape[0])
        self.log('val_dice_ET', per_ch_score[3], batch_size=labels.shape[0])
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
    def __init__(self, depth: int = 5, in_channels: int = 4, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder = Encoder3D(depth=depth, in_channels=in_channels)
        else:
            self.encoder = encoder
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
                strides=2 if i < self.depth-1 else 1,
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


class ProjectionHead(nn.Module):
    # def __init__(self, input_dim=768, hidden_dim=512, output_dim=64):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64, avg_pool_size=(15, 15, 10)):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=avg_pool_size)
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            # nn.BatchNorm1d(hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        # x_pool = x.mean(dim=1, keepdim=False)
        x_pool = self.avg_pool(x)
        x_flat = x_pool.flatten(1)
        return self.projection_head(x_flat)


class BarlowTwins(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=64,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
    ):
        super().__init__()
        self.encoder = encoder
        # self.projection_head = ProjectionHead(input_dim=z_dim, hidden_dim=z_dim, output_dim=z_dim)
        self.projection_head = ProjectionHead()  # TODO: do not init with default values
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        x1, x2 = batch
        x1 = torch.cat([x1['t1'][tio.DATA], x1['t1ce'][tio.DATA], x1['t2'][tio.DATA], x1['flair'][tio.DATA]], dim=1)
        x2 = torch.cat([x2['t1'][tio.DATA], x2['t1ce'][tio.DATA], x2['t2'][tio.DATA], x2['flair'][tio.DATA]], dim=1)
        if x1.size()[-1] != 160:  # pad to even dimension
            x1 = torch.cat([x1, torch.zeros([*x1.size()[:-1], 5], device=x1.device)], dim=-1)
            x2 = torch.cat([x2, torch.zeros([*x2.size()[:-1], 5], device=x2.device)], dim=-1)

        out1, _ = self.encoder(x1)
        out2, _ = self.encoder(x2)
        z1 = self.projection_head(out1)
        z2 = self.projection_head(out2)

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.shared_step(batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
