import os
import sys
import argparse
from pathlib import Path

import numpy as np
from copy import deepcopy
from datetime import datetime

import wandb
import torch
from easydict import EasyDict
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


user = os.getlogin()
sys.path.append(f"/home/{user}")
from shared_utils.io import yaml_read  # noqa: E402
from brats18_project.data_utils import Brats18DataModule  # noqa: E402
from brats18_project.models import SegModel, ClassifierModel  # noqa: E402


sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402
from torch.nn import CrossEntropyLoss  # noqa: E402
# import lib.medzoo as md  # noqa: E402
from brats18_project.models import Encoder3D, Unet3D, BarlowTwins  # noqa: E402

# for reproducibility
torch.manual_seed(42)
np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, help="path to YAML config file")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    assert os.path.exists(args.cfg), f"cfg file {args.cfg} does not exist!"
    cfg = yaml_read(args.cfg, easy_dict=True)

    data_module = Brats18DataModule(cfg.data)

    api = wandb.Api()
    run = api.run(cfg.model.wandb_path)
    run_name = run.name.replace(' ', '_')
    ckpts_dir = Path(run.config['ckpt_path'])
    latest_ckpt = sorted([x for x in (ckpts_dir / run_name).glob('*')])[-1]
    print("using ckpt:", latest_ckpt)

    net = BarlowTwins.load_from_checkpoint(latest_ckpt, encoder=Encoder3D(), num_training_samples=7, batch_size=7)

    unet = Unet3D()
    unet.encoder = deepcopy(net.encoder)
    # unet = Unet3D(encoder=deepcopy(net.encoder))
    model = SegModel(net=unet, criterion=DiceLoss(classes=4), optimizer_params=cfg.optimizer,
                     scheduler_params=cfg.scheduler)

    if args.debug:
        trainer = pl.Trainer(fast_dev_run=True)
    else:
        # define pl loggers and callbacks
        time_suffix = datetime.now().strftime(r"%y%m%d_%H%M%S")
        wandb_logger = WandbLogger(project="intro2dl_brats2018", name=f"{cfg.name}_{time_suffix}")
        wandb.config.update(cfg)
        ckpt_clbk = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg.ckpt_path, f"{cfg.name}_{time_suffix}"),
                                                 filename=cfg.name + '_epoch{epoch:02d}', auto_insert_metric_name=False,
                                                 save_top_k=-1, every_n_epochs=10)
        lr_clbk = pl.callbacks.LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[cfg.trainer.gpu],
            max_epochs=cfg.trainer.epochs,
            callbacks=[ckpt_clbk, lr_clbk],
            logger=wandb_logger
        )
    trainer.fit(model=model, datamodule=data_module)
