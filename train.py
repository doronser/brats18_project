import os
import sys
import argparse
import numpy as np
from datetime import datetime

import wandb
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


user = os.getlogin()
sys.path.append(f"/home/{user}/workspace/")
from shared_utils.io import yaml_read  # noqa: E402
from brats18_project.data_utils import Brats18DataModule  # noqa: E402
from brats18_project.models import SegModel, ClassifierModel  # noqa: E402


sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402
from torch.nn import CrossEntropyLoss  # noqa: E402
# import lib.medzoo as md  # noqa: E402
import brats18_project.models as model_zoo  # noqa: E402

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

    model_cls = getattr(model_zoo, cfg.model.name)
    net = model_cls(**cfg.model.kwargs)

    if cfg.loss.name == 'DiceLoss':
        model = SegModel(net=net, criterion=DiceLoss(classes=4),
                         optimizer_params=cfg.optimizer, scheduler_params=cfg.scheduler)
    elif cfg.loss.name == 'CrossEntropyLoss':
        model = ClassifierModel(net=net, criterion=nn.CrossEntropyLoss(),
                                optimizer_params=cfg.optimizer, scheduler_params=cfg.scheduler)
    else:
        raise ValueError("Unknown loss")

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
