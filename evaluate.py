import sys
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict

import torchio as tio
import monai.networks.nets as model_zoo

sys.path.append(f"/home/doronser/workspace/")
from brats18_project.models import SegModel  # noqa: E402
from brats18_project.data_utils import Brats18DataModule  # noqa: E402

sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402

# for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_wandb_model(wandb_path: str):
    api = wandb.Api()
    run = api.run(wandb_path)
    cfg = EasyDict(run.config)
    run_name = run.name.replace(' ', '_')
    ckpts_dir = Path(run.config['ckpt_path'])
    latest_ckpt = sorted([x for x in (ckpts_dir / run_name).glob('*')])[-1]
    model_cls = getattr(model_zoo, cfg.model.name)
    net = model_cls(**cfg.model.kwargs)
    model = SegModel.load_from_checkpoint(latest_ckpt, net=net, criterion=DiceLoss(classes=4),
                                          optimizer_params=cfg.optimizer, scheduler_params=cfg.scheduler)
    return model, cfg


def calc_batch_metrics(model: torch.nn.Module, batch: dict) -> (float, float, float):
    """calculate WT,TC and ET dice scores for a batch of size 1

    :param model: segmentor model
    :param batch: batch dictionary obtained from dataloader
    :return: WT, TC, ET dice scores for given MRI scan
    """
    # generate raw predictions for BG, NCR/NET, ED and ET
    with torch.inference_mode():
        seg_pred, seg_label = model.infer_batch(batch)

    # convert: BG, NCR/NET, ED, ET -> WT, ET, TC
    WT = seg_pred[:, 1:, :, :]
    TC = torch.cat([seg_pred[:, 1:2, :, :], seg_pred[:, 3:4, :, :]], dim=1)
    pred_soft_wt, _ = WT.max(dim=1)
    pred_soft_et = seg_pred[:, 3, :, :]
    pred_soft_tc, _ = TC.max(dim=1)

    label_hard = seg_label.argmax(dim=1)
    label_wt = label_hard.clamp(0, 1)
    label_et = torch.where(label_hard == 3, 1, 0).to(int)
    label_tc = torch.where((label_hard == 1) | (label_hard == 3), 1, 0).to(int)

    dice = DiceLoss(classes=1)
    _, dice_wt = dice(pred_soft_wt.unsqueeze(0), label_wt.unsqueeze(0))
    wt = dice_wt.item()

    _, dice_tc = dice(pred_soft_tc.unsqueeze(0), label_tc.unsqueeze(0))
    tc = dice_tc.item()

    _, dice_et = dice(pred_soft_et.unsqueeze(0), label_et.unsqueeze(0))
    et = dice_et.item()
    return wt, tc, et


if __name__ == '__main__':
    task_id = sys.argv[1]
    print("getting model")
    wandb_path = f"bio-vision-lab/intro2dl_brats2018/{task_id}"
    model, cfg = get_wandb_model(wandb_path)
    model.eval()

    print("getting data")
    dm = Brats18DataModule(cfg.data)
    dm.setup()
    dl = dm.val_dataloader()
    ds = dl.dataset.dataset
    ds.set_transform(tio.Compose([]))

    subj_metrics = []
    for batch in tqdm(dl, desc='calculating metrics'):
        wt, tc, et = calc_batch_metrics(model, batch)
        subj_metrics.append(dict(subj_name=batch['subj_name'][0], dice_wt=wt, dice_tc=tc, dice_et=et))

    metrics_df = pd.DataFrame(subj_metrics)
    t = wandb.Table(dataframe=metrics_df)
    with wandb.init(project='intro2dl_brats2018', id=task_id, resume="allow") as run:
        run.log(data={'metrics_df': t})

    print("done!")
