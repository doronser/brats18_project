import sys
import torch
import wandb
from pathlib import Path
from easydict import EasyDict
import monai.networks.nets as model_zoo

sys.path.append(f"/home/doronser/workspace/")
from brats18_project.models import SegModel  # noqa: E402

sys.path.append(f"/home/doronser/workspace/MedicalZooPytorch/")
from lib.losses3D import DiceLoss  # noqa: E402


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
