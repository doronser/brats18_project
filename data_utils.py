import os
import sys
import numpy as np
from easydict import EasyDict
from typing import Optional, Union, Tuple

import torch
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, default_collate

sys.path.append(f'/home/{os.getlogin()}/workspace')
from brats18_project.custom_transforms import BarlowTwinsTransform


def load_brats2018_data(base_path, prep=None, aug=None, train_ssl=False):
    """load BraTS2018 data as TorchIO SubjectsDataset

    :param base_path: path to BraTS2018 training data
    :param prep: preprocessing transform object
    :param aug: augmentations transform object
    :param train_ssl: bool parameter for setting the augmentations needed for training barlow twins
    :return: torchio.SubjectsDataset of entire dataset or train/val datasets when val_split is given
    """
    # get data
    subjects = []
    for glioma_grade in ['LGG', 'HGG']:
        for subj_name in sorted(os.listdir(f"{base_path}/{glioma_grade}/")):
            subjects.append(tio.Subject(
                t1=tio.ScalarImage(os.path.join(base_path, glioma_grade, subj_name, f"{subj_name}_t1.nii.gz")),
                t1ce=tio.ScalarImage(os.path.join(base_path, glioma_grade, subj_name, f"{subj_name}_t1ce.nii.gz")),
                t2=tio.ScalarImage(os.path.join(base_path, glioma_grade, subj_name, f"{subj_name}_t2.nii.gz")),
                flair=tio.ScalarImage(os.path.join(base_path, glioma_grade, subj_name, f"{subj_name}_flair.nii.gz")),
                seg=tio.LabelMap(os.path.join(base_path, glioma_grade, subj_name, f"{subj_name}_seg.nii.gz")),
                glioma_grade=glioma_grade,
                subj_name=subj_name,
                # label=0
            ))

    # add transformations
    transform = []
    if prep is not None:
        transform.append(prep)
    if aug is not None:
        transform.append(aug)

    if train_ssl:
        print("using BarlowTwinsTransform!")
        barlow_transform = BarlowTwinsTransform(tio.Compose(transform))
        return Brats18Dataset(subjects=subjects, transform=barlow_transform)
    else:
        print("Using default Transform!")
        return Brats18Dataset(subjects=subjects, transform=tio.Compose(transform))


class Brats18Dataset(tio.SubjectsDataset):
    def __init__(self, subjects, transform=None, rot=False):
        self.rot = rot
        super(Brats18Dataset, self).__init__(subjects, transform)

    def get_subject(self, subj_name, pad=True):
        for subj in self._subjects:
            if subj.subj_name == subj_name:
                if pad and subj['t1'].tensor.size()[-1] != 160:
                    for k in ['t1', 't1ce', 't2', 'flair', 'seg']:
                        v = subj[k].tensor
                        v = torch.cat([v, torch.zeros([*subj[k].tensor.size()[:-1], 5], device=v.device)], dim=-1)
                        subj[k].set_data(v)
                return subj
        print("No such subject!", subj_name)
        return None

    def __getitem__(self, idx):
        subj = super().__getitem__(idx)
        if self.rot:
            # apply random rotation and add label
            subj.label = np.random.randint(0, 4)
            for k in ['t1', 't1ce', 't2', 'flair']:
                v = subj[k].tensor
                v = v.rot90(subj.label, dims=(1, 2))
                subj[k].set_data(v)
        return subj


class Brats18DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: EasyDict):
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.val_split_size = data_cfg.val_split_size
        self.data_dir = data_cfg.dir
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers
        self.ssl = data_cfg.get('ssl', False)
        self.collate_fn = barlow_collate if self.ssl else None
        print("SSL=", self.ssl)
        self.preprocessing_transform, self.augmentation_transform = self.parse_cfg_transform(data_cfg)

    def setup(self, stage: Optional[str] = None):
        brats_ds = load_brats2018_data(self.data_dir, prep=self.preprocessing_transform,
                                       aug=self.augmentation_transform, train_ssl=self.ssl)
        num_val = int(self.val_split_size*len(brats_ds))
        num_train = len(brats_ds) - num_val
        self.train_set, self.val_set = torch.utils.data.random_split(brats_ds, lengths=[num_train, num_val])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.ssl, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.ssl, collate_fn=self.collate_fn)

    @staticmethod
    def parse_cfg_transform(yml_cfg: Union[dict, EasyDict]) -> Tuple[tio.Transform, None]:
        """Parse YAML config from plain text to TorchIO.Transform

        :param yml_cfg: YAML config dictionary
        :return: tuple of preprocessing transform and augmentation transform (torchio.Transform objects)
        """
        augs = yml_cfg.get('augmentations', None)
        pre_proc = yml_cfg.get('preprocessing', None)

        if pre_proc is not None:
            trans_list = []
            for trans, params in pre_proc.items():
                trans_func = getattr(tio, trans)
                trans_list.append(trans_func(**params))
            preprocessing = tio.Compose(trans_list)
        else:
            preprocessing = None

        if augs is not None:
            aug_list = []
            for aug, params in augs.items():
                aug_func = getattr(tio, aug)
                aug_list.append(aug_func(p=0.5, **params))
            augmentations = tio.Compose(aug_list)
        else:
            augmentations = None

        return preprocessing, augmentations


def barlow_collate(batch):
    """custom collate function to correctly stack augmented data for barlow twins training"""
    # print(f"{type(batch[0][0])=}")
    # print(f"{len(batch[0])=}")
    # print("BARLOW COLLATE 2!")
    x1_list = []
    x2_list = []
    for (x1, x2) in batch:
        for x in [x1, x2]:
            del x['seg']
            del x['subj_name']
            del x['glioma_grade']
        x1_list.append(x1)
        x2_list.append(x2)
    x1_batch = default_collate(x1_list)
    x2_batch = default_collate(x2_list)
    return x1_batch, x2_batch
