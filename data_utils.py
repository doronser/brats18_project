import os
import sys
import numpy as np
from glob import glob

import torch
import torchio as tio
from torch.utils.data import Dataset


def load_brats2018_data(base_path):
    """load BraTS2018 data as TorchIO SubjectsDataset

    :param base_path: path to BraTS2018 training data
    :return: torchio.SubjectsDataset
    """
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
                subj_name=subj_name
            ))

    return tio.SubjectsDataset(subjects=subjects)

