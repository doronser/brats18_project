import pandas as pd
import numpy as np
from functools import partial


def get_mri_indices(mri) -> (int, int, int):
    """get (x,y,z) indices of best channels to visualize given MRI image

    :param mri: 3D MRI scan
    :return:
    """
    nz_indices = mri.nonzero()
    df = pd.DataFrame(np.array(nz_indices).T, columns=['x', 'y', 'z'])
    x_dy = df.groupby('x')['y'].agg(np.ptp)
    x_idx = int(x_dy.idxmax())

    y_dz = df.groupby('y')['z'].agg(np.ptp)
    y_idx = int(y_dz.idxmax())

    z_dx = df.groupby('z')['x'].agg(np.ptp)
    z_idx = int(z_dx.idxmax())
    return x_idx, y_idx, z_idx


class BarlowTwinsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform(sample), self.transform(sample)


#  needed for BarlowTwins learning rate scheduler
def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


#  needed for BarlowTwins learning rate scheduler
def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)
