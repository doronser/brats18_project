import pandas as pd
import numpy as np


def get_mri_indices(mri) -> (int, int, int):
    """get (x,y,z) indices of best channels to visualize given MRI image

    :param mri: 3D MRI scan
    :return:
    """
    nz_indices = mri.nonzero()
    df = pd.DataFrame(nz_indices, columns=['x', 'y', 'z'])
    x_dy = df.groupby('x')['y'].agg(np.ptp)
    x_idx = x_dy.idxmax()

    y_dz = df.groupby('y')['z'].agg(np.ptp)
    y_idx = y_dz.idxmax()

    z_dx = df.groupby('z')['x'].agg(np.ptp)
    z_idx = z_dx.idxmax()
    return x_idx, y_idx, z_idx
