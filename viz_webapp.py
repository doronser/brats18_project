import numpy as np
import torchio as tio
import streamlit as st
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt


@st.experimental_memo
def load_subject(subj_dir: Union[str, Path]):
    """Load a BraTS18 subject from a directory

    :param subj_dir: pathlib.Path or str path of subject
    :return: torchio.Subject of all MRI images + labels
    """
    if type(subj_dir) == str:
        subj_dir = Path(subj_dir)
    subj_files = sorted([str(x) for x in subj_dir.iterdir()])
    subj_obj = tio.Subject(
        name=subj_dir.stem,
        flair=tio.ScalarImage(subj_files[0]),
        seg=tio.LabelMap(subj_files[1]),
        t1=tio.ScalarImage(subj_files[2]),
        t1ce=tio.ScalarImage(subj_files[3]),
        t2=tio.ScalarImage(subj_files[4]),
    )
    return subj_obj


def plot_mri_slice(mri_slice: np.ndarray, seg_slice: Union[None, np.ndarray] = None,
                   seg_alpha: Union[None, float] = None, title=""):
    assert mri_slice.ndim == 2
    fig, ax = plt.subplots()
    ax.imshow(mri_slice, cmap='gray')
    if seg_slice is not None:
        ax.imshow(seg_slice, cmap='hot', alpha=seg_alpha)
    ax.set_title(title)
    plt.axis(False)
    return fig


st.set_page_config(layout="wide")
st.title("BraTS 2018 Data Visualization")

base_dir = Path("/raid/data/users/doronser/")
hgg_subjects_dirs = sorted([subj_dir for subj_dir in (base_dir / "MICCAI_BraTS_2018_Data_Training/HGG").iterdir()])
lgg_subjects_dirs = sorted([subj_dir for subj_dir in (base_dir / "MICCAI_BraTS_2018_Data_Training/LGG").iterdir()])

col1, col2, col3 = st.columns(3)
gg = col1.selectbox("Glioma Grade", ["Low", "High"])
subj_dirs = hgg_subjects_dirs if gg == "Low" else lgg_subjects_dirs
subj_dir = col2.selectbox("Subject ID", subj_dirs, format_func=lambda x: x.stem)
modal = col3.selectbox("Modality", ["t1", "t1ce", "t2", "flair", "seg"], format_func=lambda x: x.capitalize())
if modal != "seg":
    plot_seg = col1.checkbox("Overlay segmentation?")
    alpha = col2.slider("Alpha:", min_value=0., max_value=1., step=0.1, value=0.3, disabled=not plot_seg)

if subj_dir is not None and modal is not None:
    col3.write(subj_dir)
    subj = load_subject(subj_dir)
    vol = subj[modal].numpy()[0]
    seg = subj['seg'].numpy()[0]

    col1, col2, col3 = st.columns(3)
    x_ch = col1.slider("X Channel", 0, vol.shape[0]-1)
    y_ch = col2.slider("Y Channel", 0, vol.shape[1]-1)
    z_ch = col3.slider("Z Channel", 0, vol.shape[2]-1)

    x_seg = None
    y_seg = None
    z_seg = None
    plt_alpha = None
    if modal != 'seg' and plot_seg:
        x_seg = seg[x_ch, :, :]
        y_seg = seg[:, y_ch, :]
        z_seg = seg[:, :, z_ch]
        plt_alpha = alpha

    x_fig = plot_mri_slice(vol[x_ch, :, :], x_seg, plt_alpha, title=f"X Channel #{x_ch}")
    y_fig = plot_mri_slice(vol[:, y_ch, :], y_seg, plt_alpha, title=f"Y Channel #{y_ch}")
    z_fig = plot_mri_slice(vol[:, :, z_ch], z_seg, plt_alpha, title=f"Z Channel #{z_ch}")
    col1.pyplot(x_fig)
    col2.pyplot(y_fig)
    col3.pyplot(z_fig)
