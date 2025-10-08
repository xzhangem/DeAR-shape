# DeAR-shape
Implementation of "DeAR: Deformation-Aware Reguarlization Based Anatomical Shape Model".

### Getting Started
We highly recommend using Anaconda enviornment. The basic packages include PyTorch, PyKeops, pyg and PyTorch3d, other related tools are included in the pytorch3d.yaml file. You can import the environments via yaml file as follows:

`conda env create -f pytorch3d.yaml`

### Descriptions

The graphical abstract of the proposed DeAR shape metric is summarzied in the figure below
![image](https://github.com/xzhangem/DeAR-shape/blob/main/Figures/DeAR_figure.jpg)
DeAR shape metric can be used for optimization-based Riemannian statistical shape model (DeAR-OP) and deep learning-based displacement field shape model (DeAR-FUSS).

#### DeAR-OP
[DeAR-OP](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-OP) file contains the demo of surface registration & interpolation, deformation transfer and nonlinear statistical shape analysis.

#### DeAR-FUSS
DeAR-FUSS is based on [FUSS](https://github.com/NafieAmrani/FUSS) model, and included in File [DeAR-FUSS](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-FUSS). The data processing  

