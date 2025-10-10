# DeAR-shape
Implementation of "**DeAR**: **De**formation-**A**ware **R**eguarlization Based Anatomical Shape Model".

## Getting Started
We highly recommend using Anaconda enviornment. The basic packages include PyTorch, PyKeops, pyg and PyTorch3d, other related tools are included in the pytorch3d.yaml file. You can import the environments via yaml file as follows:

`conda env create -f pytorch3d.yaml`

This yaml file may contains too many dependencies beyond DeAR, but it contains the necessary dependencies to re-implement [FlowSSM](https://github.com/davecasp/flowssm) and [Mesh2SSM](https://github.com/iyerkrithika21/mesh2SSM_2023).

## Descriptions

The graphical abstract of the proposed DeAR shape metric is summarzied in the figure below
![image](https://github.com/xzhangem/DeAR-shape/blob/main/Figures/DeAR_figure.jpg)
DeAR shape metric can be used for optimization-based Riemannian statistical shape model ([DeAR-OP](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-OP)) and deep learning-based displacement field shape model ([DeAR-FUSS](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-FUSS)).

## DeAR-OP
[DeAR-OP](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-OP) file contains the demo of [surface registration/reconstruction](https://github.com/xzhangem/DeAR-shape/blob/main/DeAR-OP/dear_test.py) & interpolation, deformation transfer and nonlinear statistical shape analysis.

## DeAR-FUSS
DeAR-FUSS is based on [FUSS](https://github.com/NafieAmrani/FUSS) model, and included in File [DeAR-FUSS](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-FUSS). The data processing takes the following steps: (1) Make data file with filename data (Take ./data by default); (2) For different kinds of anatomical surfaces, we suggest using its name as the file name, and the mesh data are in off file. The name of each mesh is "organ name_sample_num.off". Take pancreas data as example: 

```
├── data
   ├── pancreas
        ├── off
           ├── pancreas_001.off
```
Then under [DeAR-FUSS](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-FUSS) file, run the script using pancreas dataset as follow:

`python preprocess.py --data_root ./data/pancreas/ --n_eig 200`

### DeAR-FUSS training 
Run the following script to train model on pancreas dataset:

`python train.py --opt options/train/pancreas.yaml `

### DeAR-FUSS for downstream tasks
Also with pancreas dataset, we release the trained DeAR-FUSS model [here](https://github.com/xzhangem/DeAR-shape/blob/main/DeAR-FUSS/experiments/fuss_pancreas_dear/models/final.pth).

For surface reconstruction (registration) with geometrical metric:

`python compare_test.py`

For interpolation experiment, run the script:

`python shape_interpolate.py`

For deformation transfer experiment, run the script:

`python shape_explorate.py`

The experiments results are stored in [results](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-FUSS/results) file. 

## Datasets:
We use the public datasets [Pancreas](http://medicaldecathlon.com/), [Liver](http://medicaldecathlon.com/), [Thyrold](https://medshapenet.ikim.nrw/), and [Femur](https://www.sciencedirect.com/science/article/pii/S1361841518304882). The processed pancreas data can be available at [Shapeworks](https://sciinstitute.github.io/ShapeWorks/latest/), and the number of Femur data can be found at the appendix of [FCM](https://www.sciencedirect.com/science/article/pii/S1361841521002243). 

## Acknowledgements
The implementation of [DeAR-OP](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-OP) is adapted from [H2_SurfaceMatch](https://github.com/emmanuel-hartman/H2_SurfaceMatch), and implementation of [DeAR-FUSS](https://github.com/xzhangem/DeAR-shape/tree/main/DeAR-FUSS) is adpated from [FUSS](https://github.com/NafieAmrani/FUSS). 

## License
This repo is licensed under MIT licence.



