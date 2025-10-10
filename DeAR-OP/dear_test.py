import sys
import numpy as np
import utils.input_output as io
import plotly.graph_objects as go
import os
import random
from dear_match import DeARStandardIterative, DeARMultiRes
import torch
import time

from pytorch3d.loss import chamfer_distance

from torch import nn
from pytorch3d.ops.points_alignment import corresponding_points_alignment, _apply_similarity_transform, iterative_closest_point


mesh_file = './data/pancreas_dataset/'
mesh_dir = os.listdir(mesh_file)
template_name = './data/pancreas_dataset/pancreas_001.ply'
[v_template, faces, Fun0] = io.loadData(template_name)
template = [np.array(v_template), np.array(faces)]
test_file_name = './dear_op_pancreas/'

if not os.path.exists(test_file_name):
    os.makedirs(test_file_name)

a0 = 1
a1 = 100 #200
b1 = 10   #0
c1 = 1 #200
d1 = 1   #1
a2 = 200 #200

class ICP(nn.Module):
    def __init__(self, correspondence):
        """ ICP alignment implementation in torch.
        """
        super(ICP, self).__init__()
        self.correspondence = correspondence

    def forward(self, from_vertices, to_vertices):
        full_source = from_vertices.clone()
        start_shape = full_source.shape

        # Apply correct initial alignment
        if self.correspondence:
            R, T, s = corresponding_points_alignment(
                from_vertices,
                to_vertices,
                weights=None,
                estimate_scale=False,
                allow_reflection=False,
            )

            from_vertices = _apply_similarity_transform(from_vertices, R, T, s)

        icp = iterative_closest_point(from_vertices, to_vertices, relative_rmse_thr=1e-5)
        from_vertices = icp.Xt

        assert start_shape == from_vertices.shape, "Shape mismatch"
        return from_vertices

icp = ICP(False)

param1_mul = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 10*2.0,'sig_geom': 20,\
                  'max_iter': 3000,'time_steps': 2, 'tri_unsample': True, 'index':0}

param2_mul = {'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 10*2.0,'sig_geom': 10,\
                  'max_iter': 3000,'time_steps': 2, 'tri_unsample': False, 'index':1}

param3_mul =  {'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 10*2.0,'sig_geom':10,\
                  'max_iter': 3000, 'time_steps': 2, 'tri_unsample': True, 'index':1}

param4_mul = {'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 10*2.0,'sig_geom': 4,\
                  'max_iter': 3000, 'time_steps': 2, 'tri_unsample': False, 'index':2}

param5_mul = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 10*2.0,'sig_geom':4,\
                  'max_iter': 3000, 'time_steps': 2, 'tri_unsample': False, 'index':2}

paramlist = [param1_mul, param2_mul, param3_mul, param4_mul, param5_mul]

chamfer_list = []

tic = time.time()
test_file_name = './dear_liver_113/'
for i in range(len(mesh_dir)):
    print("Processing sample of {}/{}".format(i+1, len(mesh_dir)))
    #template = [np.array(v_template), np.array(faces)]
    [V, F, Fun] = io.loadData(mesh_file + mesh_dir[i])
    target = [np.array(V), np.array(F)]
    geod, F0 = DeARMultiRes(template, target, a0,a1,b1,c1,d1,a2, 2, paramlist)
    result_v = icp(torch.Tensor(np.array(geod[-1])).unsqueeze(0), torch.Tensor(target[0]).unsqueeze(0))
    loss, _ = chamfer_distance(result_v, torch.Tensor(target[0]).unsqueeze(0), point_reduction=None, batch_reduction=None)
    chamfer_dist = (0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))).numpy()
    chamfer_list.append(chamfer_dist)
    io.saveData(file_name= test_file_name+mesh_dir[i].split('.')[0], extension='ply', V=result_v.squeeze(dim=0).cpu().numpy(), F=F0, Rho=None, color=None)

print(np.mean(chamfer_list))
print(np.std(chamfer_list))
toc = time.time()
print(toc - tic)
