import trimesh
from pytorch3d.structures import Meshes
import torch
import numpy as np

mesh_s = trimesh.load('../pancreas_mean.ply')
mesh_t = trimesh.load('../pancreas_001_bas_reg.ply')


vertice_s = np.array(mesh_s.vertices)
vertice_t = np.array(mesh_t.vertices)

face_s = np.array(mesh_s.faces)
face_t = np.array(mesh_t.faces)

vertice_s = torch.from_numpy(vertice_s)
face_s = torch.from_numpy(face_s)
vertice_t = torch.from_numpy(vertice_t)
face_t = torch.from_numpy(face_t)

print(vertice_s.shape, face_s.shape, vertice_t.shape, face_t.shape)

from H2 import deformEnergy

a, b, c = deformEnergy(vertice_s, vertice_t, face_s)

print(a.shape)
print(b.shape)
print(c.shape)
