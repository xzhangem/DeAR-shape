import numpy as np
import gpytoolbox as gp
import scipy.sparse as sp
from hessian_loss import grad_torch, hessian_vert

import trimesh 
import torch 


def stein_et_al_2018(V, F):
    # from Stein et al 2018
    # 9 entries per interior vertex; non-intrinsic

    # get counts
    n = V.shape[0]
    m = F.shape[0]
    int_vtx = np.setdiff1d(np.arange(n), gp.boundary_vertices(F))
    #print(gp.boundary_vertices(F).shape)
    i = int_vtx.shape[0]
    #print(i)

    # construct matrices according to the 2018 paper
    G = gp.grad(V, F)
    A = sp.diags(np.tile(0.5*gp.doublearea(V, F), 3))
    #print(np.tile(gp.doublearea(V, F),3).shape)
    #print(A.shape)
    Dsub = sp.hstack([G[:m, int_vtx], G[m:2*m, int_vtx], G[2*m:3*m, int_vtx]]) # Gx, Gy, Gz on interior vertices
    #print(Dsub.shape)
    D = sp.block_diag([Dsub, Dsub, Dsub])
    #print(D.shape, A.shape, G.shape)
    H = D.T@A@G # v -> face x, face y, face z -> area scaled -> divergence
    gxx = H[0:n, :]
    #print(np.sum(gxx - gxx.T))

    #print(H.shape)
    # reshape so that 9 adjacent entries are the hessian of that vertex
    s = np.reshape(np.reshape(np.arange(9*i), (9, -1)), (-1,), order='F')
    H = H[s, :]

    return H

mesh_s = trimesh.load('pancreas_mean.ply')
mesh_t = trimesh.load('pancreas_001.ply')

vertice_s = np.array(mesh_s.vertices)
vertice_t = np.array(mesh_t.vertices)

face_s = np.array(mesh_s.faces)
face_t = np.array(mesh_t.faces)

vertice_s = torch.from_numpy(vertice_s)
face_s = torch.from_numpy(face_s)
vertice_t = torch.from_numpy(vertice_t)
face_t = torch.from_numpy(face_t)
H = hessian_vert(vertice_s, vertice_t, face_s)
print(H)
'''
G_torch = grad_torch(torch_v, torch_f).numpy()
G = gp.grad(mesh_v, mesh_f).toarray()
#G = torch.from_numpy(G)
print(type(G))
print(type(G_torch))
print(np.array_equal(G, G_torch))
print(np.sum(np.abs(G - G_torch)))

#H = stein_et_al_2018(mesh_v, mesh_f)
#print(mesh_v.shape)
#print(mesh_f.shape)
#print(H.shape)
'''
