import torch
import torch.nn as nn

import os


from utils.registry import LOSS_REGISTRY

def get_neigh(face):
    torch.cat((face[:, [0, 1]], face[:, [0, 2]], face[:, [1, 2]]), dim=0)


def get_one_form(vert, face):
    nF = face.shape[0]
    alpha = torch.zeros((nF, 3, 2))
    v0, v1, v2 = vert.index_select(0, face[:,0]), vert.index_select(0, face[:,1]), vert.index_select(0, face[:,2])
    #print("v0 shape: {}".format(v0.shape))
    #print("v1 shape: {}".format(v1.shape))
    #print("v2 shape: {}".format(v2.shape))
    alpha[:,:,0] = v1 - v0
    alpha[:,:,1] = v2 - v0
    return alpha


def get_tangent_info(vert, face):
    nF = face.shape[0]

    alpha = torch.zeros((nF, 3, 2))
    v0, v1, v2 = vert.index_select(0, face[:,0]), vert.index_select(0, face[:,1]), vert.index_select(0, face[:,2])

    alpha[:,:,0] = v1 - v0
    alpha[:,:,1] = v2 - v0

    metric_tensor = torch.matmul(alpha.transpose(1,2),alpha)

    #A = (v1 - v2).norm(dim=1)
    #B = (v0 - v2).norm(dim=1)
    #C = (v0 - v1).norm(dim=1)
    #s = 0.5 * (A + B + C)
    #area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt()

    face_norm = 0.5 * torch.cross(v1-v0, v2-v0)
    area = face_norm.norm(dim=1)
    ### The output is the FACE-WISE tangent basis, metric tensor, area and norm vector 
    return alpha, metric_tensor, area, face_norm


def get_deformation_loss(vert_s, vert_t, face, para_shear, para_scale, para_bend):
    diff = vert_s - vert_t
    of_diff = get_one_form(diff, face)

    nF = face.shape[0]
    alpha = torch.zeros((nF, 3, 2))

    v0, v1, v2 = vert_s.index_select(0, face[:,0]), vert_s.index_select(0, face[:,1]), vert_s.index_select(0, face[:,2])
    alpha[:,:,0] = v1 - v0
    alpha[:,:,1] = v2 - v0

    mt = torch.matmul(alpha.transpose(1,2),alpha)

    fnorm =  0.5 * torch.cross(v1-v0, v2-v0)
    area = fnorm.norm(dim=1)
    #area = area.clamp_(min=1e-3)

    #alpha, mt, area, fnorm = get_tangent_info(vert_s, face)
    inv_mt = torch.inverse(mt)

    #print(alpha.shape)
    #print(inv_mt.shape)
    #print(diff.shape)

    dq_Uq_dh = torch.matmul(torch.matmul(alpha, inv_mt), of_diff.transpose(1,2))
    dq_Uq_dq = torch.matmul(torch.matmul(alpha, inv_mt), alpha.transpose(1,2))
    ### norm directon one-form
    of_diff_norm = of_diff - torch.matmul(dq_Uq_dq, of_diff)


    norm_loss = torch.matmul(torch.matmul(of_diff_norm, inv_mt), of_diff_norm.transpose(1,2))
    norm_loss = torch.einsum('nii->n', norm_loss).to('cuda')

    distort_mat = torch.matmul(of_diff.transpose(1,2), alpha) + torch.matmul(alpha.transpose(1,2), of_diff)
    distort_mat_normalize = torch.matmul(inv_mt, distort_mat).to('cuda')
    stretch_mat = torch.einsum('nii->n', distort_mat_normalize).to('cuda')
    shear_loss = (distort_mat_normalize.sum(dim=(1,2)) - stretch_mat)
    scale_loss = (stretch_mat)

    loss = (0.01 * para_bend * norm_loss + 0.1 * para_scale * scale_loss + para_shear * shear_loss)
    
    loss = torch.mean(loss.mul(area))

    return loss

@LOSS_REGISTRY.register()
class DeformLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DeformLoss, self).__init__()
        self.loss_weight = loss_weight
        self.raw_weights = nn.Parameter(torch.zeros(3))

    def forward(self, vert_s, vert_t, face):
        weights = torch.softmax(self.raw_weights, dim=0)
        deformation_loss = get_deformation_loss(vert_s, vert_t, face, weights[0], weights[1], weights[2])
        return self.loss_weight * deformation_loss



