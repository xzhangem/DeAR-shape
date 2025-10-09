import os
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from networks.diffusion_network import DiffusionNet
from networks.edgeconv_network import ResnetECPos
from networks.permutation_network import Similarity
from datasets.shape_dataset import OnePairDataset
from utils.tensor_util import to_device, to_numpy, to_tensor
from utils.shape_util import write_off
from utils.fmap_util import fmap2pointmap, nn_query
from models.fuss_model import compute_deformation
import trimesh 
from pytorch3d.loss import chamfer_distance


from datasets import build_dataloader, build_dataset
from utils.options import dict2str, parse_options

from enr.H2 import *
from H2_ivp import stepforward, H2InitialValueProblem
from H2_param import H2Parameterized
from H2_match import H2StandardIterative
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from enr.DDG import computeBoundary
from torch.autograd import grad

use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32


def H2ParamDeformationTransfer(VS, VT, VNS, F,a0,a1,b1,c1,d1,a2, geod1_params, geod2_params):
    print("Calculating Deformation Geodesic")
    geod1,F0=H2Parameterized([VS,F],[VT,F],a0,a1,b1,c1,d1,a2,geod1_params)
    print("Calculating Transfer Geodesic")
    geod2,F0=H2Parameterized([VS,F],[VNS,F],a0,a1,b1,c1,d1,a2,geod2_params)

    print("Performing Parallel Transport Using Schild's Ladder")
    F_init=torch.from_numpy(F).to(dtype=torch.long, device=torchdeviceId)
    X0=torch.from_numpy(geod1[1,...]).to(dtype=torchdtype, device=torchdeviceId)
    for i in range(0,geod2.shape[0]-1):
        A0=torch.from_numpy(geod2[i,...]).to(dtype=torchdtype, device=torchdeviceId)
        A1=torch.from_numpy(geod2[i+1,...]).to(dtype=torchdtype, device=torchdeviceId)
        P1=(A1+X0)/2
        X1,cost1=stepforward(A0,P1,a0,a1,b1,c1,d1,a2,F_init)
        X0=torch.from_numpy(X1).to(dtype=torchdtype, device=torchdeviceId)

    N=geod1.shape[0]
    print("Calculating Transfered Deformation")
    Ngeod1,F_init=H2InitialValueProblem(VNS,(N-1)*(X0.cpu().numpy()-VNS),N,a0,a1,b1,c1,d1,a2,F0)
    return geod1,geod2,Ngeod1,F


def FUSSDeformationTransfer(source, target, new_source, geod1, geod2, a0=1, a1=100, b1=10, c1=1, d1=1, a2=200):
    F0 = source[1]
    print("Performing Parallel Transport Using Schild's Ladder")
    F_init=torch.from_numpy(F0).to(dtype=torch.long, device=torchdeviceId)
    X0=torch.from_numpy(geod1[1,...]).to(dtype=torchdtype, device=torchdeviceId)
    print(geod1.shape)
    print(X0.shape)
    for i in range(0,geod2.shape[0]-1):
        A0=torch.from_numpy(geod2[i,...]).to(dtype=torchdtype, device=torchdeviceId)
        A1=torch.from_numpy(geod2[i+1,...]).to(dtype=torchdtype, device=torchdeviceId)
        P1=(A1+X0)/2
        X1,cost1=stepforward(A0,P1,a0,a1,b1,c1,d1,a2,F_init)
        X0=torch.from_numpy(X1).to(dtype=torchdtype, device=torchdeviceId)

    N=geod1.shape[0]
    N1=geod2.shape[0]
    print("Calculating Transfered Deformation")
    Ngeod1,F_1=H2InitialValueProblem(geod2[N1-1], (X0.cpu().numpy()-geod2[N1-1]),N,a0,a1,b1,c1,d1,a2,F0)

    return Ngeod1,F_1.cpu().numpy()

def H2UnparamDeformationTransfer(source, target, new_source, a0,a1,b1,c1,d1,a2, geod1_params,geod2_params, parameterizedDeform=False, parameterizedTransfer=False, match_newsource=False):
    print("Calculating Deformation Geodesic")
    if(parameterizedDeform):
        geod1,F0=H2Parameterized(source,target,a0,a1,b1,c1,d1,a2,geod1_params)
    else:
        geod1,F0=H2StandardIterative(source,target,a0,a1,b1,c1,d1,a2,geod1_params)

    print("Calculating Transfer Geodesic")
    if(parameterizedTransfer):
        geod2,F0=H2Parameterized(source,new_source,a0,a1,b1,c1,d1,a2,geod2_params)
    else:
        geod2,F0=H2StandardIterative(source,new_source,a0,a1,b1,c1,d1,a2,geod2_params)

    print("Performing Parallel Transport Using Schild's Ladder")
    F_init=torch.from_numpy(F0).to(dtype=torch.long, device=torchdeviceId)
    X0=torch.from_numpy(geod1[1]).to(dtype=torchdtype, device=torchdeviceId)
    for i in range(0,geod2.shape[0]-1):
        A0=torch.from_numpy(geod2[i]).to(dtype=torchdtype, device=torchdeviceId)
        A1=torch.from_numpy(geod2[i+1]).to(dtype=torchdtype, device=torchdeviceId)
        P1=(A1+X0)/2
        X1,cost1=stepforward(A0,P1,a0,a1,b1,c1,d1,a2,F_init)
        X0=torch.from_numpy(X1).to(dtype=torchdtype, device=torchdeviceId)

    N=geod1.shape[0]
    N1=geod2.shape[0]
    print("Calculating Transfered Deformation")
    Ngeod1,F_1=H2InitialValueProblem(geod2[N1-1], N*(X0.cpu().numpy()-geod2[N1-1]),N,a0,a1,b1,c1,d1,a2,F0)

    if match_newsource:
        print("Calculating Transfered Deformation on New Source mesh structure")
        Ngeod2,F_2=H2StandardIterative(new_source,[Ngeod1[N-1],F_1.cpu().numpy()],a0,a1,b1,c1,d1,a2,geod1_params)
        return geod1,geod2,Ngeod1,Ngeod2,F_1.cpu().numpy(),F_2

    return geod1,geod2,Ngeod1,F_1.cpu().numpy()

@torch.no_grad()
def compute_permutation_matrix(feat_x, feat_y, permutation):
    feat_x = F.normalize(feat_x, dim=-1, p=2)
    feat_y = F.normalize(feat_y, dim=-1, p=2)
    similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

    Pxy = permutation(similarity)
    Pyx = permutation(similarity.transpose(1, 2))

    return Pxy, Pyx


@torch.no_grad()
def compute_displacement(vert_x, vert_y, face_x, p2p_xy, interpolator, pose_timestep, device='cuda'):
    n_vert_x, n_vert_y = vert_x.shape[0], vert_y.shape[0]

    # construct time step
    step_size = 1 / (pose_timestep + 1)
    # [T+1, 1, 1]
    time_steps = step_size + torch.arange(0, 1, step_size,
                                            device=device, dtype=torch.float32).unsqueeze(1).unsqueeze(2)

    # [T+1, 1, 7]
    time_steps_up = time_steps * (torch.tensor([0, 0, 0, 0, 0, 0, 1],
                                                device=device, dtype=torch.float32)).unsqueeze(0).unsqueeze(1)

    # [1, n_vert_x, 7]
    vert_y_align = vert_y[p2p_xy]
    inputs = torch.cat((
        vert_x, vert_y_align - vert_x,
        torch.zeros(size=(n_vert_x, 1), device=device, dtype=torch.float32)
    ), dim=1).unsqueeze(0)
    # [T+1, n_vert_x, 7]
    inputs = inputs + time_steps_up

    # [n_vert_x, 3, Tp]
    displacements = torch.zeros(size=(inputs.shape[0], inputs.shape[1], 3), device=device, dtype=torch.float32)
    for i in range(inputs.shape[0]):
        displacements[i] = interpolator(inputs[i].unsqueeze(0), face_x.unsqueeze(0)).squeeze(0)

    vert_x_pred_arr = vert_x.unsqueeze(0) + displacements * time_steps
    vert_x_pred_arr = vert_x_pred_arr.permute([1, 2, 0]).contiguous()  # [n_vert_x, 3, T+1]

    return vert_x_pred_arr, displacements * time_steps


def FUSS_displace(dataloader, feature_extractor, permutation, interpolator):
    data = next(iter(dataloader))
    data_x, data_y = to_device(data['first'], 'cuda'), to_device(data['second'], 'cuda')
    assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'
    evecs_x = data_x['evecs'].squeeze()
    evecs_y = data_y['evecs'].squeeze()
    evecs_trans_x = data_x['evecs_trans'].squeeze()
    evecs_trans_y = data_y['evecs_trans'].squeeze()
    Lx, Ly = data_x['L'].squeeze(), data_x['L'].squeeze()
    with torch.no_grad():
        # extract feature
        feat_x = feature_extractor(data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = feature_extractor(data_y['verts'], data_y['faces'])  # [B, Ny, C]
        # compute permutation matrices
        Pxy, Pyx = compute_permutation_matrix(feat_x, feat_y, permutation)  # [B, Nx, Ny], [B, Ny, Nx]
        
        if non_isometric:
            p2p_yx = nn_query(feat_x.squeeze(0), feat_y.squeeze(0))
            p2p_xy = nn_query(feat_y.squeeze(0), feat_x.squeeze(0))
        else:
            Pxy, Pyx = Pxy.squeeze(0), Pyx.squeeze(0)
            Cxy = evecs_trans_y @ (Pyx @ evecs_x)
            Cyx = evecs_trans_x @ (Pxy @ evecs_y)
            # convert functional map to point-to-point map
            p2p_yx = fmap2pointmap(Cxy, evecs_x, evecs_y)
            p2p_xy = fmap2pointmap(Cyx, evecs_y, evecs_x)

        vert_x, vert_y = data_x['verts'].squeeze(0), data_y['verts'].squeeze(0)
        face_x, face_y = data_x['faces'].squeeze(0), data_y['faces'].squeeze(0)

        # from shape x to shape y [n_vert_x, 3, Tp]
        vert_x_pred_arr, dis_x = compute_displacement(vert_x, vert_y, face_x, p2p_xy, interpolator, pose_timestep)
        vert_x_pred_arr = vert_x_pred_arr.squeeze(-1)

        # from shape y to shape x
        vert_y_pred_arr, dis_y = compute_displacement(vert_y, vert_x, face_y, p2p_yx, interpolator, pose_timestep)
        vert_y_pred_arr = vert_y_pred_arr.squeeze(-1)

    return vert_x_pred_arr, vert_y_pred_arr, dis_x, dis_y

if __name__ == '__main__':
    # FAUST
    data_root = './data/pancreas/'
    prefix = 'pancreas_'
    network_path = './experiments/fuss_pancreas_dear/models/final.pth'
    result_root = 'results/pancreas_deform_trans/'
    first_iter_range = range(80, 90)
    iter_range = range(90, 100)
    n_iter = 1000
    non_isometric = False

    # specify pose step and shape step
    pose_timestep = 6
    shape_timestep = 3

    os.makedirs(result_root, exist_ok=True)

    feature_extractor = DiffusionNet(in_channels=128, out_channels=384,
                                     input_type='wks', cache_dir=os.path.join(data_root, 'diffusion')).cuda()
    permutation = Similarity(tau=0.07, hard=True).cuda()
    interpolator = ResnetECPos(c_dim=3, dim=7, hidden_dim=128, use_mlp=False).cuda()

    state_dict = torch.load(network_path)
    feature_extractor.load_state_dict(state_dict['networks']['feature_extractor'])
    interpolator.load_state_dict(state_dict['networks']['interpolator'])
    feature_extractor.eval()
    interpolator.eval()
    print('Load pretrained networks')


    num_evecs = 40
    pancreas_dir = os.listdir(data_root + 'off/')
    print(pancreas_dir)
    template_name = data_root + 'off/pancreas_001.off'
    fisrt_pbar = tqdm(first_iter_range, leave=False)
    pbar = tqdm(iter_range)
    cham_dist_list = []
    s2s_dist_list = []
    deformed_training_shapes = []
    deformed_testing_shapes = []
    tic = time.time()
    for j in range(1):
        for i in range(1): 
            source_shape_index = '001'
            target_shape_index = '087'
            newsource_shape_index = '001'
            source_shape = data_root + 'off/' + 'pancreas_' + source_shape_index + '.off'
            target_shape = data_root + 'off/' + 'pancreas_' + target_shape_index + '.off'
            newsource_shape = data_root + 'off/' + 'pancreas_' + newsource_shape_index + '.off'

            st_dataset = OnePairDataset(source_shape, target_shape, num_evecs=num_evecs)
            st_dataloader = DataLoader(st_dataset, batch_size=1, shuffle=False)

            sns_dataset = OnePairDataset(source_shape, newsource_shape, num_evecs=num_evecs)
            sns_dataloader = DataLoader(sns_dataset, batch_size=1, shuffle=False)

            vert_st_arr, vert_ts_arr, dis_st, dis_ts = FUSS_displace(st_dataloader, feature_extractor, permutation, interpolator)

            vert_sns_arr, vert_nss_arr, dis_sns, dis_nss = FUSS_displace(sns_dataloader, feature_extractor, permutation, interpolator)

            tri_source = trimesh.load(source_shape)
            tri_target = trimesh.load(target_shape)
            tri_newsource = trimesh.load(newsource_shape)
            tri_source = [np.array(tri_source.vertices), np.array(tri_source.faces)]
            tri_target = [np.array(tri_target.vertices), np.array(tri_target.faces)]
            tri_newsource = [np.array(tri_newsource.vertices), np.array(tri_newsource.faces)]

            geod, F = FUSSDeformationTransfer(tri_source, tri_target, tri_newsource, to_numpy(dis_st), to_numpy(dis_sns), a0=1, a1=100, b1=10, c1=1, d1=1, a2=200)

            print(geod.shape)
            print(F.shape)

            for k in range(geod.shape[0]):
                vert_exp = geod[k,...]
                save_name = result_root + 'explore_' + source_shape_index + '_' + target_shape_index + '_' + newsource_shape_index + '_' + str(k) + '.ply'
                trimesh.Trimesh(vertices=vert_exp, faces=F).export(save_name)


