import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd

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

    return vert_x_pred_arr



if __name__ == '__main__':
    # FAUST
    data_root = './data/pancreas/'
    mesh_info_file = os.path.join(data_root, 'mesh_info.csv')
    mesh_info = pd.read_csv(mesh_info_file)
    prefix = 'pancreas_'
    network_path = './experiments/fuss_pancreas_dear/models/final.pth'
    result_root = 'results/pancreas_metric_compare/'
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
    template_name = data_root + 'off/pancreas_001.off'
    fisrt_pbar = tqdm(first_iter_range, leave=False)
    pbar = tqdm(iter_range)
    cham_dist_list = []
    s2s_dist_list = []
    deformed_training_shapes = []
    deformed_testing_shapes = []
    tic = time.time()
    for j in range(1):
        for i in range(len(pancreas_dir)):
            #first_shape_index = f'{j:03d}'
            first_shape_index = '001'
            print(type(pancreas_dir[i]))
            second_shape_index = pancreas_dir[i].split('.')[0].split('_')[1]
            print(second_shape_index)
            first_shape = template_name
            second_shape = data_root + 'off/' + pancreas_dir[i]
            #first_shape = data_root + 'off/' + pancreas_dir[j] #os.path.join(data_root, 'off', f'{prefix}{first_shape_index}.off')
            #second_shape = data_root + 'off/' + pancreas_dir[i] #os.path.join(data_root, 'off', f'{prefix}{second_shape_index}.off')

            dataset = OnePairDataset(first_shape, second_shape, num_evecs=num_evecs)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # get data pair
            data = next(iter(dataloader))
            data_x, data_y = to_device(data['first'], 'cuda'), to_device(data['second'], 'cuda')
            assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'
            data_x_info = np.array(mesh_info[mesh_info['file_name'] == prefix + first_shape_index])[0]
            data_x_face_area = data_x_info[4]
            print(data_x_face_area)
            data_y_info = np.array(mesh_info[mesh_info['file_name'] == prefix + second_shape_index])[0]
            data_y_face_area = data_y_info[4]
            print(data_y_face_area)
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
            vert_x_pred_arr = compute_displacement(vert_x, vert_y, face_x, p2p_xy, interpolator, pose_timestep).squeeze(-1)
            vert_x_pred_arr = vert_x_pred_arr * data_y_face_area

            # from shape y to shape x
            vert_y_pred_arr = compute_displacement(vert_y, vert_x, face_y, p2p_yx, interpolator, pose_timestep).squeeze(-1)
            vert_y_pred_arr = vert_y_pred_arr * data_x_face_area

            # align
            vert_x_align = vert_x[p2p_yx]
            vert_y_align = vert_y[p2p_xy]

            #vert_x_star = compute_deformation(data_x, data_y, n_iter)
            vert_y_align = vert_x_pred_arr[..., -1]
            vert_y = vert_y * data_y_face_area


            if i < int(len(pancreas_dir) * 0.8):
                deformed_training_shapes.append(vert_y)#_align)
            else:
                deformed_testing_shapes.append(vert_y)

            save_name = result_root + pancreas_dir[i].split('.')[0] + '.ply'
            result_mesh = trimesh.Trimesh(vertices=to_numpy(vert_y_align), faces=to_numpy(face_x))
            loss, _ = chamfer_distance(vert_y_align.unsqueeze(0), vert_y.unsqueeze(0), point_reduction=None, batch_reduction=None)
            cham_dist = 0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))
            cham_dist_list.append(to_numpy(cham_dist))

            original_mesh = trimesh.Trimesh(vertices=to_numpy(vert_y), faces=to_numpy(face_y))
            c = trimesh.proximity.ProximityQuery(original_mesh)
            p2mDist = c.signed_distance(to_numpy(vert_y_align))
            p2mDist = np.abs(p2mDist)
            p2mDist = np.mean(p2mDist)
            s2s_dist_list.append(p2mDist)


            trimesh.Trimesh(vertices=to_numpy(vert_y_align), faces=to_numpy(face_x)).export(save_name)

    print("chamfer distance: {} +/- {}".format(np.mean(cham_dist_list), np.std(cham_dist_list)))
    print("surface to surface distance: {} +/- {}".format(np.mean(s2s_dist_list), np.std(s2s_dist_list)))

