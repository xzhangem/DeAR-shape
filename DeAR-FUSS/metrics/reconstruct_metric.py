import trimesh, os
from tqdm import tqdm
import numpy as np

from utils.misc import plot_with_std
from utils.tensor_util import to_numpy
from pytorch3d.loss import chamfer_distance
from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_reconstruction(dataloader_test, deformed_testing_shapes, logger, device, output_path, template):
    surface_distance = SurfaceDistance()

    surf_dist_list = []
    cham_dist_list = []
    s2s_dist_list = []
    print(deformed_testing_shapes.shape)

    label_save_path = output_path + '/' + 'label_dir/'
    recon_save_path = output_path + '/' + 'recon_dir/'

    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
        print("label dir set up")
    if not os.path.exists(recon_save_path):
        os.makedirs(recon_save_path)
        print("recon dir set up")

    for index, test_data in enumerate(dataloader_test):
        original_verts = (test_data['verts'].to(device) * test_data['face_area'].to(device)).float().squeeze()
        #print(original_verts.shape)
        original_mesh = trimesh.Trimesh(vertices=to_numpy(original_verts), faces=to_numpy(test_data['faces']))
        save_name = str(index) + '.ply'
        original_mesh.export(label_save_path + save_name)
        
        deformed_verts = deformed_testing_shapes[index].to(device).float()
        #print(deformed_verts.shape)
        deformed_mesh = trimesh.Trimesh(vertices=to_numpy(deformed_verts), faces=to_numpy(template['faces']))
        deformed_mesh = trimesh.smoothing.filter_humphrey(deformed_mesh, alpha=0.1, beta=0.1, iterations=5)
        deformed_mesh.export(recon_save_path + save_name)

        surf_dist = surface_distance(original_mesh, deformed_mesh)[0]
        surf_dist_list.append(surf_dist)
        #print(type(original_verts.unsqueeze(0)))
        #print(type(deformed_verts.unsqueeze(0)))

        loss, _ = chamfer_distance(original_verts.unsqueeze(0), deformed_verts.unsqueeze(0), point_reduction=None, batch_reduction=None)
        cham_dist = 0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1))
        cham_dist_list.append(to_numpy(cham_dist))

        c = trimesh.proximity.ProximityQuery(original_mesh)
        p2mDist = c.signed_distance(to_numpy(deformed_verts))
        p2mDist = np.abs(p2mDist)
        p2mDist = np.mean(p2mDist)
        s2s_dist_list.append(p2mDist)


    surf_dist_mean = np.mean(surf_dist_list)
    surf_dist_std = np.std(surf_dist_list)
    cham_dist_mean = np.mean(cham_dist_list)
    cham_dist_std = np.std(cham_dist_list)
    s2s_dist_mean = np.mean(s2s_dist_list)
    s2s_dist_std = np.std(s2s_dist_list)

    logger.info(
            f'Point-to-surface distance is {surf_dist_mean:.4f} +/- {surf_dist_std:.4f}')

    logger.info(
            f'Chamfer distance is {cham_dist_mean:.4f} +/- {cham_dist_std:.4f}')

    logger.info(
            f'Surface-to-surface distance is {s2s_dist_mean:.4f} +/- {s2s_dist_std:.4f}')


class SurfaceDistance():
    """This class calculates the symmetric vertex to surface distance of two
    trimesh meshes.
    """

    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Args:
          A: trimesh mesh
          B: trimesh mesh
        """
        _, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
        _, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
        distance = .5 * np.array(A_B_dist).mean() + .5 * \
            np.array(B_A_dist).mean()

        return np.array([distance])
