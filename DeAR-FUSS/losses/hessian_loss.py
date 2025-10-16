import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY

def grad_torch(V, F, dim=3):
    """
    PyTorch 版本的离散梯度算子计算
    输入:
        V: 顶点坐标 (n, 3) 或 (n, 2) 的 Tensor
        F: 三角形面片索引 (m, 3) 的 LongTensor
        dim: 维度 (2 或 3)
    输出:
        G: 稀疏梯度矩阵 (3m, n) 或 (2m, n)
    """
    device = V.device
    
    
    m, n = F.shape[0], V.shape[0]
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]  # 三角形顶点索引
    
    # 计算边向量
    v21 = V[i2] - V[i1]  # 边 v1→v2
    v02 = V[i0] - V[i2]  # 边 v2→v0
    v10 = V[i1] - V[i0]  # 边 v0→v1
    
    # 计算法向量和双倍面积
    n_vec = torch.cross(v21, v02, dim=1)
    dblA = torch.norm(n_vec, dim=1)
    u = n_vec / dblA.unsqueeze(1)  # 单位法向量
    
    # 计算梯度基向量 (eperp)
    def compute_eperp(u, v, dblA):
        cross_uv = torch.cross(u, v, dim=1)
        norm_v = torch.norm(v, dim=1, keepdim=True)
        norm_cross = torch.norm(cross_uv, dim=1, keepdim=True)
        return cross_uv * (norm_v / (dblA.unsqueeze(1) * norm_cross))
    
    eperp10 = compute_eperp(u, v10, dblA)
    eperp02 = compute_eperp(u, v02, dblA)
    
    # 构造稀疏矩阵的索引和值
    Find = torch.arange(m, device=device)  # 三角形索引 [0, 1, ..., m-1]
    
    # 行索引 (I)
    #I_x = Find
    #I_y = Find + m
    #I_z = Find + 2 * m
    #I = torch.cat([I_x.repeat(4), I_y.repeat(4), I_z.repeat(4)])  # 每分量重复4次
    I = torch.cat([
        Find, Find, Find, Find,
        Find + F.shape[0], Find + F.shape[0], Find + F.shape[0], Find + F.shape[0],
        Find + 2 * F.shape[0], Find + 2 * F.shape[0], Find + 2 * F.shape[0], Find + 2 * F.shape[0]])
    
    # 列索引 (J)
    J_base = torch.cat([F[:, 1], F[:, 0], F[:, 2], F[:, 0]])
    J = torch.cat([J_base, J_base, J_base])  # 重复3次（x, y, z分量）
    
    vals = torch.cat([
        eperp02[:, 0], -eperp02[:, 0], eperp10[:, 0], -eperp10[:, 0],
        eperp02[:, 1], -eperp02[:, 1], eperp10[:, 1], -eperp10[:, 1],
        eperp02[:, 2], -eperp02[:, 2], eperp10[:, 2], -eperp10[:, 2]])
    # 非零值 (vals)
    #vals_xy = torch.cat([eperp02, -eperp02, eperp10, -eperp10], dim=0)
    #vals = vals_xy.flatten()  # 展平为1D
    
    # 构造稀疏矩阵
    G = torch.sparse_coo_tensor(
        indices=torch.stack([I, J]),
        values=vals,
        size=(3 * m, n),
        device=device
    ).to_dense()
    #print(vals.shape)
    #print(G.shape)
    
    
    return G


def hessian_vert(vert_s, vert_t, face):
    diff = vert_t - vert_s

    n = vert_s.shape[0]
    m = face.shape[0]
    G = grad_torch(vert_s, face)
    v0, v1, v2 = vert_s.index_select(0, face[:,0]), vert_s.index_select(0, face[:,1]), vert_s.index_select(0, face[:,2])
    face_norm = torch.cross(v1-v0, v2-v0)
    area = 0.5 * face_norm.norm(dim=1)
    area = torch.diag(area)

    Gx = G[:m, :]
    Gy = G[m:2*m, :]
    Gz = G[2*m:3*m, :]
    Gxx_df = Gx.T @ area @ Gx @ diff
    Gyy_df = Gy.T @ area @ Gy @ diff
    Gzz_df = Gz.T @ area @ Gz @ diff
    Gxy_df = Gx.T @ area @ Gy @ diff
    Gxz_df = Gx.T @ area @ Gz @ diff
    Gyz_df = Gy.T @ area @ Gz @ diff
    Hdf = torch.mean(Gxx_df**2 + Gyy_df**2 + Gzz_df**2 + 2*(Gxy_df**2 + Gxz_df**2 + Gyz_df**2))
    '''
    A = torch.diag(double_area.repeat(3))
    print(A.shape)
    Dsub = torch.cat([G[:m, :], G[m:2*m, :], G[2*m:3*m, :]], dim=1)
    D = torch.block_diag(Dsub, Dsub, Dsub)
    print(D.shape)
    H = D.T @ A @ G
    '''
    return Hdf


@LOSS_REGISTRY.register()
class HessianLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(HessianLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, vert_s, vert_t, face):
        hessian_loss = hessian_vert(vert_s, vert_t, face) * self.loss_weight
        return hessian_loss
