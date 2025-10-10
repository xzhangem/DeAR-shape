import sys
sys.path.append("../")

import torch
import numpy as np
from enr.DDG import *
from enr.varifold import *
from enr.regularizers import *
from torch.autograd import grad

from pytorch3d.loss import chamfer_distance

use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

##############################################################################################################################
#H2 Helper Functions
##############################################################################################################################



def deformEnergy(vert_s, vert_t, face): #a0, a1, b1, c1, d1, a2):
    v_num = vert_s.shape[0]
    f_num = face.shape[0]
    diff = vert_t - vert_s
    of_diff = getMeshOneForms(diff, face)

    alpha = getMeshOneForms(vert_s, face)
    mt = getSurfMetric(vert_s, face)
    n0 = getNormal(face, vert_s)

    inv_mt = torch.inverse(mt)
    Jacob = torch.matmul(torch.matmul(of_diff, inv_mt), alpha.transpose(1,2))
    JtJ = torch.matmul(Jacob.transpose(1,2), Jacob)
    I_JtJ = torch.eye(3).expand(f_num, 3, 3).to(dtype=torchdtype, device=torchdeviceId) - JtJ # + Jacob.transpose(1,2) + Jacob
    #I_JtJ = JtJ + Jacob.transpose(1,2) + Jacob

    dq_Uq_dh = torch.matmul(torch.matmul(alpha, inv_mt), of_diff.transpose(1,2))
    dq_Uq_dq = torch.matmul(torch.matmul(alpha, inv_mt), alpha.transpose(1,2))

    of_diff_norm = of_diff - torch.matmul(dq_Uq_dq, of_diff)

    # Bending
    norm_loss = torch.matmul(torch.matmul(of_diff_norm, inv_mt), of_diff_norm.transpose(1,2))
    norm_loss = torch.einsum('nii->n', norm_loss).to(dtype=torchdtype, device=torchdeviceId)

    # Distortion
    #print(of_diff.shape)
    #print(alpha.shape)
    distort_mat = torch.matmul(of_diff.transpose(1,2), alpha) + torch.matmul(alpha.transpose(1,2), of_diff)
    distort_mat_normalize = torch.matmul(inv_mt, distort_mat)
    #distort_mat_sq = torch.matmul(distort_mat, distort_mat)
    stretch_mat = torch.einsum('nii->n', distort_mat_normalize).to(dtype=torchdtype, device=torchdeviceId)
    distort_loss = torch.square(distort_mat_normalize.sum(dim=(1,2)) - stretch_mat) 
    #distort_loss = torch.square(torch.square(distort_mat_tr) / 6 - torch.einsum('nii->n', distort_mat_sq).to(dtype=torchdtype, device=torchdeviceId) / 2)
    # Stretching
    #stretch_mat = torch.matmul(torch.matmul(dq_Uq_dq.transpose(1,2), I_JtJ), dq_Uq_dq)
    #stretch_loss = torch.norm(stretch_mat, p='fro', dim=(1, 2))
    stretch_loss = (torch.square(stretch_mat))

    return norm_loss, distort_loss, stretch_loss

        
def getPathEnergyDeAR(geod, a0, a1, b1, c1, d1, a2, F_sol, VperEdge=None,stepwise=False):
    N = geod.shape[0]
    diff = (geod[1:,:,:] - geod[:-1,:,:])
    midpoints=geod[0:N-1,:,:]+diff/2
    step_enr=torch.zeros((N-1,1),dtype=torchdtype)
    enr=0
    for i in range(0,N-1):
        dv = diff[i]
        if a2>0 or a0>0:
            M = getVertAreas(geod[i], F_sol)
        if a2>0:
            L = getLaplacian(geod[i],F_sol)
            L = L(dv)
            NL = (batchDot(L, L)) / M
            enr += a2 * torch.sum(NL) * N

        norm_loss, distort_loss, strech_loss = deformEnergy(geod[i], geod[i+1], F_sol)
        g = getSurfMetric(geod[i],F_sol)
        area = torch.sqrt(torch.det(g)).to(dtype=torchdtype, device=torchdeviceId)

        if a1>0:
            enr += a1 * torch.sum(distort_loss.mul(area)) * N
        if b1>0:
            enr += b1 * torch.sum(strech_loss.mul(area)) * N
        if c1>0:
            enr += c1 * torch.sum(norm_loss.mul(area)) * N

        if a0>0:
            Ndv=M*batchDot(dv,dv)
            enr+=a0*torch.sum(Ndv)*N

        if stepwise:
            if i==0:
                step_enr[0]=enr
            else:
                step_enr[i]=enr-torch.sum(step_enr[0:i])

        if stepwise:
            return enr,step_enr

    return enr



def getDeARNorm(M, dv, a0,a1,b1,c1,d1,a2,F_sol):
    M1 = M + dv
    enr = 0
    A=getVertAreas(M,F_sol)
    if a2>0:
        L=getLaplacian(M,F_sol)
        L=L(dv)
        NL=batchDot(L,L)/A
        enr+=a2*torch.sum(NL)

    norm_loss, distort_loss, strech_loss = deformEnergy(M, M1, F_sol)

    if a1>0:
        enr += a1 * torch.sum(distort_loss.mul(A))
    if b1>0:
        enr += b1 * torch.sum(strech_loss.mul(A))
    if c1>0:
        enr += c1 * torch.sum(norm_loss.mul(A))

    if a0>0:
        Ndv=M*batchDot(dv,dv)
        enr+=a0*torch.sum(Ndv)*N
    return enr


def getDeARMetric(M,dv1,dv2,a0,a1,b1,c1,d1,a2,F_sol):
    M1=M+dv1
    M2=M+dv2
    enr=0
    f_num = F_sol.shape[0]
    A=getVertAreas(M,F_sol)
    if a2>0:
        L=getLaplacian(M,F_sol)
        NL=batchDot(L(dv1),L(dv2))/A
        enr+=a2*torch.sum(NL)
    if a1>0 or b1>0 or c1>0:
        alpha_0=getMeshOneForms(M,F_sol)
        mt_0 = getSurfMetric(M, F_sol)
        n0_0=getNormal(F_sol, M)
        inv_mt_0 = torch.inverse(mt_0)
        dq_Uq_dq_0 = torch.matmul(torch.matmul(alpha_0, inv_mt_0), alpha_0.transpose(1,2))

        ### dv1 & dv2 geometry
        of_diff_1 = getMeshOneForms(dv1, F_sol)
        of_diff_2 = getMeshOneForms(dv2, F_sol)

        dq_Uq_dh_1 = torch.matmul(torch.matmul(alpha_0, inv_mt_0), of_diff_1.transpose(1,2))
        dq_Uq_dh_2 = torch.matmul(torch.matmul(alpha_0, inv_mt_0), of_diff_2.transpose(1,2))

        of_diff_norm_1 = of_diff_1 - torch.matmul(dq_Uq_dq_0, of_diff_1)
        of_diff_norm_2 = of_diff_2 - torch.matmul(dq_Uq_dq_0, of_diff_2)

        ### bending metric
        norm_metric = torch.matmul(torch.matmul(of_diff_norm_1, inv_mt), of_diff_norm_2.transpose(1,2))
        norm_metric = torch.einsum('nii->n', norm_metric).to(dtype=torchdtype, device=torchdeviceId)

        ### distortion metric
        distort_mat_1 = torch.matmul(of_diff_1.transpose(1,2), alpha_0) + torch.matmul(alpha_0.transpose(1,2), of_diff_1)
        distort_mat_normalize_1 = torch.matmul(inv_mt_0, distort_mat_1)
        distort_mat_2 = torch.matmul(of_diff_2.transpose(1,2), alpha_0) + torch.matmul(alpha_0.transpose(1,2), of_diff_2)
        distort_mat_normalize_2 = torch.matmul(inv_mt_0, distort_mat_2)

        stretch_mat_1 = torch.einsum('nii->n', distort_mat_normalize_1).to(dtype=torchdtype, device=torchdeviceId)
        stretch_mat_2 = torch.einsum('nii->n', distort_mat_normalize_2).to(dtype=torchdtype, device=torchdeviceId)

        distortion_metric = (distort_mat_normalize_1.sum(dim=(1,2)) - stretch_mat_1) * (distort_mat_normalize_2.sum(dim=(1,2)) - stretch_mat_2)

        ### stretching metric
        stretch_metric = stretch_mat_1 * stretch_mat_2

        if a1>0:
            enr += a1 * torch.sum(distort_metric.mul(A))
        if b1>0:
            enr += b1 * torch.sum(strech_metric.mul(A))
        if c1>0:
            enr += c1 * torch.sum(norm_metric.mul(A))
    if a0>0:
        Ndv=A*batchDot(dv1,dv2)
        enr+=a0*torch.sum(Ndv)
        
    return enr




##############################################################################################################################
#DeAR_Matching_Energies
##############################################################################################################################

def chamfer_dist(vs, vt):
    if (torch.is_tensor(vs)==False):
        vs = torch.from_numpy(vs).to(dtype=torchdtype, device=torchdeviceId)
    if (torch.is_tensor(vt)==False):
        vt = torch.from_numpy(vt).to(dtype=torchdtype, device=torchdeviceId)
    if vs.dim() == 2:
        vs = vs.unsqueeze(0)
    if vt.dim() == 2:
        vt = vt.unsqueeze(0)
    loss, _ = chamfer_distance(vs, vt, point_reduction=None, batch_reduction=None)
    chamfer = (0.5 * (loss[0].sqrt().mean(dim=1) + loss[1].sqrt().mean(dim=1)))

    return chamfer 

def enr_match_DeAR_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, VperEdge=None, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1,curvature_coeff=0,**objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod):
        enr=getPathEnergyDeAR(geod,a0,a1,b1,c1,d1,a2,F_sol,VperEdge,stepwise=False)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1])
        #E = weight_Gab*enr + weight_coef_dist_S*chamfer_dist(geod[0], VS) + weight_coef_dist_T*chamfer_dist(geod[N-1],VT)
        return E
    return energy

def enr_match_DeAR(VS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_T=1, weight_Gab=1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)   
    def energy(geod):
        geod=torch.cat((torch.unsqueeze(VS,dim=0),geod),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergDeAR(geod,a0,a1,b1,c1,d1,a2,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_T*dataloss_T(geod[N-1])
        return E
    return energy



def enr_match_DeAR_sym_w(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, Rho, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol,VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf_Weighted(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod,Rho):
        enr=getPathEnergyDeAR(geod,a0,a1,b1,c1,d1,a2,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1],Rho)#torch.clamp(Rho,-.25,1.25)+.01*penalty(geod[N-1],F_sol, Rho)
        return E
    return energy


def enr_match_weight(VT,FT,FunT,V_sol,F_sol, Fun_sol, B_sol,weight_coef_dist_T=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_T = lossVarifoldSurf_Weighted(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(Rho):
        E=weight_coef_dist_T*dataloss_T(V_sol,Rho)#torch.clamp(Rho,-.25,1.25)
        return E
    return energy



def enr_param_DeAR(left,right, F_sol,a0,a1,b1,c1,d1,a2):    
    def energy(mid):
        geod=torch.cat((left, mid,right),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyDeAR(geod,a0,a1,b1,c1,d1,a2,F_sol)        
        return enr 
    return energy

def enr_param_DeARKmean(samples,F_sol,a0,a1,b1,c1,d1,a2):   
    N=samples.shape[0]
    n=samples.shape[1]
    def energy(mu,mid):
        geods=torch.cat((mu.repeat((N,1,1,1)), mid,samples.unsqueeze(dim=1)),dim=1).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)        
        enr=0
        for i in range(0,N):
            enr+=getPathEnergyDeAR(geods[i],a0,a1,b1,c1,d1,a2,F_sol)        
        return enr 
    return energy

def enr_unparam_DeARKmean(samples,Sample_Funs,F_Sol,Fun_Sol,a0,a1,b1,c1,d1,a2,weight_coef_dist_T=1, **objfun):   
    N=len(samples)
    K=VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_Ts=[lossVarifoldSurf(F_Sol, Fun_Sol, sample[0],sample[1], Sample_Funs[ind], K) for ind,sample in enumerate(samples)]
    
    
    def energy(mu,mid):
        geods=torch.cat((mu.repeat((N,1,1,1)), mid),dim=1).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)     
        enr=0
        n=geods.shape[1]
        for i in range(0,N):
            enr+=getPathEnergyDeAR(geods[i],a0,a1,b1,c1,d1,a2,F_Sol)+weight_coef_dist_T*dataloss_Ts[i](geods[i,n-1])  
        return enr 
    return energy

def getFlatMap(M,V,F,a0,a1,b1,c1,d1,a2):
    B=torch.zeros((M.shape[0],3)).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    return grad(getDeARMetric(M,V,B,a0,a1,b1,c1,d1,a2,F), B, create_graph=True)[0]


def enr_step_forward(M0,M1,a0,a1,b1,c1,d1,a2,F):
    def energy(M2):
        M1dot=M1-M0
        M2dot=M2-M1
        qM1 = M1.clone().requires_grad_(True)
        sys=2*getFlatMap(M0,M1-M0,F,a0,a1,b1,c1,d1,a2)-2*getFlatMap(M1,M2-M1,F,a0,a1,b1,c1,d1,a2)+grad(getDeARNorm(qM1,M2dot,a0,a1,b1,c1,d1,a2,F), qM1, create_graph=True)[0]
        return (sys**2).sum()
    return energy
    

def enr_step_forward(M0,M1,a0,a1,b1,c1,d1,a2,F):
    def energy(M2):
        M1dot=M1-M0
        M2dot=M2-M1
        qM1 = M1.clone().requires_grad_(True)
        sys=2*getFlatMap(M0,M1-M0,F,a0,a1,b1,c1,d1,a2)-2*getFlatMap(M1,M2-M1,F,a0,a1,b1,c1,d1,a2)+grad(getDeARNorm(qM1,M2dot,a0,a1,b1,c1,d1,a2,F), qM1, create_graph=True)[0]
        return (sys**2).sum()
    return energy


