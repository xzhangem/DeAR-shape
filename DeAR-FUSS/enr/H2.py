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

        
def getPathEnergyH2(geod, a0, a1, b1, c1, d1, a2, F_sol, VperEdge=None,stepwise=False,curvature_term=0):
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




def getPathEnergyH2_bauer(geod,a0,a1,b1,c1,d1,a2,F_sol,VperEdge=None,stepwise=False,curvature_term=0):
    b1=(b1-a1)/8
    a1 = a1 / 4
    N=geod.shape[0]        
    diff=(geod[1:,:,:] - geod[:-1,:,:])    
    midpoints=geod[0:N-1,:,:]+diff/2
    #diff=diff*N
    enr=0
    step_enr=torch.zeros((N-1,1),dtype=torchdtype)
    alpha0=getMeshOneForms(geod[0],F_sol)
    g0=getSurfMetric(geod[0],F_sol)
    n0=getNormal(F_sol, geod[0])    
    for i in range(0,N-1):   
        dv=diff[i]        
        if a2>0 or a0>0:
            M=getVertAreas(geod[i],F_sol)
        if a2>0:
            L=getLaplacian(midpoints[i],F_sol) 
            L=L(dv)
            K=getGaussian(midpoints[i],F_sol,VperEdge)
            K=K(dv)
            if curvature_term == 0:
                NL = (batchDot(L, L)) / M#+ 0.2 * batchDot(K, K)) / M # 0.25
                #NL = batchDot(K, K) / M
            else:
                #curvature_coeff = curvatureCoeff(dv, F_sol, 0.0)
                #NL = curvature_coeff * batchDot(L, L) / M
                NL = (batchDot(L, L)) / M  #+  0.2 * batchDot(K, K)) / M # 0.25
                #NL = batchDot(K, K) / M
            enr+=a2*torch.sum(NL)*N
        if a1>0 or b1>0 or c1>0 or d1>0:
            alpha1=getMeshOneForms(geod[i+1],F_sol)
            g1=getSurfMetric(geod[i+1],F_sol)
            n1=getNormal(F_sol, geod[i+1])      
            xi=alpha1-alpha0
            dg=(g1-g0)
            dn=(n1-n0)
            enr+=getGabNorm(getMeshOneForms(midpoints[i],F_sol),xi,getSurfMetric(midpoints[i],F_sol),dg,dn,a1,b1,c1,d1)*N
            g0=g1
            n0=n1
            alpha0=alpha1
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

def getH2Norm(M,dv,a0,a1,b1,c1,d1,a2,F_sol):    
    b1=(b1-a1)/8
    M1=M+dv
    enr=0
    if a2>0 or a0>0:
        A=getVertAreas(M,F_sol)
    if a2>0:      
        L=getLaplacian(M,F_sol) 
        L=L(dv)
        NL=batchDot(L,L)/A
        enr+=a2*torch.sum(NL)
    if a1>0 or b1>0 or c1>0:
        alpha0=getMeshOneForms(M,F_sol)
        g0=getSurfMetric(M,F_sol)
        n0=getNormal(F_sol, M)
        alpha1=getMeshOneForms(M1,F_sol)
        g1=getSurfMetric(M1,F_sol)
        n1=getNormal(F_sol, M1)
        xi=alpha1-alpha0
        dg=(g1-g0)
        dn=(n1-n0)
        enr+=getGabNorm(alpha0,xi,g0,dg,dn,a1,b1,c1,d1)
    if a0>0:
        Ndv=A*batchDot(dv,dv)
        enr+=a0*torch.sum(Ndv)
    return enr


def getH2Metric(M,dv1,dv2,a0,a1,b1,c1,d1,a2,F_sol):    
    b1=(b1-a1)/8
    M1=M+dv1   
    M2=M+dv2   
    enr=0
    if a2>0 or a0>0:
        A=getVertAreas(M,F_sol)
    if a2>0:      
        L=getLaplacian(M,F_sol) 
        NL=batchDot(L(dv1),L(dv2))/A
        enr+=a2*torch.sum(NL)
    if a1>0 or b1>0 or c1>0:
        
        alpha0=getMeshOneForms(M,F_sol)
        g0=getSurfMetric(M,F_sol)
        n0=getNormal(F_sol, M)  
        
        
        alpha1=getMeshOneForms(M1,F_sol)
        g1=getSurfMetric(M1,F_sol)
        n1=getNormal(F_sol, M1)      
        dg1=(g1-g0)
        dn1=(n1-n0)
        xi1=alpha1-alpha0
        
        
        alpha2=getMeshOneForms(M2,F_sol)
        g2=getSurfMetric(M2,F_sol)
        n2=getNormal(F_sol, M2)      
        dg2=(g2-g0)
        dn2=(n2-n0)
        xi2=alpha2-alpha0
        
        enr+=getGabMetric(alpha0,xi1,xi2,g0,dg1,dg2,dn1,dn2,a1,b1,c1,d1)
    if a0>0:
        Ndv=A*batchDot(dv1,dv2)
        enr+=a0*torch.sum(Ndv)
    return enr

def getGabNorm(alpha,xi,g,dg,dn,a,b,c,d):
    n=g.shape[0]
    areas=torch.sqrt(torch.det(g)).to(dtype=torchdtype, device=torchdeviceId)
    ginv=torch.inverse(g)
    ginvdg=torch.matmul(ginv,dg)    
    A=0
    B=0
    C=0
    D=0
    if a>0:
        afunc = torch.einsum('bii->b', torch.matmul(ginvdg,ginvdg))
        A=a*torch.sum(afunc*areas)
    if a>0:        
        bfunc = torch.einsum('bii->b', ginvdg)
        bfunc = bfunc*bfunc
        B=b*torch.sum(bfunc*areas)
    if c>0:        
        cfunc = torch.einsum('bi,bi->b', dn,dn)
        C=c*torch.sum(cfunc*areas)    
    if d>0:
        xi_0=torch.matmul(torch.matmul(alpha,ginv),torch.matmul(xi.transpose(1,2),alpha)-torch.matmul(alpha.transpose(1,2),xi))
        dfunc = torch.einsum('bii->b',torch.matmul(xi_0,torch.matmul(ginv,xi_0.transpose(1,2))))      
        D=d*torch.sum(dfunc*areas)    
    return (A+B+C+D)    
        
    
def getGabMetric(alpha,xi1,xi2,g,dg1,dg2,dn1,dn2,a,b,c,d):
    n=g.shape[0]
    areas=torch.sqrt(torch.det(g)).to(dtype=torchdtype, device=torchdeviceId)
    ginv=torch.inverse(g)
    ginvdg1=torch.matmul(ginv,dg1) 
    ginvdg2=torch.matmul(ginv,dg2)    
    A=0
    B=0
    C=0
    D=0
    if a>0:
        afunc = torch.einsum('bii->b', torch.matmul(ginvdg1,ginvdg2))
        A=a*torch.sum(afunc*areas)
    if a>0:        
        bfunc1 = torch.einsum('bii->b', ginvdg1)
        bfunc2 = torch.einsum('bii->b', ginvdg2)
        bfunc = bfunc1*bfunc2
        B=b*torch.sum(bfunc*areas)
    if c>0:        
        cfunc = torch.einsum('bi,bi->b', dn1,dn2)
        C=c*torch.sum(cfunc*areas)        
    if d>0:
        xi1_0=torch.matmul(torch.matmul(alpha,ginv),torch.matmul(xi1.transpose(1,2),alpha)-torch.matmul(alpha.transpose(1,2),xi1))
        xi2_0=torch.matmul(torch.matmul(alpha,ginv),torch.matmul(xi2.transpose(1,2),alpha)-torch.matmul(alpha.transpose(1,2),xi2))
        dfunc = torch.einsum('bii->b',torch.matmul(xi1_0,torch.matmul(ginv,xi2_0.transpose(1,2))))      
        D=d*torch.sum(dfunc*areas) 
    return (A+B+C+D)    

##############################################################################################################################
#H2_Matching_Energies
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

def enr_match_H2_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, VperEdge=None, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1,curvature_coeff=0,**objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod):
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol,VperEdge,stepwise=False,curvature_term=curvature_coeff)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1])
        #E = weight_Gab*enr + weight_coef_dist_S*chamfer_dist(geod[0], VS) + weight_coef_dist_T*chamfer_dist(geod[N-1],VT)
        return E
    return energy

def enr_match_H2(VS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_T=1, weight_Gab=1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)   
    def energy(geod):
        geod=torch.cat((torch.unsqueeze(VS,dim=0),geod),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_T*dataloss_T(geod[N-1])
        return E
    return energy



def enr_match_H2_sym_w(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, Rho, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol,VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf_Weighted(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod,Rho):
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)
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



def enr_param_H2(left,right, F_sol,a0,a1,b1,c1,d1,a2):    
    def energy(mid):
        geod=torch.cat((left, mid,right),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)        
        return enr 
    return energy

def enr_param_H2Kmean(samples,F_sol,a0,a1,b1,c1,d1,a2):   
    N=samples.shape[0]
    n=samples.shape[1]
    def energy(mu,mid):
        geods=torch.cat((mu.repeat((N,1,1,1)), mid,samples.unsqueeze(dim=1)),dim=1).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)        
        enr=0
        for i in range(0,N):
            enr+=getPathEnergyH2(geods[i],a0,a1,b1,c1,d1,a2,F_sol)        
        return enr 
    return energy

def enr_unparam_H2Kmean(samples,Sample_Funs,F_Sol,Fun_Sol,a0,a1,b1,c1,d1,a2,weight_coef_dist_T=1, **objfun):   
    N=len(samples)
    K=VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_Ts=[lossVarifoldSurf(F_Sol, Fun_Sol, sample[0],sample[1], Sample_Funs[ind], K) for ind,sample in enumerate(samples)]
    
    
    def energy(mu,mid):
        geods=torch.cat((mu.repeat((N,1,1,1)), mid),dim=1).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)     
        enr=0
        n=geods.shape[1]
        for i in range(0,N):
            enr+=getPathEnergyH2(geods[i],a0,a1,b1,c1,d1,a2,F_Sol)+weight_coef_dist_T*dataloss_Ts[i](geods[i,n-1])  
        return enr 
    return energy

def enr_step_forward(M0,M1,a0,a1,b1,c1,d1,a2,F):
    def energy(M2):
        M1dot=M1-M0
        M2dot=M2-M1
        qM1 = M1.clone().requires_grad_(True)
        sys=2*getFlatMap(M0,M1-M0,F,a0,a1,b1,c1,d1,a2)-2*getFlatMap(M1,M2-M1,F,a0,a1,b1,c1,d1,a2)+grad(getH2Norm(qM1,M2dot,a0,a1,b1,c1,d1,a2,F), qM1, create_graph=True)[0]
        return (sys**2).sum()
    return energy
    
def getFlatMap(M,V,F,a0,a1,b1,c1,d1,a2):
    B=torch.zeros((M.shape[0],3)).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    return grad(getH2Metric(M,V,B,a0,a1,b1,c1,d1,a2,F), B, create_graph=True)[0] 


def enr_match_H2_sym_coeff(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, geod, basis,
                         weight_coef_dist_S=1, weight_coef_dist_T=1, weight_Gab=1, a0=1, a1=1, b1=1, c1=1, d1=1, a2=1, **objfun):
    K = VKerenl(objfun['kernel_geom'], objfun['kernel_grass'], objfun['kernel_fun'], objfun['sig_geom'],
                objfun['sig_grass'], objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)

    def energy(X):
        geod_torch = geod + torch.einsum("ij, jkl-> ikl", X, basis)
        enr = getPathEnergyH2(geod_torch, a0, a1, b1, c1, d1, a2, F_sol)

        N = geod.shape[0]
        var1 = weight_coef_dist_S * dataloss_S(geod_torch[0])
        var2 = weight_coef_dist_T * dataloss_T(geod_torch[N - 1])
        #E = weight_Gab*enr + weight_coef_dist_S*chamfer_dist(geod_torch[0], VS) + weight_coef_dist_T*chamfer_dist(geod_torch[N-1],VT)
        E = weight_Gab * enr + var1 + var2
        return E

    return energy


def enr_match_H2_coeff(VS, VT, FT, FunT, F_sol, Fun_sol, geod, basis, weight_coef_dist_T=1, weight_Gab=1, a0=1, a1=1, b1=1, c1=1, d1=1, a2=1, **objfun):
    K = VKerenl(objfun['kernel_geom'], objfun['kernel_grass'], objfun['kernel_fun'], objfun['sig_geom'],
                objfun['sig_grass'], objfun['sig_fun'])
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)

    def energy(X):
        X=torch.cat((torch.unsqueeze(torch.zeros((X.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),X),dim=0).requires_grad_(True)
        geod_torch = geod + torch.einsum("ij, jkl-> ikl", X, basis)
        enr = getPathEnergyH2(geod_torch, a0, a1, b1, c1, d1, a2, F_sol)

        N = geod.shape[0]
        var2 = weight_coef_dist_T * dataloss_T(geod_torch[N - 1])
        E = weight_Gab * enr + var2
        return E
    return energy

def enr_param_H2_coeff( F_sol, geod, basis, a0=1, a1=1, b1=1, c1=1, d1=1, a2=1, **objfun):
    def energy(X):
        X=torch.cat((torch.unsqueeze(torch.zeros((X.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),X,torch.unsqueeze(torch.zeros((X.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0)),dim=0).requires_grad_(True)
        geod_torch = geod + torch.einsum("ij, jkl-> ikl", X, basis)
        return getPathEnergyH2(geod_torch, a0, a1, b1, c1, d1, a2, F_sol)
    return energy


def enr_step_forward(M0,M1,a0,a1,b1,c1,d1,a2,F):
    def energy(M2):
        M1dot=M1-M0
        M2dot=M2-M1
        qM1 = M1.clone().requires_grad_(True)
        sys=2*getFlatMap(M0,M1-M0,F,a0,a1,b1,c1,d1,a2)-2*getFlatMap(M1,M2-M1,F,a0,a1,b1,c1,d1,a2)+grad(getH2Norm(qM1,M2dot,a0,a1,b1,c1,d1,a2,F), qM1, create_graph=True)[0]
        return (sys**2).sum()
    return energy

def getFlatMap(M,V,F,a0,a1,b1,c1,d1,a2):
    B=torch.zeros((M.shape[0],3)).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    return grad(getH2Metric(M,V,B,a0,a1,b1,c1,d1,a2,F), B, create_graph=True)[0]

def enr_step_forward_coeff(M0,M1,a0,a1,b1,c1,d1,a2,F,temp,basis):
    def energy(M2):
        M1dot=M1-M0
        M2dot=M2-M1
        qM1 = M1.clone().requires_grad_(True)
        sys=2*getFlatMap_coeff(M0,M1-M0,F,a0,a1,b1,c1,d1,a2,temp,basis)-2*getFlatMap_coeff(M1,M2-M1,F,a0,a1,b1,c1,d1,a2,temp,basis)+grad(getH2Norm(temp + torch.einsum("j, jkl-> kl", qM1, basis),torch.einsum("j, jkl-> kl", M2dot, basis),a0,a1,b1,c1,d1,a2,F), qM1, create_graph=True)[0]
        return (sys**2).sum()
    return energy

def getFlatMap_coeff(M,V,F,a0,a1,b1,c1,d1,a2,temp,basis):
    B=torch.zeros((M.shape[0])).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    return grad(getH2Metric(temp + torch.einsum("j, jkl-> kl", M, basis),torch.einsum("j, jkl-> kl", V, basis),torch.einsum("j, jkl-> kl", B, basis),a0,a1,b1,c1,d1,a2,F), B, create_graph=True)[0]
