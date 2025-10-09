import torch
import numpy as np
import scipy

use_cuda=1
torchdeviceId=torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype=torch.float32
np.int = np.int32
np.bool = np.bool_

##############################################################################################################################
# Discrete Differential Geometry Helper Functions
##############################################################################################################################


def batchDot(dv1,dv2):
    '''Parallel computation of batches of dot products.
    
    Input:
        - dv1 [Vxd torch tensor]
        - dv2 [Vxd torch tensor]
        
    Output:
        - tensor of dot products between corresponding rows of dv1 and dv2 [Vx1 torch tensor]
    '''

    return torch.einsum('bi,bi->b', dv1,dv2)


def getSurfMetric(V,F):
    '''Computation of the Riemannian metric evaluated at the faces of a triangulated surface.
    
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - g: Riemannian metric evaluated at each face of the triangulated surface [nFx2x2 torch tensor]
    '''

    # Number of faces
    nF = F.shape[0]

    # Preallocate tensor for Riemannian metric
    alpha = torch.zeros((nF,3,2)).to(dtype=torchdtype, device=torchdeviceId)   

    # Compute Riemannian metric at each face
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2]) 
    
    alpha[:,:,0]=V1-V0
    alpha[:,:,1]=V2-V0    

    return torch.matmul(alpha.transpose(1,2),alpha)

def getMeshOneForms(V,F):
    '''Computation of the Riemannian metric evaluated at the faces of a triangulated surface.
    
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - alpha: One form evaluated at each face of the triangulated surface [nFx3x2 torch tensor]
    '''

    # Number of faces
    nF = F.shape[0]

    # Preallocate tensor for Riemannian metric
    alpha = torch.zeros((nF,3,2)).to(dtype=torchdtype, device=torchdeviceId)   

    # Compute Riemannian metric at each face
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2]) 
    
    alpha[:,:,0]=V1-V0
    alpha[:,:,1]=V2-V0
    
    return alpha


def getLaplacian(V,F):   
    '''Computation of the mesh Laplacian operator of a triangulated surface evaluated at one of its tangent vectors h.
    
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - L: function that will evaluate the mesh Laplacian operator at a tangent vector to the surface [function]
    ''' 

    # Number of vertices and faces
    nV, nF = V.shape[0], F.shape[0]

    # Get x,y,z coordinates of each face
    face_coordinates = V[F]
    v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]

    # Compute the area of each face using Heron's formula
    A = (v1 - v2).norm(dim=1) 
    B = (v0 - v2).norm(dim=1) # lengths of each side of the faces
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C) # semi-perimeter
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt() # Apply Heron's formula and clamp areas of small faces for numerical stability

    # Compute cotangent expressions for the mesh Laplacian operator
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 2.0

    # Find indices of adjacent vertices in the triangulated surface (i.e., edge list between vertices)
    ii = F[:, [1, 2, 0]]
    jj = F[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, nF * 3)

    # Define function that evaluates the mesh Laplacian operator at one of the surface's tangent vectors
    def L(h):
        '''Function that evaluates the mesh Laplacian operator at a tangent vector to the surface.

        Input:
            - h: tangent vector to the triangulated surface [nVx3 torch tensor]

        Output:
            - Lh: mesh Laplacian operator of the triangulated surface applied to one its tangent vectors h [nVx3 torch tensor]
        '''

        # Compute difference between tangent vectors at adjacent vertices of the surface
        hdiff = h[idx[0]]-h[idx[1]]

        # Evaluate mesh Laplacian operator by multiplying cotangent expressions of the mesh Laplacian with hdiff
        values = (torch.stack([cot.view(-1)]*3, dim=1)*hdiff)

        # Sum expression over adjacent vertices for each coordinate
        Lh = torch.zeros((nV,3)).to(dtype=torchdtype, device=torchdeviceId)  
        Lh[:,0]=Lh[:,0].scatter_add(0,idx[1,:],values[:,0])
        Lh[:,1]=Lh[:,1].scatter_add(0,idx[1,:],values[:,1])
        Lh[:,2]=Lh[:,2].scatter_add(0,idx[1,:],values[:,2])    

        return Lh

    return L

def vertexOppositeEdge(F):
    '''Computation the opposite vertices of edge on each triangle mesh 
    Input:
        - F: faces of the triangulated surface [nFx3 ndarray]

    Output:
        - vertex_opposite_edge: opposite vertices of edges on each triangle mesh [(nFx3)x4 ndarray]
    '''
    nF = F.shape[0]
    nV = F.max() + 1

    # Find the opposite sides of the vertice on each triangle mesh
    ii = F[:, [1,2]]
    jj = F[:, [0,2]]
    kk = F[:, [0,1]]

    ii_max = (np.max(ii, axis=1)).reshape((nF,1))
    ii_min = (np.min(ii, axis=1)).reshape((nF,1))
    ii = np.append(ii_max, ii_min, axis=1)

    jj_max = (np.max(jj, axis=1)).reshape((nF,1))
    jj_min = (np.min(jj, axis=1)).reshape((nF,1))
    jj = np.append(jj_max, jj_min, axis=1)

    kk_max = (np.max(kk, axis=1)).reshape((nF,1))
    kk_min = (np.min(kk, axis=1)).reshape((nF,1))
    kk = np.append(kk_max, kk_min, axis=1)

    ii_unsq = ii.reshape((nF, 2, 1))
    jj_unsq = jj.reshape((nF, 2, 1))
    kk_unsq = kk.reshape((nF, 2, 1))
    stack_idx = np.stack((ii_unsq, jj_unsq, kk_unsq), axis=1).reshape((nF, 3, 2))
    
    vertex_in_face = F.reshape((nF,3,1))
    vertex_opposite_edge = np.append(vertex_in_face, stack_idx, axis=2).reshape((nF*3, 3))

    mat = np.zeros((nV, nV))
    for i in range(nF*3):
        mat[vertex_opposite_edge[i, 1], vertex_opposite_edge[i, 2]] = mat[vertex_opposite_edge[i,1], vertex_opposite_edge[i,2]] + vertex_opposite_edge[i, 0]

    mat = mat.astype(vertex_opposite_edge.dtype)
    opposite_edge_vec = np.zeros((nF*3, 1))
    for i in range(nF*3):
        opposite_edge_vec[i, 0] = mat[vertex_opposite_edge[i,1], vertex_opposite_edge[i,2]] - vertex_opposite_edge[i, 0]

    opposite_edge_vec = opposite_edge_vec.astype(vertex_opposite_edge.dtype)
    vertex_opposite_edge = np.append(vertex_opposite_edge, opposite_edge_vec, axis=1)

    return vertex_opposite_edge


def normalVec(left_v, mid_v, right_v):
    return 0.5 * torch.cross(left_v - mid_v, right_v - mid_v)


def getGaussian(V, F, vertex_opposite_edge):
    '''Computation of mesh Gaussian operator of a triangulated surface evaluated at one of its tangent vectors h. 
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        - vertex_opposite_edge: opposite vertices of edges on each triangle mesh [(nFx3)x4 torch tensor]
    Output:
        - K: function that will evaluate the mesh Gaussian curvature-like operator at a tangent vector to the surface [function]
    '''
    # Compute the dihedral angle of each edge 
    nV, nF = V.shape[0], F.shape[0]
    vertex_coordinates = V[vertex_opposite_edge]
    v_left, edge_v1, edge_v2, v_right = vertex_coordinates[:,0], vertex_coordinates[:,1], vertex_coordinates[:,2], vertex_coordinates[:, 3]
    left_edge, mid_edge, right_edge = v_left - edge_v1, edge_v2 - edge_v1, v_right - edge_v1

    #edge_len = mid_edge.norm(dim=1)

    #A = (edge_v1 - edge_v2).norm(dim=1)
    #B1 = (v_left - edge_v2).norm(dim=1)
    #C1 = (v_left - edge_v1).norm(dim=1)
    #s1 = 0.5 * (A + B1 + C1)

    #B2 = (v_right - edge_v2).norm(dim=1)
    #C2 = (v_right - edge_v1).norm(dim=1)
    #s2 = 0.5 * (A + B2 + C2)
    #area_left = torch.unsqueeze((s1 * (s1-A) * (s1-B1) * (s1-C1)).clamp_(min=1e-6).sqrt(),dim=1).repeat(1,3)
    #area_right = torch.unsqueeze((s2 * (s2-A) * (s2-B2) * (s2-C2)).clamp_(min=1e-6).sqrt(),dim=1).repeat(1,3)

    '''
    face_normal_cos = ((((v_left[:,1] - edge_v1[:,1])*(edge_v2[:,2] - edge_v1[:,2]) - (edge_v2[:,1] - edge_v1[:,1])*(v_left[:,2] - edge_v1[:,2])) * \
            ((edge_v2[:,1] - edge_v1[:,1])*(v_right[:,2] - edge_v1[:,2]) - (edge_v2[:,2] - edge_v1[:,2])*(v_right[:,1] - edge_v1[:,1])) + \
            ((v_left[:,0] - edge_v1[:,0])*(edge_v2[:,2] - edge_v1[:,2]) - (edge_v2[:,0] - edge_v1[:,0])*(v_left[:,2] - edge_v1[:,2])) * \
            ((edge_v2[:,0] - edge_v1[:,0])*(v_right[:,2] - edge_v1[:,2]) - (edge_v2[:,2] - edge_v1[:,2])*(v_right[:,0] - edge_v1[:,0])) + \
            ((v_left[:,0] - edge_v1[:,0])*(edge_v2[:,1] - edge_v1[:,1]) - (edge_v2[:,0] - edge_v1[:,0])*(v_left[:,1] - edge_v1[:,0])) * \
            ((edge_v2[:,0] - edge_v1[:,0])*(v_right[:,1] - edge_v1[:,1]) - (edge_v2[:,1] - edge_v1[:,1])*(v_right[:,0] - edge_v1[:,0]))) / (4 * area1 * area2)).clamp_(min=-1+1e-8)
    '''


    face_normal_left = 0.5 * torch.cross(left_edge, mid_edge)
    #face_normal_left = normalVec(v_left, edge_v1, edge_v2)
    face_normal_left_norm = (torch.unsqueeze(face_normal_left.norm(dim=1),dim=1)).repeat(1,3) + 1e-24
    face_normal_left = face_normal_left / face_normal_left_norm
    #face_normal_left = face_normal_left / area_left

    #mid_edge_clone = mid_edge
    face_normal_right = 0.5 * torch.cross(mid_edge, right_edge)
    #face_normal_right = normalVec(edge_v2, edge_v1, v_right)
    face_normal_right_norm = (torch.unsqueeze(face_normal_right.norm(dim=1),dim=1)).repeat(1,3) + 1e-24
    face_normal_right = face_normal_right / face_normal_right_norm
    #face_normal_right = face_normal_right / area_right

    face_normal_cos = (torch.sum(face_normal_left * face_normal_right, dim=1)).clamp_(min=-1+1e-8, max=1.0)

    #print(face_normal_cos)

    #face_normal_angle = torch.acos(face_normal_cos)
    #face_normal_angle = 0.5 * torch.pi - face_normal_cos
    face_normal_angle = (1 - face_normal_cos) / (1 + face_normal_cos)
    #face_normal_angle = 1 / (1 + face_normal_cos)
    #face_normal_angle = 0.5 * torch.pi - face_normal_cos - (face_normal_cos * face_normal_cos * face_normal_cos) / 6

    # Compute the length of each edge and use it divide angle

    edge_len = (edge_v1 - edge_v2).norm(dim=1)+1e-24

    face_angle_len = face_normal_angle #* edge_len

    idx_j = F[:, [1,2,0]]
    idx_i = F[:, [2,0,1]]

    idx_ij = torch.stack([idx_j, idx_i], dim=0).view(2, nF*3)

    def K(h):

        '''Function that evaluate the Gaussian curvature-like operator of the vector field.
        Input: 
            - h: tangent vector to the triangulated surface [nVx3 torch tensor]
        Output:
            - Kh: Gaussian curvature-like operator of the triangulated surface applied to on its tangent vector h [nVx3 torch tensor]
        '''

        # Compute difference between tangent vectors at adjacent vertices of the surface
        hdiff = h[idx_ij[0]] - h[idx_ij[1]]

        # Evaluate Gaussian curvature-like operator by multiplying dihedral angle with hdiff
        values = (torch.unsqueeze(face_angle_len, dim=1)).repeat(1,3) * hdiff

        # Sum expression over adjacent vertices for each coordinate
        Kh = torch.zeros((nV, 3)).to(dtype=torchdtype, device=torchdeviceId)
        Kh[:, 0] = Kh[:, 0].scatter_add(0, idx_ij[1,:], values[:,0])
        Kh[:, 1] = Kh[:, 1].scatter_add(0, idx_ij[1,:], values[:,1])
        Kh[:, 2] = Kh[:, 2].scatter_add(0, idx_ij[1,:], values[:,2])

        return Kh

    return K




def curvatureCoeff(h, F, param=0.01):
    '''Give the displacement h on vertex, and connection faces F, get the mean & Gauss curvature determined coefficient
    
    Input:
        - h: displacement of each vertice [nVx3 torch tensor]
        - F: faces of the triangluated surface [nFx3 torch tensor]
        - param: a scalar, weight of curvature term 

    Output:
        - coeff: coefficent on each vertice determined by the mean & Gauss cuvature, emphasizing convex structure of h [nVx1 torch tensor]
    '''
    nV, nF = h.shape[0], F.shape[0]

    # Get displacement of each vertice on each face
    face_displacement = h[F]
    h0, h1, h2 = face_displacement[:, 0], face_displacement[:, 1], face_displacement[:, 2]

    # Compute the area of each face of the manifold formed by h using Heron's formula
    A = (h1 - h2).norm(dim=1)
    B = (h0 - h2).norm(dim=1)
    C = (h0 - h1).norm(dim=1)
    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s -C)).clamp_(min=1e-6).sqrt()

    # Compute cotangent expressions for the Laplacian operator for mean curvature norml of manifold h

    # 1. Compute cotangent formula 
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # 2. Compute angle by arccos

    cosa = (B2 + C2 - A2) / ((2 * B * C).clamp_(min=1e-6))
    cosb = (A2 + C2 - B2) / ((2 * A * C).clamp_(min=1e-6))
    cosc = (A2 + B2 - C2) / ((2 * A * B).clamp_(min=1e-6))
    angle_a = torch.acos(cosa)
    angle_b = torch.acos(cosb)
    angle_c = torch.acos(cosc)
    angle = torch.stack([angle_a, angle_b, angle_c], dim=1)

    # 3. Compute Area normal of each face
    face_normal = 0.5 * torch.cross(h1 - h0, h2 - h0)

    # 4. Find indices of adjacent vertices in the manifold (i.e., edge list between vertices) to get mean curvature normal
    ii = F[:, [1,2,0]]
    jj = F[:, [2,0,1]]
    idx = torch.stack([ii, jj], dim=0).view(2, nF * 3)

    hdiff = h[idx[0]] - h[idx[1]]
    values = (torch.stack([cot.view(-1)]*3, dim=1)*hdiff)
    Lh = torch.zeros((nV,3)).to(dtype=torchdtype, device=torchdeviceId)
    Lh[:,0] = Lh[:,0].scatter_add(0, idx[1,:], values[:,0])
    Lh[:,1] = Lh[:,1].scatter_add(0, idx[1,:], values[:,1])
    Lh[:,2] = Lh[:,2].scatter_add(0, idx[1,:], values[:,2])

    # 5. Calculate the Gaussian curvature by angles
    F_idx = F.view(1, nF * 3)
    angle_values = torch.stack([angle.view(-1)], dim=1)
    Gh = torch.zeros((nV, 1)).to(dtype=torchdtype, device=torchdeviceId)
    Gh[:, 0] = Gh[:, 0].scatter_add(0, F_idx[0, :], angle_values[:, 0])
    Gh = 2 * torch.pi - Gh

    # 6. Calculate the normal direction of each vertice
    face_normal_values = (face_normal.repeat(1,3)).view(nF*3, 3)
    Nh = torch.zeros((nV, 3)).to(dtype=torchdtype, device=torchdeviceId)
    Nh[:,0] = Nh[:,0].scatter_add(0, F_idx[0, :], face_normal_values[:,0])
    Nh[:,1] = Nh[:,1].scatter_add(0, F_idx[0, :], face_normal_values[:,1])
    Nh[:,2] = Nh[:,2].scatter_add(0, F_idx[0, :], face_normal_values[:,2])
    Nhnorm = (torch.unsqueeze(Nh.norm(dim=1),dim=1)).repeat(1,3).clamp_(min=1e-6)
    Nh = Nh / Nhnorm

    # 7. Use normal and mean curvature normal to get mean curvature
    Hh = -0.5 * torch.unsqueeze(torch.sum(Lh * Nh, dim=1), dim=1)

    # 8. Divide the face area in mean & Gaussian curvature
    area_values = ((torch.unsqueeze(area,dim=1)).repeat(1,3)).view(nF*3,1)
    Ah = torch.zeros((nV, 1)).to(dtype=torchdtype, device=torchdeviceId)
    Ah[:,0] = (Ah[:,0].scatter_add(0, F_idx[0,:], area_values[:,0]) / 3.0) + 1e-16
    Hh = 0.5 * Hh / Ah
    Gh = Gh / Ah
    
    #zeros_tensor = torch.zeros((nV, 1)).to(dtype=torchdtype, device=torchdeviceId)
    coeff = torch.ones((nV, 1)).to(dtype=torchdtype, device=torchdeviceId) + param * Hh.clamp_(min=0.0, max=1.0) * Gh.clamp_(min=0.0,max=1.0)

    return coeff




def getVertAreas(V,F):
    '''Computation of vertex areas for a triangulated surface.

    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - VertAreas: vertex areas [nVx1 torch tensor]
    '''

    # Number of vertices
    nV = V.shape[0]
    
    # Get x,y,z coordinates of each face
    face_coordinates = V[F]
    v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]

    # Compute the area of each face using Heron's formula
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1) # lengths of each side of the faces
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C) # semi-perimeter
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt() # Apply Heron's formula and clamp areas of small faces for numerical stability

    # Compute the area of each vertex by averaging over the number of faces that it is incident to
    idx = F.view(-1)
    incident_areas = torch.zeros(nV, dtype=torch.float32, device=torchdeviceId)
    val = torch.stack([area] * 3, dim=1).view(-1)
    incident_areas.scatter_add_(0, idx, val)    
    vertAreas = 2*incident_areas/3.0+1e-24   

    return vertAreas


def getNormal(F, V):
    '''Computation of normals at each face of a triangulated surface.

    Input:
        - F: faces of the triangulated surface [nFx3 torch tensor]
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        
    Output:
        - N: vertex areas [nFx1 torch tensor]
    '''

    # Compute normals at each face by taking the cross product between edges of each face that are incident to its x-coordinate
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
    N = .5 * torch.cross(V1 - V0, V2 - V0)

    return N


def computeBoundary(F):
    '''Determining if a vertex is at the boundary of the mesh of a triagulated surface.

    Input:
        - F: faces of the triangulated surface [nFx3 ndarray]

    Output:
        - BoundaryIndicatorOfVertex: boolean vector indicating which vertices are at the boundary of the mesh [nVx1 boolean ndarray]

    Note: This is a CPU computation
    '''
    
    # Get number of vertices and faces
    nF = F.shape[0]
    nV = F.max()+1

    # Find whether vertex is at the boundary of the mesh
    Fnp = F # F.detach().cpu().numpy()
    rows = Fnp[:,[0,1,2]].reshape(3*nF)
    cols = Fnp[:,[1,2,0]].reshape(3*nF)
    vals = np.ones(3*nF,dtype=np.int)
    E = scipy.sparse.coo_matrix((vals,(rows,cols)),shape=(nV,nV))
    E -= E.transpose()
    i,j = E.nonzero()
    BoundaryIndicatorOfVertex = np.zeros(nV,dtype=np.bool)
    BoundaryIndicatorOfVertex[i] = True
    BoundaryIndicatorOfVertex[j] = True

    return BoundaryIndicatorOfVertex
