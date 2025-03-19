# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import numpy as np
import torch
import trimesh
import kaolin
from tqdm import tqdm
import point_cloud_utils as pcu

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return x / length(x, eps)

def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)
    
class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        
    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        nrm = safe_normalize(torch.cross(v1 - v0, v2 - v0))
        self.nrm = nrm

def load_mesh(path, device):
    mesh_np = trimesh.load(path)
    vertices = torch.tensor(mesh_np.vertices, device=device, dtype=torch.float)
    faces = torch.tensor(mesh_np.faces, device=device, dtype=torch.long)
    
    # Normalize
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2 # Center mesh on origin
    vertices = vertices * scale # Rescale to [-0.9, 0.9]
    return Mesh(vertices, faces)

def compute_sdf(points, vertices, faces):
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices.clone().unsqueeze(0), faces)
    distance = kaolin.metrics.trianglemesh.point_to_mesh_distance(points.unsqueeze(0), face_vertices)[0]
    with torch.no_grad():
        sign = (kaolin.ops.mesh.check_sign(vertices.unsqueeze(0), faces, points.unsqueeze(0))<1).float() * 2 - 1
    sdf = (sign*distance).squeeze(0)
    return sdf

def sample_random_points(n, mesh):
    pts_random = (torch.rand((n//2,3),device='cuda') - 0.5) * 2
    pts_surface = kaolin.ops.mesh.sample_points(mesh.vertices.unsqueeze(0), mesh.faces, 500)[0].squeeze(0)
    pts_surface += torch.randn_like(pts_surface) * 0.05
    pts = torch.cat([pts_random, pts_surface])
    return pts #pts_surface


def load_meshes_seq(path_list, return_centere_scale=False):
    v_list=[]
    f_list=[]
    scaled_v_list=[]

    for mesh_path in tqdm(path_list,desc='loading mesh'):
        v,f=pcu.load_mesh_vf(mesh_path)
        v_list.append(v)
        f_list.append(f)

    v_all=np.concatenate(v_list,axis=0)

    center = (v_all.max(0)+v_all.min(0))/2   #v_all.mean(0)
    scale=np.max(v_all.max(0)-v_all.min(0))

    for v in v_list:
        scaled_v_list.append((v-center.reshape(1,3))/scale*1.99)

    if not return_centere_scale:

        return scaled_v_list,f_list
    
    else:
        return scaled_v_list, f_list, center, scale
    
def filter_connected_components(mesh, min_triangle_count):
    # 获取网格的连通组件
    components = mesh.split()

    # 过滤出满足条件的连通组件
    filtered_components = [component for component in components if component.faces.shape[0] > min_triangle_count]

    # 获取满足条件的三角形索引
    filtered_mesh =  trimesh.util.concatenate(filtered_components)           #[triangle for component in filtered_components for triangle in component]

    return filtered_mesh

class SSIM3D(torch.nn.Module):
    def __init__(self,window_size=5,sigma=1.5,channel=4):
        super().__init__()

        self.padding = window_size // 2
        self.channel=channel
        #self.data_range=data_range
        self.register_buffer('kernel',self.create_gaussian_kernel(window_size, sigma, channel))
        

    def create_gaussian_kernel(self,window_size, sigma, channels):
        kernel = torch.tensor([
            [[self.gaussian(x-window_size//2, y-window_size//2, z-window_size//2, sigma) for z in range(window_size)]
            for y in range(window_size)]
            for x in range(window_size)
        ])
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, window_size, window_size, window_size).repeat(channels, 1, 1, 1, 1)
        return kernel

    def gaussian(self,x, y, z, sigma):
        return math.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    
    def forward(self,x,y):
        x=(x+1)/2
        y=(y+1)/2
        mu1 = F.conv3d(x, self.kernel, padding=self.padding, groups=self.channel)
        mu2 = F.conv3d(y, self.kernel, padding=self.padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu12 = mu1 * mu2

        sigma1_sq = F.conv3d(x * x, self.kernel, padding=self.padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv3d(y * y, self.kernel, padding=self.padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv3d(x * y, self.kernel, padding=self.padding, groups=self.channel) - mu12

        c1 = 0.01**2    #(0.01 * self.data_range) ** 2
        c2 = 0.03**2    #(0.03 * self.data_range) ** 2

        numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator
        ssim_val = torch.mean(ssim_map)

        return ssim_val
