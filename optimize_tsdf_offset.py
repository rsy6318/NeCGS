import os
import numpy as np
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import random
from util import *
import render
import loss
import imageio

from fmc import dynamic_marching_cubes, construct_voxel_grid, base_cube_edges
import point_cloud_utils as pcu

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.004)) # Exponential falloff from [1.0, 0.1] over 500 epochs.    

class STEQuantize(torch.autograd.Function):
  """Straight-Through Estimator for Quantization.

  Forward pass implements quantization by rounding to integers,
  backward pass is set to gradients of the identity function.
  """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return x.round()

  @staticmethod
  def backward(ctx, grad_outputs):
    return grad_outputs
  
def diff_quantized_tensor(input,num_bits=8,min=-1,max=1,quant=True):
    input=torch.clamp(input,min,max)
    if True:
        quant=STEQuantize.apply
        scale=(max - min) / (2**num_bits)
        quanted_tensor=quant((input-min)/(scale))*scale+min
        return quanted_tensor
    else:
        return input 
    

def opt_fmc(input_v,input_f,iter=500,train_res=[2048,2048],lr=0.01,batch=4,voxel_grid_res=127, device='cuda', sdf_reg_weights=0):
    x_nx3, cube_fx8 = construct_voxel_grid(voxel_grid_res,device)
    x_nx3 *= 2 # scale up the grid so that it's larger than the target object

    all_edges = cube_fx8[:, base_cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)

    gt_v=input_v
    gt_f=input_f

    #gt_mesh=Mesh(torch.from_numpy(gt_v).float().to(device),torch.from_numpy(gt_f).long().to(device))
    #gt_mesh.auto_normals()

    init_sdf_np,_,_=pcu.signed_distance_to_mesh(x_nx3.cpu().numpy(),gt_v.astype(np.float32),gt_f)
    if init_sdf_np[-1]<0:
        init_sdf_np*=-1

    sdf=torch.from_numpy(init_sdf_np).float().cuda()
    sdf=torch.clip(sdf/(2*2/voxel_grid_res),-1,1 ).cpu().numpy()
    
    deform = torch.zeros_like(x_nx3).cpu().numpy()

    return sdf.reshape(voxel_grid_res+1,voxel_grid_res+1,voxel_grid_res+1,1),deform.reshape(voxel_grid_res+1,voxel_grid_res+1,voxel_grid_res+1,3)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--niter',type=int,default=500)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--save_path',type=str)
    parser.add_argument('--num_frames',type=int)
    args=parser.parse_args()
    #seq_name=args.seq   #'soldier'

    save_path= args.save_path #     

    data_path=args.data_path 

    os.makedirs(save_path,exist_ok=True)
    

    index_list=list(range(args.num_frames))
    random.shuffle(index_list)

    for i in tqdm(index_list):
        gt_mesh=trimesh.load_mesh(os.path.join(data_path,'%04d.obj'%i))
        gt_v,gt_f=gt_mesh.vertices,gt_mesh.faces


        if os.path.exists(os.path.join(save_path,'data','%04d.npz'%i)):
            print(os.path.join(save_path,'data','%04d.npz'%i),' exists, skip !!!')
            continue

        sdf_grid,offset_grid=opt_fmc(gt_v,gt_f,iter=args.niter,sdf_reg_weights=0,batch=args.batch)

        os.makedirs(os.path.join(save_path,'meshes','%04d'%i),exist_ok=True)
        os.makedirs(os.path.join(save_path,'data',),exist_ok=True)

        np.savez_compressed(os.path.join(save_path,'data','%04d.npz'%i),sdf=sdf_grid,offset=offset_grid)
        pcu.save_mesh_vf(os.path.join(save_path,'meshes','%04d'%i,'gt_mesh.obj'),gt_v,gt_f)

