import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import math
from tqdm import tqdm
from config_load import get_config, save_config
from network import get_network,adjust_lr,diff_quantized_tensor
from dataset import get_dataset

from fmc import dynamic_marching_cubes, construct_voxel_grid, base_cube_edges
from util import Mesh, SSIM3D
import imageio
import trimesh
import time

def train(args):
    args.log_path=os.path.join(args.log_path,args.decoder_dim_list+'_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) )
    os.makedirs(args.log_path,exist_ok=True)

    #dataset
    voxel_dataset=get_dataset(args.dataset,args)    #SDF_dataset(args.data_path)
    
    data_loader=DataLoader(dataset=voxel_dataset,batch_size=args.batch_size,shuffle=True,num_workers=1)
    val_data_loader=DataLoader(dataset=voxel_dataset,batch_size=1,shuffle=False,num_workers=1)
    args.num_frames=len(voxel_dataset)

    #network
    if args.init_method=='uniform':
        all_embed_features=torch.rand((args.num_frames,args.embed_hwd,args.embed_hwd,args.embed_hwd,args.embed_dim)).to(args.device)
    elif args.init_method=='normal':
        all_embed_features=torch.randn((args.num_frames,args.embed_hwd,args.embed_hwd,args.embed_hwd,args.embed_dim)).to(args.device)/3
    all_embed_features.requires_grad=True
    net=get_network(args.model,args).to(args.device)
    
    ssim_1_channel=SSIM3D(channel=1).to(args.device) #cuda()
    ssim_3_channel=SSIM3D(channel=3).to(args.device) #cuda()

    #optimizer=torch.optim.Adam([net.parameters(),all_embed_features],lr=args.lr)
    optimizer=torch.optim.Adam(
        [   {"params": net.parameters(),"lr": args.lr,},
            {"params": [all_embed_features],"lr": args.lr,} ]
    )
    
    #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.n_epoch,args.lr_min)

    save_config(os.path.join(args.log_path,'config.txt'),args)
    
    for epoch in range(1,args.n_epoch+1):
        for step,data_dict in enumerate(data_loader):
            
            #input_t=data_dict['t'].cuda()
            gt_sdf_offset=data_dict['sdf_offset'].to(args.device)
            gt_mask=data_dict['mask'].to(args.device) #cuda()
            index=data_dict['index']
            embed_features=all_embed_features[index]
            quant_embed_features=diff_quantized_tensor(embed_features,args.num_bits)
            pred_sdf_offset=net(quant_embed_features)

            loss=torch.mean(torch.abs(gt_sdf_offset-pred_sdf_offset))

            if args.important_weight:
                loss+=args.important_weight*torch.sum(gt_mask*torch.abs(gt_sdf_offset-pred_sdf_offset))/torch.sum(gt_mask)
                
            if args.ssim_weight:
                loss+=args.ssim_weight*(1-ssim_1_channel(pred_sdf_offset[...,0:1].permute(0,4,1,2,3,),gt_sdf_offset[...,0:1].permute(0,4,1,2,3,))+1-ssim_3_channel(pred_sdf_offset[...,1:].permute(0,4,1,2,3,),gt_sdf_offset[...,1:].permute(0,4,1,2,3,)))
            if args.l1_reg:
                loss+=args.l1_reg*(net.get_quantparams())

            if args.offset_weight:
                offset_length=torch.sqrt(torch.sum(gt_sdf_offset[...,1:]**2,dim=-1,keepdim=True))
                loss+=args.offset_weight*torch.sum( offset_length*torch.abs(gt_sdf_offset-pred_sdf_offset))/torch.sum(offset_length)
            
            if args.embed_reg:
                loss+=args.embed_reg*torch.abs(quant_embed_features).mean()
            
            current_lr=adjust_lr(optimizer, (epoch-1) % args.n_epoch, step, len(voxel_dataset), args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%30==0:
                print('%s epoch: %04d, step: %d/%d, current lr: %f, loss: %f'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),epoch,step,len(data_loader)/args.batch_size,current_lr,loss.cpu().detach().numpy()))

        if epoch%args.val_frequence==0:
            x_nx3, cube_fx8 = construct_voxel_grid(args.voxel_grid_res,args.device)
            x_nx3 *= 2

            os.makedirs(os.path.join(args.log_path,'checkpoint_%04d'%epoch,'rec_mesh'))
            os.makedirs(os.path.join(args.log_path,'checkpoint_%04d'%epoch,'embed_features'))

            for index_t,data_dict in enumerate(tqdm(val_data_loader,desc='val epoch %d'%epoch)):
                try:
                    gt_sdf_offset=data_dict['sdf_offset'].to(args.device) 
                    
                    index=data_dict['index']
                    embed_features=all_embed_features[index]
                    quant_embed_features=diff_quantized_tensor(embed_features,args.num_bits)
                    pred_sdf_offset=net(quant_embed_features)

                    pred_sdf=pred_sdf_offset[...,0].reshape(-1)  
                    pred_offset=pred_sdf_offset[...,1:].reshape(-1,3)
                    grid_verts=x_nx3+pred_offset * (2-1e-8) / (args.voxel_grid_res * 2)

                    #vertices, faces, L_dev = fc(grid_verts, pred_sdf, cube_fx8, args.voxel_grid_res, training=True)
                    vertices, faces=dynamic_marching_cubes(grid_verts,cube_fx8,pred_sdf)
                    
                    if epoch%100==0 :
                        mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
                        mesh_np.export(os.path.join(args.log_path,'checkpoint_%04d'%epoch,'rec_mesh','rec_mesh_%04d.obj'%int(index_t)))
                        np.save(os.path.join(args.log_path,'checkpoint_%04d'%epoch,'embed_features','embed_feature_%04d.npy'%index_t),quant_embed_features.cpu().detach().numpy())
                except:
                    pass
            

            #torch.save(net.state_dict(),os.path.join(args.log_path,'model%d.pt'%epoch))
            net.save_quanted_decoder_weights(os.path.join(args.log_path,'checkpoint_%04d'%epoch,'decoder.pt'))
            #net.save_encoder_weights(os.path.join(args.log_path,'checkpoint_%04d'%epoch,'encoder.pt'))

        net.save_quanted_decoder_weights(os.path.join(args.log_path,'decoder_last.pt'))
        

if __name__=='__main__':
    args=get_config().parse_args()
    train(args)