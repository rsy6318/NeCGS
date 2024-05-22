import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

class SDF_dataset(Dataset):
    def __init__(self,args):
        super().__init__()
        data_path=args.data_path
        num_frames=args.num_frames
        pin_memory=args.pin_memory
        self.mask_threshold=args.mask_threshold
        self.data_list=[]
        self.sdf_data_path_list=[]
        self.offset_data_path_list=[]
        self.pin_memory=pin_memory

        if not num_frames:
            num_frames=len(glob(os.path.join(data_path,'*','sdf_grid.npy')))

        for i in tqdm(range(num_frames)):
            
            self.sdf_data_path_list.append(os.path.join(data_path,'%04d'%i,'sdf_grid.npy'))
            self.offset_data_path_list.append(os.path.join(data_path,'%04d'%i,'offset_grid.npy'))

            if self.pin_memory: 
                sdf_grids=np.load(os.path.join(data_path,'%04d'%i,'sdf_grid.npy'))
                #sdf_grids=sdf_grids
                mask=(np.abs(sdf_grids)<self.mask_threshold).astype(np.float32)

                offset_grids=np.load(os.path.join(data_path,'%04d'%i,'offset_grid.npy'))

                sdf_offset=np.concatenate([sdf_grids,offset_grids],axis=-1).astype(np.float32)

                self.data_list.append((sdf_offset,mask))
                


    def __len__(self):

        return len(self.sdf_data_path_list)
    

    def __getitem__(self, index):
        if self.pin_memory:
            sdf_offset,mask=self.data_list[index]
        else:
            sdf_grids=np.load(self.sdf_data_path_list[index])
            mask=(np.abs(sdf_grids)<self.mask_threshold).astype(np.float32)
            offset_grids=np.load(self.offset_data_path_list[index])
            sdf_offset=np.concatenate([sdf_grids,offset_grids],axis=-1).astype(np.float32)

        return {'t':torch.tensor(float(index)/len(self.data_list)),
                'index': torch.tensor(index).long(),
                'sdf_offset':torch.from_numpy(sdf_offset).float(),
                'mask':torch.from_numpy(mask).float()}
    
class SDF_dataset_npz(Dataset):
    def __init__(self,args):
        super().__init__()
        data_path=args.data_path
        num_frames=args.num_frames
        pin_memory=args.pin_memory
        self.mask_threshold=args.mask_threshold
        self.data_list=[]
        #self.sdf_data_path_list=[]
        #self.offset_data_path_list=[]
        self.npz_path_list=[]
        self.pin_memory=pin_memory

        if num_frames<1:
            num_frames=len(glob(os.path.join(data_path,'data','*.npz')))

        #print(os.path.join(data_path,'data','*.npz'))

        for i in tqdm(range(num_frames)):
            
            #self.sdf_data_path_list.append(os.path.join(data_path,'%04d'%i,'sdf_grid.npy'))
            #self.offset_data_path_list.append(os.path.join(data_path,'%04d'%i,'offset_grid.npy'))
            self.npz_path_list.append(os.path.join(data_path,'data','%04d.npz'%i))
            

            if self.pin_memory: 
                npz_data=np.load(os.path.join(data_path,'data','%04d.npz'%i))
                #sdf_grids=np.load(os.path.join(data_path,'%04d'%i,'sdf_grid.npy'))
                #sdf_grids=sdf_grids
                sdf_grids=npz_data['sdf']
                offset_grids=npz_data['offset']
                mask=(np.abs(sdf_grids)<self.mask_threshold).astype(np.float32)

                #offset_grids=np.load(os.path.join(data_path,'%04d'%i,'offset_grid.npy'))

                sdf_offset=np.concatenate([sdf_grids,offset_grids],axis=-1).astype(np.float32)
                self.data_list.append((sdf_offset,mask))
                


    def __len__(self):

        return len(self.npz_path_list)
    

    def __getitem__(self, index):
        if self.pin_memory:
            sdf_offset,mask=self.data_list[index]
        else:
            npz_data=np.load(self.npz_path_list[index])
            #sdf_grids=np.load(os.path.join(data_path,'%04d'%i,'sdf_grid.npy'))
            #sdf_grids=sdf_grids
            sdf_grids=npz_data['sdf']
            offset_grids=npz_data['offset']
            #sdf_grids=np.load(self.sdf_data_path_list[index])
            mask=(np.abs(sdf_grids)<self.mask_threshold).astype(np.float32)
            #offset_grids=np.load(self.offset_data_path_list[index])
            sdf_offset=np.concatenate([sdf_grids,offset_grids],axis=-1).astype(np.float32)

        return {'t':torch.tensor(float(index)/len(self.data_list)),
                'index':torch.tensor(index).long(),
                'sdf_offset':torch.from_numpy(sdf_offset).float(),
                'mask':torch.from_numpy(mask).float()}


class SDF_dataset6(Dataset):
    def __init__(self,args):
        super().__init__()
        data_path=args.data_path
        num_frames=args.num_frames
        pin_memory=args.pin_memory
        self.mask_threshold=args.mask_threshold
        self.data_list=[]
        self.sdf_data_path_list=[]
        self.offset_data_path_list=[]
        self.pin_memory=pin_memory

        if not num_frames:
            num_frames=len(glob(os.path.join(data_path,'*','sdf_grid.npy')))

        for i in tqdm(range(num_frames)):
            
            self.sdf_data_path_list.append(os.path.join(data_path,'%06d'%i,'sdf_grid.npy'))
            self.offset_data_path_list.append(os.path.join(data_path,'%06d'%i,'offset_grid.npy'))

            if self.pin_memory: 
                sdf_grids=np.load(os.path.join(data_path,'%06d'%i,'sdf_grid.npy'))
                #sdf_grids=sdf_grids
                mask=(np.abs(sdf_grids)<self.mask_threshold).astype(np.float32)

                offset_grids=np.load(os.path.join(data_path,'%06d'%i,'offset_grid.npy'))

                sdf_offset=np.concatenate([sdf_grids,offset_grids],axis=-1).astype(np.float32)

                self.data_list.append((sdf_offset,mask))

    def __len__(self):

        return len(self.sdf_data_path_list)
    
    def __getitem__(self, index):
        if self.pin_memory:
            sdf_offset,mask=self.data_list[index]
        else:
            sdf_grids=np.load(self.sdf_data_path_list[index])
            mask=(np.abs(sdf_grids)<self.mask_threshold).astype(np.float32)
            offset_grids=np.load(self.offset_data_path_list[index])
            sdf_offset=np.concatenate([sdf_grids,offset_grids],axis=-1).astype(np.float32)

        return {'t':torch.tensor(float(index)/len(self.data_list)),
                'sdf_offset':torch.from_numpy(sdf_offset).float(),
                'mask':torch.from_numpy(mask).float()}
    
def get_dataset(name,args):
    if name == 'SDF_dataset':
        return SDF_dataset(args)
    elif name == 'SDF_dataset6':
        return SDF_dataset6(args)
    elif name=='SDF_dataset_npz':
        return SDF_dataset_npz(args)
    else:
        assert False


