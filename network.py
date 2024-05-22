import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math
from tqdm import tqdm
from quantize_utils import QuantConv3d,QuantLinear,QuantConvTranspose3d
from glob import glob
from timm.models.layers import trunc_normal_, DropPath

def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        
        # Reshape to prepare for pixel shuffle
        x = x.view(batch_size, channels // self.upscale_factor ** 3, self.upscale_factor, self.upscale_factor, self.upscale_factor, depth, height, width)
        
        # Permute dimensions for pixel shuffle
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        
        # Reshape to get the final result
        x = x.view(batch_size, channels // self.upscale_factor ** 3, depth * self.upscale_factor, height * self.upscale_factor, width * self.upscale_factor)
        
        return x

"""class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv3d(ngf, new_ngf * stride * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale =  PixelShuffle3D(stride)         #nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose3d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv3d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)"""
    

class QuantSepConv(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_size, num_bits,bias=True):
        super().__init__()

        self.conv1=QuantConv3d(in_channel,out_channel//3,(kernel_size,1,1),1,(kernel_size//2,0,0),bias=bias,num_bits=num_bits)
        self.conv2=QuantConv3d(in_channel,out_channel//3,(1,kernel_size,1),1,(0,kernel_size//2,0),bias=bias,num_bits=num_bits)
        self.conv3=QuantConv3d(in_channel,out_channel//3,(1,1,kernel_size),1,(0,0,kernel_size//2),bias=bias,num_bits=num_bits)
        self.conv=QuantConv3d(out_channel//3*3, out_channel,1,1,0,bias=bias,num_bits=num_bits)

    def forward(self, input):
        feature1=self.conv1(input)
        feature2=self.conv2(input)
        feature3=self.conv3(input)

        output=self.conv(torch.cat([feature1,feature2,feature3],dim=1))
        return output


class QuantCustomConv(nn.Module):
    def __init__(self, **kargs):
        super(QuantCustomConv, self).__init__()

        ngf, new_ngf, stride,num_bits = kargs['ngf'], kargs['new_ngf'], kargs['stride'], kargs['num_bits']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = QuantConv3d(ngf, new_ngf * stride * stride * stride, 3, 1, 1, bias=kargs['bias'],num_bits=num_bits)
            self.up_scale =  PixelShuffle3D(stride)         #nn.PixelShuffle(stride)
        elif self.conv_type == 'conv2':
            self.conv = QuantConv3d(ngf, new_ngf//4 * stride * stride * stride, 3, 1, 1, bias=kargs['bias'],num_bits=num_bits)
            self.up_scale =  nn.Sequential(PixelShuffle3D(stride), QuantConv3d(new_ngf//4, new_ngf, 3, 1, 1, bias=kargs['bias'],num_bits=num_bits))         #nn.PixelShuffle(stride)
        elif self.conv_type== 'sepconv':
            self.conv = QuantSepConv(ngf, new_ngf * stride * stride * stride, 3, num_bits=num_bits, bias=kargs['bias'])
            self.up_scale =  PixelShuffle3D(stride)         #nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = QuantConvTranspose3d(ngf, new_ngf, stride, stride, num_bits=num_bits)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=True)
            #self.up_scale = QuantConv3d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'],num_bits=num_bits)
            self.up_scale = QuantConv3d(ngf, new_ngf, 3, 1, 1, bias=kargs['bias'],num_bits=num_bits)
        else:
            print('no such conv type')
            assert False

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)

def MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)

def QuantMLP(dim_list, act='relu', bias=True,num_bits=8):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [QuantLinear(dim_list[i], dim_list[i+1], bias=bias,num_bits=num_bits), act_fn]
    return nn.Sequential(*fc_list)

class PositionalEncoding(nn.Module):
    def __init__(self, lbase=1.25,levels=40):
        super(PositionalEncoding, self).__init__()
        """self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels"""

        self.lbase=lbase
        self.levels=levels
        self.embed_length = 2 * self.levels

    def forward(self, pos):
        
        pe_list = []
        for i in range(self.levels):
            temp_value = pos * self.lbase **(i) * math.pi
            pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
        return torch.stack(pe_list, 1).squeeze(-1)

"""class NeRVBlock3D(nn.Module):
    def __init__(self, in_channel,out_channel,scale,bias,act,conv_type):
        super().__init__()

        self.conv = CustomConv(ngf=in_channel, new_ngf=out_channel, stride=scale, bias=bias, 
            conv_type=conv_type)
        self.act = ActivationLayer(act)

    def forward(self, x):
        return self.act(self.conv(x))"""
    
class QuantNeRVBlock3D(nn.Module):
    def __init__(self, in_channel,out_channel,scale,bias,act,conv_type,num_bits):
        super().__init__()

        self.conv = QuantCustomConv(ngf=in_channel, new_ngf=out_channel, stride=scale, bias=bias, 
            conv_type=conv_type,num_bits=num_bits)
        self.act = ActivationLayer(act)

    def forward(self, x):
        return self.act(self.conv(x))


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
  
def diff_quantized_tensor(input,num_bits=8,min=-1,max=1):
    quant=STEQuantize.apply
    scale=(max - min) / (2**num_bits)
    input=torch.clamp(input,min,max)
    quanted_tensor=quant((input-min)/(scale))*scale+min
    #quanted_tensor=torch.clamp(quanted_tensor,min,max)
    return quanted_tensor

"""class QuantOutputGenerator(nn.Module):
    def __init__(self, args): # lbase=1.25, levels=40, fc_num=1, fc_hwd=8, fc_dim=512, fc_dim_last=24, scale_list=[2, 2, 2, 2],channel_list=[24,24,24,24],bias=True,act='swish',conv_type='conv'
                 
        super().__init__()
        #fc_dim=fc_dim
        
        lbase = args.lbase                   #1.25 
        levels =  args.levels                #40 
        fc_num =  args.fc_num                #1 
        fc_hwd =  args.fc_hwd                #8 
        fc_dim =  args.fc_dim                #512 
        fc_dim_last =  args.fc_dim_last      #24 
        scale_list = [int(scale) for scale in args.scale_list.split('_')]       #[2, 2, 2, 2]
        channel_list = [int(channel) for channel in args.channel_list.split('_')]    #[24,24,24,24]
        bias = args.bias             #True
        act=args.act                      #'swish'
        conv_type=args.conv_type                #'conv'

        self.PE=PositionalEncoding(lbase,levels)

        self.fc_hwd=fc_hwd

        mlp_dim_list=[levels*2] + [fc_dim] * fc_num + [fc_hwd**3*fc_dim_last]

        self.fc_layers = MLP(dim_list=mlp_dim_list, act=act)

        self.layers= nn.ModuleList()
        self.last_layer= nn.Conv3d(channel_list[-1],4,3,1,1,bias=bias) #nn.ModuleList()
        
        for i in range(len(scale_list)):
            if i==0:
                in_channel=fc_dim_last
                out_channel=channel_list[i]
                scale=scale_list[i]
            else:
                in_channel=channel_list[i-1]
                out_channel=channel_list[i]
                scale=scale_list[i]
            self.layers.append(QuantNeRVBlock3D(in_channel,out_channel,scale,bias,act,conv_type))

    
    def forward(self,input_t):
        embed_t=self.PE(input_t)
        output=self.fc_layers(embed_t)

        output=output.reshape(output.size(0),-1,self.fc_hwd,self.fc_hwd,self.fc_hwd) #.permute(0,4,1,2,3)

        for layer in self.layers:
            output=layer(output)

        output=self.last_layer(output)

        output=output.permute(0,2,3,4,1)    #(B,N,N,N,4)

        #output_offset=torch.clip(output[...,1:],-1,1)
        #output_sdf=torch.clip(output[...,0:1],-1,1)

        #return torch.concat([output_sdf,output_offset],dim=-1)
        return  diff_quantized_tensor(output) #torch.clip(output,-1,1)"""

"""class Generator_OCC_Dis(nn.Module):
    def __init__(self, lbase=1.25, levels=40, fc_num=1, fc_hwd=8, fc_dim=512, fc_dim_last=24, scale_list=[2, 2, 2, 2],channel_list=[24,24,24,24],bias=True,act='swish',conv_type='conv'):
        super().__init__()
        fc_dim=fc_dim
        
        self.PE=PositionalEncoding(lbase,levels)

        self.fc_hwd=fc_hwd

        mlp_dim_list=[levels*2] + [fc_dim] * fc_num + [fc_hwd**3*fc_dim_last]

        self.fc_layers = MLP(dim_list=mlp_dim_list, act=act)

        self.layers= nn.ModuleList()
        
        for i in range(len(scale_list)):
            if i==0:
                in_channel=fc_dim_last
                out_channel=channel_list[i]
                scale=scale_list[i]
            else:
                in_channel=channel_list[i-1]
                out_channel=channel_list[i]
                scale=scale_list[i]
            self.layers.append(NeRVBlock3D(in_channel,out_channel,scale,bias,act,conv_type))

        self.last_layer= nn.Conv3d(channel_list[-1],5,3,1,1,bias=bias) #nn.ModuleList()

    
    def forward(self,input_t):
        embed_t=self.PE(input_t)
        output=self.fc_layers(embed_t)

        output=output.reshape(output.size(0),-1,self.fc_hwd,self.fc_hwd,self.fc_hwd) #.permute(0,4,1,2,3)

        for layer in self.layers:
            output=layer(output)

        output=self.last_layer(output)

        output=output.permute(0,2,3,4,1)    #(B,N,N,N,5)

        output_offset=torch.tanh(output[...,1:4])
        output_dis=torch.abs(output[...,0:1])
        output_occ=torch.sigmoid(output[...,4:])

        return torch.concat([output_dis,output_offset],dim=-1),output_occ"""

"""
class QuantGenerator(nn.Module):
    def __init__(self, args): # lbase=1.25, levels=40, fc_num=1, fc_hwd=8, fc_dim=512, fc_dim_last=24, scale_list=[2, 2, 2, 2],channel_list=[24,24,24,24],bias=True,act='swish',conv_type='conv'
                 
        super().__init__()
        self.args=args

        lbase = args.lbase                   #1.25 
        levels =  args.levels                #40 
        fc_num =  args.fc_num                #1 
        fc_hwd =  args.fc_hwd                #8 
        fc_dim =  args.fc_dim                #512 
        fc_dim_last =  args.fc_dim_last      #24 
        scale_list = [int(scale) for scale in args.scale_list.split('_')]       #[2, 2, 2, 2]
        channel_list = [int(channel) for channel in args.channel_list.split('_')]    #[24,24,24,24]
        bias = args.bias             #True
        act=args.act                      #'swish'
        conv_type=args.conv_type                #'conv'

        num_bits=args.num_bits
        self.num_bits=args.num_bits

        self.PE=PositionalEncoding(lbase,levels)

        self.fc_hwd=fc_hwd

        mlp_dim_list=[levels*2] + [fc_dim] * fc_num + [fc_hwd**3*fc_dim_last]

        self.fc_layers = QuantMLP(dim_list=mlp_dim_list, act=act,num_bits=num_bits)

        self.layers= nn.ModuleList()
        self.last_layer= QuantConv3d(channel_list[-1],4,3,1,1,bias=bias,num_bits=num_bits) #nn.ModuleList()

        for i in range(len(scale_list)):
            if i==0:
                in_channel=fc_dim_last
                out_channel=channel_list[i]
                scale=scale_list[i]
            else:
                in_channel=channel_list[i-1]
                out_channel=channel_list[i]
                scale=scale_list[i]
            self.layers.append(QuantNeRVBlock3D(in_channel,out_channel,scale,bias,act,conv_type,num_bits=num_bits))

    def forward(self,input_t):
        embed_t=self.PE(input_t)
        output=self.fc_layers(embed_t)

        output=output.reshape(output.size(0),-1,self.fc_hwd,self.fc_hwd,self.fc_hwd) #.permute(0,4,1,2,3)

        for layer in self.layers:
            output=layer(output)

        output=self.last_layer(output)

        output=output.permute(0,2,3,4,1)    #(B,N,N,N,4)

        #output_offset=torch.clip(output[...,1:],-1,1)
        #output_sdf=torch.clip(output[...,0:1],-1,1)

        #return torch.concat([output_sdf,output_offset],dim=-1)
        return  torch.clip(output,-1,1)  #diff_quantized_tensor(output,num_bits=self.num_bits)  # torch.clip(output,-1,1)
    
    def get_params(self,):
        pass

    def get_quantparams(self):
        all_params=[]
        for param in self.parameters():
            all_params.append(diff_quantized_tensor(param.reshape(-1),self.num_bits))
        all_params=torch.cat(all_params,dim=0)
        return torch.mean(all_params)
    
    def save_quanted_weights(self,save_path):
        ori_weight_dict=self.state_dict()
        quanted_weight_dict={}
        for key in ori_weight_dict.keys():
            quanted_weight_dict[key]=diff_quantized_tensor(ori_weight_dict[key],self.num_bits)

        torch.save(quanted_weight_dict,save_path)

"""



def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, args):
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if args.lr_type == 'cosine':
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - int(args.warmup*args.n_epoch))/ (args.n_epoch - int(args.warmup*args.n_epoch))) + 1.0)
    elif args.lr_type == 'step':
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(args.lr_steps)))
    elif args.lr_type == 'const':
        lr_mult = 1
    elif args.lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < int(args.warmup*args.n_epoch):
        lr_mult = 0.1 + 0.9 * cur_epoch / int(args.warmup*args.n_epoch)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult


#---------------convnext-----------

class Block3D(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x


class ConvNeXt3D(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm3D(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv3d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv3d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm3D(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block3D(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm3D(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width, depth).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class QuantGeneratorV2(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        encoder_dim_list=[int(encoder_dim) for encoder_dim in args.encoder_dim_list.split('_')]
        encoder_stride_list=[int(encoder_stride) for encoder_stride in args.encoder_stride_list.split('_')]

        decoder_dim_list=[int(decoder_dim) for decoder_dim in args.decoder_dim_list.split('_')]
        decoder_stride_list=[int(decoder_stride) for decoder_stride in args.decoder_stride_list.split('_')]

        bias = args.bias             
        act=args.act                      
        conv_type=args.conv_type   

        after_embed_dim=args.after_embed_dim        

        num_bits=args.num_bits
        self.num_bits=args.num_bits

        self.encoder=ConvNeXt3D(stage_blocks=1,strds=encoder_stride_list,dims=encoder_dim_list,in_chans=4,drop_path_rate=0)

        decoder_layers_list=[]
        if after_embed_dim>0:
            decoder_layers_list.append(QuantConv3d(encoder_dim_list[-1],after_embed_dim,1,1,bias=bias,num_bits=num_bits))
        else:
            after_embed_dim=encoder_dim_list[-1]

        for i in range(len(decoder_dim_list)):
            if i==0:
                in_channel=after_embed_dim
                out_channel=decoder_dim_list[i]
                scale=decoder_stride_list[i]
            else:
                in_channel=decoder_dim_list[i-1]
                out_channel=decoder_dim_list[i]
                scale=decoder_stride_list[i]
            decoder_layers_list.append(QuantNeRVBlock3D(in_channel,out_channel,scale,bias,act,conv_type,num_bits=num_bits))

        decoder_layers_list.append( QuantConv3d(decoder_dim_list[-1],4,3,1,1,bias=bias,num_bits=num_bits))
        self.decoder=nn.Sequential(*decoder_layers_list)

    def forward(self, input_voxel,embed_features=None):
        if input_voxel is not None:
            input_voxel=input_voxel.permute(0,4,1,2,3)
            embed_features=self.encoder(input_voxel)
            embed_features=diff_quantized_tensor(embed_features,self.num_bits)
        pred_voxel=self.decoder(embed_features)
        
        return pred_voxel.permute(0,2,3,4,1),embed_features

    def get_encoder_params(self):
        all_params=[]
        for param in self.encoder.parameters():
            all_params.append(param.reshape(-1))
        all_params=torch.cat(all_params,dim=0)
        return torch.mean(all_params) 

    def get_decoder_quantparams(self):
        all_params=[]
        for param in self.decoder.parameters():
            all_params.append(diff_quantized_tensor(param.reshape(-1),self.num_bits))
        all_params=torch.cat(all_params,dim=0)
        return torch.mean(all_params) 
    
    def save_encoder_weights(self,save_path):
        ori_weight_dict=self.encoder.state_dict()
        torch.save(ori_weight_dict,save_path)

    def save_quanted_decoder_weights(self,save_path):
        ori_weight_dict=self.decoder.state_dict()
        quanted_weight_dict={}
        for key in ori_weight_dict.keys():
            quanted_weight_dict[key]=diff_quantized_tensor(ori_weight_dict[key],self.num_bits)
        torch.save(quanted_weight_dict,save_path)

def get_network(name,args):
    if name == "QuantGeneratorV2":
        return QuantGeneratorV2(args)
    else:
        print('no selected model !!!')
        assert False



class QuantDecoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        decoder_dim_list=[int(decoder_dim) for decoder_dim in args.decoder_dim_list.split('_')]
        decoder_stride_list=[int(decoder_stride) for decoder_stride in args.decoder_stride_list.split('_')]

        embed_dim=args.embed_dim

        bias = args.bias             
        act=args.act                      
        conv_type=args.conv_type   

        after_embed_dim=args.after_embed_dim        

        num_bits=args.num_bits
        self.num_bits=args.num_bits

        decoder_layers_list=[]
        if after_embed_dim>0:
            decoder_layers_list.append(QuantConv3d(embed_dim,after_embed_dim,1,1,bias=bias,num_bits=num_bits))
        else:
            after_embed_dim=embed_dim

        for i in range(len(decoder_dim_list)):
            if i==0:
                in_channel=after_embed_dim
                out_channel=decoder_dim_list[i]
                scale=decoder_stride_list[i]
            else:
                in_channel=decoder_dim_list[i-1]
                out_channel=decoder_dim_list[i]
                scale=decoder_stride_list[i]
            decoder_layers_list.append(QuantNeRVBlock3D(in_channel,out_channel,scale,bias,act,conv_type,num_bits=num_bits))

        decoder_layers_list.append( QuantConv3d(decoder_dim_list[-1],4,3,1,1,bias=bias,num_bits=num_bits))
        self.decoder_layers=nn.Sequential(*decoder_layers_list)

    def forward(self,embed_features):
        #embed_features:    B,N,N,N,C
        embed_features=embed_features.permute(0,4,1,2,3)
        pred_voxel=self.decoder_layers(embed_features)  #(B,C,N,N,N)
        return pred_voxel.permute(0,2,3,4,1)        #(B,N,N,N,C)

    def get_quantparams(self):
        all_params=[]
        for param in self.parameters():
            all_params.append(diff_quantized_tensor(param.reshape(-1),self.num_bits))
        all_params=torch.cat(all_params,dim=0)
        return torch.mean(all_params) 
    
    def save_quanted_decoder_weights(self,save_path):
        ori_weight_dict=self.state_dict()
        quanted_weight_dict={}
        for key in ori_weight_dict.keys():
            quanted_weight_dict[key]=diff_quantized_tensor(ori_weight_dict[key],self.num_bits)
        torch.save(quanted_weight_dict,save_path)

class QuantDecoderSDF(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        decoder_dim_list=[int(decoder_dim) for decoder_dim in args.decoder_dim_list.split('_')]
        decoder_stride_list=[int(decoder_stride) for decoder_stride in args.decoder_stride_list.split('_')]

        embed_dim=args.embed_dim

        bias = args.bias             
        act=args.act                      
        conv_type=args.conv_type   

        after_embed_dim=args.after_embed_dim        

        num_bits=args.num_bits
        self.num_bits=args.num_bits

        decoder_layers_list=[]
        if after_embed_dim>0:
            decoder_layers_list.append(QuantConv3d(embed_dim,after_embed_dim,1,1,bias=bias,num_bits=num_bits))
        else:
            after_embed_dim=embed_dim

        for i in range(len(decoder_dim_list)):
            if i==0:
                in_channel=after_embed_dim
                out_channel=decoder_dim_list[i]
                scale=decoder_stride_list[i]
            else:
                in_channel=decoder_dim_list[i-1]
                out_channel=decoder_dim_list[i]
                scale=decoder_stride_list[i]
            decoder_layers_list.append(QuantNeRVBlock3D(in_channel,out_channel,scale,bias,act,conv_type,num_bits=num_bits))

        decoder_layers_list.append( QuantConv3d(decoder_dim_list[-1],1,3,1,1,bias=bias,num_bits=num_bits))
        self.decoder_layers=nn.Sequential(*decoder_layers_list)

    def forward(self,embed_features):
        #embed_features:    B,N,N,N,C
        embed_features=embed_features.permute(0,4,1,2,3)
        pred_voxel=self.decoder_layers(embed_features)  #(B,C,N,N,N)
        return pred_voxel.permute(0,2,3,4,1)        #(B,N,N,N,C)

    def get_quantparams(self):
        all_params=[]
        for param in self.parameters():
            all_params.append(diff_quantized_tensor(param.reshape(-1),self.num_bits))
        all_params=torch.cat(all_params,dim=0)
        return torch.mean(all_params) 
    
    def save_quanted_decoder_weights(self,save_path):
        ori_weight_dict=self.state_dict()
        quanted_weight_dict={}
        for key in ori_weight_dict.keys():
            quanted_weight_dict[key]=diff_quantized_tensor(ori_weight_dict[key],self.num_bits)
        torch.save(quanted_weight_dict,save_path)

def get_network(name,args):
    if name == "QuantDecoder":
        return QuantDecoder(args)
    elif name =='QuantDecoderSDF':
        return QuantDecoderSDF(args)
    else:
        assert False, 'no selected network !!!'

if __name__=='__main__':
    
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    from config_load import get_config
    args=get_config().parse_args()
    
    net=QuantDecoder(args).cuda()

    #net.save_quanted_decoder_weights('test.pt')

    #net.decoder.load_state_dict(torch.load('test.pt'))

    input=torch.rand(10,4,128,128,128).cuda()

    output=net(input)

    print(output[1].size())
    print(net.num_bits)