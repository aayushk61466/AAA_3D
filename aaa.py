import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

def get_pad_layer(pad_type):
    if(pad_type in ['const','constant']):
        PadLayer = nn.ConstantPad3d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad3d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)

        if(self.filt_size==3):
            a = np.array([1., 2., 1.])


        filt_2d = torch.Tensor(a[:,None]*a[None,:])
        filt = torch.Tensor(filt_2d[:,:,None]*filt_2d[:,None,:]*filt_2d[None,:,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:,:].repeat((self.channels,1,1,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):

        return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
    
    
class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv3d(in_channels, group*kernel_size*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(group*kernel_size*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w,z = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w*z)

        n,c,h,w,z = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size*self.kernel_size,z*h*w))

        n,c1,p,q,r = x.shape
        x = x.permute(1,0,2,3,4).reshape(2, c1//2, n, p, q, r).permute(2,0,1,3,4,5)
        
        n,c2,p,q,r = sigma.shape
        sigma = sigma.permute(2,0,1,3,4).reshape((p//(self.kernel_size*self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size*self.kernel_size,n,c2,q.r)).permute(2,0,3,1,4,5)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w,z)
        return x[:,:,torch.arange(h)%self.stride==0,:,:][:,:,:,torch.arange(w)%self.stride==0,:][:,:,:,:,torch.arange(w)%self.stride==0]
