"""Small from-scratch centre-conditioned segmentation network; no external weights."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU())
    def forward(self,x):return self.layers(x)

class NativeUNet(nn.Module):
    def __init__(self,width=8):
        super().__init__();self.width=width
        self.e1=Block(2,width);self.e2=Block(width,width*2);self.e3=Block(width*2,width*4);self.b=Block(width*4,width*8)
        self.d3=Block(width*12,width*4);self.d2=Block(width*6,width*2);self.d1=Block(width*3,width);self.out=nn.Conv2d(width,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));z=self.b(F.max_pool2d(c,2))
        z=self.d3(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        z=self.d2(torch.cat([F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        z=self.d1(torch.cat([F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(z)

def input_channels(gray):
    # Bx1x96x96; the coordinate prior is not derived from the reference mask.
    n=gray.shape[-1];axis=torch.arange(n,device=gray.device,dtype=gray.dtype)-(n-1)/2
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');prior=torch.exp(-(xx*xx+yy*yy)/(2*12**2))
    return torch.cat([gray,prior.expand(gray.shape[0],1,n,n)],1)

def augment(x,y):
    b=len(x);theta=torch.zeros((b,2,3),device=x.device)
    angle=(torch.rand(b,device=x.device)*30-15)*torch.pi/180
    flipx=torch.where(torch.rand(b,device=x.device)<.5,-1.,1.);flipy=torch.where(torch.rand(b,device=x.device)<.5,-1.,1.)
    theta[:,0,0]=angle.cos()*flipx;theta[:,0,1]=-angle.sin()*flipy
    theta[:,1,0]=angle.sin()*flipx;theta[:,1,1]=angle.cos()*flipy
    theta[:,:,2]=(torch.rand((b,2),device=x.device)*6-3)*2/96
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x,grid,mode='bilinear',padding_mode='reflection',align_corners=False)
    y=F.grid_sample(y,grid,mode='nearest',padding_mode='zeros',align_corners=False)
    contrast=.9+.2*torch.rand((b,1,1,1),device=x.device);brightness=.9+.2*torch.rand((b,1,1,1),device=x.device)
    x=((x-.5)*contrast+.5)*brightness
    x=x+torch.randn_like(x)*(.01*torch.rand((b,1,1,1),device=x.device))
    return x.clamp(0,1),y

def loss_fn(logits,y):
    p=logits.sigmoid();axes=(1,2,3)
    dice=(2*(p*y).sum(axes)+1)/(p.sum(axes)+y.sum(axes)+1)
    return .5*F.binary_cross_entropy_with_logits(logits,y)+.5*(1-dice).mean()
