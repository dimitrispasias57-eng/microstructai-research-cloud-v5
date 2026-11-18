"""First-party synthetic raster geometry; not real cells or reserved data."""
import numpy as np

def ellipse(shape=(96,128), center=(55,43), radii=(21,12), angle=0):
    y,x=np.indices(shape);dx=x-center[0];dy=y-center[1]
    u=dx*np.cos(angle)+dy*np.sin(angle);v=-dx*np.sin(angle)+dy*np.cos(angle)
    return (u/radii[0])**2+(v/radii[1])**2<=1

def fixtures():
    f={}
    f['circle']=ellipse(radii=(15,15)).astype(np.uint16)
    f['ellipse']=ellipse().astype(np.uint16)
    f['elongated']=ellipse(radii=(29,5),angle=.43).astype(np.uint16)
    multi=np.zeros((128,160),np.uint16)
    for i,(c,r,t) in enumerate([((25,28),(13,8),0),((77,30),(16,6),.8),((128,85),(14,10),-1.1),((48,93),(8,8),0)],1):
        multi[ellipse(multi.shape,c,r,t)]=i*7
    f['different_sizes_orientations']=multi
    a=np.zeros((60,70),np.uint16);a[15:35,10:30]=1;a[15:35,30:50]=2
    f['touching_labelled']=a;f['touching_binary']=(a>0).astype(np.uint16)
    f['border_truncated']=ellipse(center=(2,43),radii=(21,12)).astype(np.uint16)
    a=np.zeros((50,60),np.uint16);a[10:38,12:22]=1;a[28:38,12:45]=1
    f['irregular']=a
    a=f['circle'].copy();a[40:46,52:58]=0;f['hole']=a
    a=np.zeros((20,20),np.uint16);a[5,5]=1;f['single_pixel']=a
    a=np.zeros((20,20),np.uint16);a[5,3:12]=1;f['line']=a
    a=np.zeros((20,20),np.uint16);a[5,5]=1;a[6,6]=1;f['diagonal_contact']=a
    a=np.zeros((20,20),np.uint16);a[3:6,3:6]=4;a[12:15,12:15]=4;f['disconnected_invalid']=a
    return f

def reference_grid():
    for radius in (7,13,22):
        for ratio in (.35,.7):
            for angle in (0,.31,.91,1.4):
                yield f'grid_r{radius}_ratio{ratio}_angle{angle}',ellipse(radii=(radius,radius*ratio),angle=angle).astype(np.uint16)
