"""CPU-only application adapter. Scientific modules are copied byte-for-byte."""
import hashlib, io, json, sys, threading, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from metrics import cleanup
from downstream import run_downstream
MODEL_LOCK=threading.Lock()

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def verify_science():
    manifest=json.loads((ROOT/'SCIENTIFIC_MANIFEST.json').read_text())
    for rel,expected in manifest['files'].items():
        if sha_bytes((ROOT/rel).read_bytes())!=expected:raise RuntimeError('Scientific integrity check failed: '+rel)
    return manifest

def load_model():
    verify_science();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return torch.jit.load(str(ROOT/'model/native_rbc_v1.ts'),map_location='cpu').eval()

def decode_image(payload):
    if len(payload)>1024*1024:raise ValueError('Maximum file size is 1 MiB.')
    with Image.open(io.BytesIO(payload)) as im:
        if getattr(im,'n_frames',1)!=1 or im.size!=(96,96):raise ValueError('Require one native 96 x 96 crop. No resizing, cropping or stack flattening is performed.')
        if im.mode=='RGB':
            a=np.asarray(im)
            if not (np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])):raise ValueError('Colour channels differ. This native model is validated only for grayscale DPC crops; no channel conversion is assumed.')
            a=a[:,:,0].copy()
        elif im.mode=='L':a=np.asarray(im).copy()
        else:raise ValueError('Require native 8-bit grayscale (or identical RGB channels).')
    if a.dtype!=np.uint8:raise ValueError('Require native uint8 image.')
    return a

def infer(image,model):
    start=time.perf_counter()
    with MODEL_LOCK,torch.inference_mode():
        prob=model(torch.tensor(image[None,None],dtype=torch.float32)/255).numpy()[0,0].copy()
    raw=prob>=.5;clean=cleanup(raw);mask,details=run_downstream(image,clean)
    return {'raw':raw,'cleaned':clean,'mask':mask,'details':details,'probability':prob,'seconds':time.perf_counter()-start}

def png_bytes(array):
    b=io.BytesIO();Image.fromarray(array).save(b,format='PNG');return b.getvalue()

def panel(image,mask):
    from scipy.ndimage import binary_erosion
    rgb=np.repeat(image[:,:,None],3,axis=2);overlay=rgb.copy()
    overlay[mask]=(0.55*rgb[mask]+0.45*np.array([0,210,255])).astype(np.uint8)
    contour=rgb.copy();contour[mask&~binary_erosion(mask)]=[255,195,30]
    return np.concatenate([rgb,np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,2),overlay,contour],axis=1)
