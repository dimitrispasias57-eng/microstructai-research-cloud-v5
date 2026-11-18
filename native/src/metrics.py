"""Prospectively specified numerical agreement, never expert quality classifications."""
import numpy as np
from scipy import ndimage as ndi
from common import ROOT
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'frozen'))
from morphology.engine import measure

def cleanup(mask):
    mask=np.asarray(mask,bool);lab,n=ndi.label(mask,np.ones((3,3)))
    if not n:return mask
    centre=((mask.shape[0]-1)/2,(mask.shape[1]-1)/2);r,c=map(round,centre)
    target=int(lab[r,c])
    if not target:
        centres=ndi.center_of_mass(mask,lab,range(1,n+1));target=1+int(np.argmin([sum((a-b)**2 for a,b in zip(v,centre)) for v in centres]))
    return ndi.binary_fill_holes(lab==target)

def metrics(pred,ref,neighbours=None,raw=None):
    p=np.asarray(pred,bool);r=np.asarray(ref,bool);tp=int((p&r).sum());pa=int(p.sum());ra=int(r.sum());union=int((p|r).sum())
    d=2*tp/max(pa+ra,1);iou=tp/max(union,1);recall=tp/max(ra,1);precision=tp/max(pa,1)
    out=dict(dice=d,iou=iou,recall=recall,precision=precision,mask_available=bool(pa),area_relative_error=(pa-ra)/max(ra,1),area_absolute_relative_error=abs(pa-ra)/max(ra,1),severe_reference_disagreement=bool(d<.70 or recall<.75 or precision<.75),incomplete_footprint_flag=recall<.75,excess_background_flag=precision<.75,wrong_cell_suspect=bool(pa and tp==0))
    if pa:
        ps=p&~ndi.binary_erosion(p);rs=r&~ndi.binary_erosion(r);a=ndi.distance_transform_edt(~rs)[ps];b=ndi.distance_transform_edt(~ps)[rs]
        out.update(boundary_mean_distance_px=float((a.mean()+b.mean())/2),boundary_hausdorff95_px=float(np.percentile(np.r_[a,b],95)))
        pm=measure(p);rm=measure(r)
        out.update(perimeter_relative_error=(pm['perimeter_px']-rm['perimeter_px'])/rm['perimeter_px'],perimeter_absolute_relative_error=abs(pm['perimeter_px']-rm['perimeter_px'])/rm['perimeter_px'],circularity_absolute_error=abs(pm['circularity']-rm['circularity']),aspect_ratio_absolute_error=abs(pm['aspect_ratio']-rm['aspect_ratio']) if pm['aspect_ratio'] and rm['aspect_ratio'] else None)
    else:out.update(boundary_mean_distance_px=None,boundary_hausdorff95_px=None,perimeter_relative_error=-1.,perimeter_absolute_relative_error=1.,circularity_absolute_error=None,aspect_ratio_absolute_error=None)
    centre_region=ndi.binary_erosion(r,iterations=3)
    out['rim_only_suspect']=bool(pa and recall<.75 and centre_region.any() and (p&centre_region).sum()/centre_region.sum()<.5)
    out['multiple_raw_components']=bool(ndi.label(raw if raw is not None else p,np.ones((3,3)))[1]>1)
    out['merged_neighbour_suspect']=bool(neighbours is not None and pa and (p&neighbours).sum()/pa>.1)
    return out

def summary(rows):
    from collections import defaultdict
    by=defaultdict(list)
    for r in rows:by[r['participant']].append(r)
    numeric=['dice','iou','boundary_mean_distance_px','boundary_hausdorff95_px','area_absolute_relative_error','perimeter_absolute_relative_error','circularity_absolute_error','aspect_ratio_absolute_error','severe_reference_disagreement']
    def mean(rr,k):
        vals=[float(r[k]) for r in rr if r.get(k) is not None];return float(np.mean(vals)) if vals else None
    participant=[dict(participant=p,n=len(rr),**{k:mean(rr,k) for k in numeric}) for p,rr in sorted(by.items())]
    macro={k:mean(participant,k) for k in numeric};rng=np.random.default_rng(20261117);ci={}
    for k in numeric:
        vals=np.array([r[k] for r in participant if r[k] is not None])
        samples=rng.choice(vals,(2000,len(vals)),replace=True).mean(1) if len(vals) else np.array([])
        ci[k]=list(map(float,np.percentile(samples,[2.5,97.5]))) if len(samples) else None
    return dict(n=len(rows),n_participants=len(by),cell_mean={k:mean(rows,k) for k in numeric},participant_macro=macro,participant_bootstrap95=ci,by_participant=participant,warning='Agreement with published algorithmic reference, not manual segmentation ground truth. Cluster CI uses few participants.')
