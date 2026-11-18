"""Frozen exploratory geometry/refinement. No MicroStructAI imports."""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.measure import perimeter_crofton,find_contours
from skimage.morphology import convex_hull_image
PARAMS={'version':'MULTIMODAL_BOUNDARY_REFINER_V1','points':128,'sigma_px':.7,'step_px':.5,'gradient_weight':.55,'contrast_weight':.25,'background_weight':.20,'displacement_penalty':.08,'inward_penalty':.12,'continuity_penalty':.20,'min_area_ratio':.90,'max_area_ratio':1.20,'min_iou':.80,'bands':{'M2':[-1,2],'NARROW':[-.5,1],'OUTWARD':[0,2]}}
def area(p):
    p=np.asarray(p,float);return .5*float(np.sum(p[:-1,0]*p[1:,1]-p[1:,0]*p[:-1,1]))
def cross(a,b,c):
    u,v=b-a,c-a;return float(u[0]*v[1]-u[1]*v[0])
def on(a,b,c):return abs(cross(a,b,c))<1e-8 and np.all(c>=np.minimum(a,b)-1e-8) and np.all(c<=np.maximum(a,b)+1e-8)
def intersects(a,b,c,d):
    x,y,z,t=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    return (x*y < -1e-12 and z*t < -1e-12) or any([on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)])
def self_intersects(p):
    n=len(p)-1
    for i in range(n):
        for j in range(i+2,n):
            if i==0 and j==n-1:continue
            # Reject via bounding boxes before exact intersection.
            if np.any(np.maximum(p[i],p[i+1])+1e-8<np.minimum(p[j],p[j+1])) or np.any(np.maximum(p[j],p[j+1])+1e-8<np.minimum(p[i],p[i+1])):continue
            if intersects(p[i],p[i+1],p[j],p[j+1]):return True
    return False
def inside(p,xy):
    x,y=xy;result=False
    for a,b in zip(p[:-1],p[1:]):
        if on(a,b,np.array(xy)):return True
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:result=not result
    return result
def raster(p,shape):
    m=np.zeros(shape,bool);rr,cc=polygon(np.asarray(p)[:,1],np.asarray(p)[:,0],shape);m[rr,cc]=True;return m
def validate_response(r,shape):
    errors=[];flags=[];h,w=shape
    if r['status']!='BOUNDARY_PROPOSED':
        if r['polygon_points']:errors.append('ABSTENTION_WITH_COORDINATES')
        if not r['source_unusable_reason'].strip():errors.append('MISSING_ABSTENTION_REASON')
        return {'geometry_valid':not errors,'mask_available':False,'errors':errors,'qc_flags':flags},None
    q=np.array([[a['x'],a['y']] for a in r['polygon_points']],float)
    if q.ndim!=2 or q.shape[1]!=2 or len(q)<4:return {'geometry_valid':False,'mask_available':False,'errors':['MALFORMED_POLYGON'],'qc_flags':[]},None
    if not 25<=len(q)<=41:errors.append('VERTEX_COUNT_OUTSIDE_PROTOCOL')
    if not np.isfinite(q).all() or np.any(q<0) or np.any(q>1):errors.append('OUT_OF_RANGE')
    if not np.array_equal(q[0],q[-1]):errors.append('NOT_EXPLICITLY_CLOSED')
    if np.any(np.linalg.norm(np.diff(q,axis=0),axis=1)<1e-8):errors.append('ZERO_LENGTH_EDGE')
    if len({tuple(x) for x in q[:-1]})!=len(q)-1:errors.append('REPEATED_NONCLOSING_VERTEX')
    p=q*np.array([w-1,h-1])
    if self_intersects(p):errors.append('SELF_INTERSECTION')
    if errors:return {'geometry_valid':False,'mask_available':False,'errors':errors,'qc_flags':flags},None
    m=raster(p,shape);af=m.mean()
    if not .05<=af<=.95:flags.append('AREA_OUTSIDE_PRESPECIFIED_RANGE')
    if m.sum()<16:flags.append('TOO_FEW_PIXELS')
    if not inside(p,((w-1)/2,(h-1)/2)):flags.append('CROP_CENTRE_OUTSIDE')
    if ndi.label(m,np.ones((3,3)))[1]!=1:flags.append('NOT_ONE_COMPONENT')
    return {'geometry_valid':True,'mask_available':True,'errors':[],'qc_flags':flags,'area_fraction':float(af)},p
def resample(p,n=128):
    ds=np.linalg.norm(np.diff(p,axis=0),axis=1);s=np.r_[0,np.cumsum(ds)];t=np.linspace(0,s[-1],n,endpoint=False)
    return np.column_stack([np.interp(t,s,p[:,0]),np.interp(t,s,p[:,1])])
def normals(q):
    tangent=np.roll(q,-1,axis=0)-np.roll(q,1,axis=0);lens=np.linalg.norm(tangent,axis=1);tangent=tangent/np.maximum(lens[:,None],1e-12)
    sign=1 if area(np.vstack([q,q[0]]))>=0 else -1
    return sign*np.column_stack([tangent[:,1],-tangent[:,0]])
def sample(im,p):return ndi.map_coordinates(im,[p[...,1].ravel(),p[...,0].ravel()],order=1,mode='nearest').reshape(p.shape[:-1])
def field(image):
    z=np.asarray(image,float)
    if z.ndim==3:z=z[...,:3].mean(axis=2)
    z=ndi.gaussian_filter(z,PARAMS['sigma_px']);gy,gx=np.gradient(z)
    gm=max(float(np.percentile(np.hypot(gx,gy),95)),1e-6);rng=max(float(np.percentile(z,90)-np.percentile(z,10)),1e-6)
    border=np.zeros(z.shape,bool);border[:2]=True;border[-2:]=True;border[:,:2]=True;border[:,-2:]=True
    return z,gx,gy,gm,rng,float(np.median(z[border]))
def refine(image,p,variant):
    shape=image.shape[:2];h,w=shape;m0=raster(p,shape);q=resample(p);n=normals(q);lo,hi=PARAMS['bands'][variant];offsets=np.arange(lo,hi+.001,.5)
    positions=q[:,None,:]+n[:,None,:]*offsets[None,:,None];z,gx,gy,gm,rng,bg=field(image)
    grad=np.abs(sample(gx,positions)*n[:,None,0]+sample(gy,positions)*n[:,None,1])/gm
    outer=sample(z,positions+n[:,None,:]);inner=sample(z,positions-n[:,None,:]);contrast=np.abs(outer-inner)/rng
    background=1-np.minimum(np.abs(outer-bg)/rng,1)
    score=.55*np.minimum(grad,2)+.25*np.minimum(contrast,2)+.20*background-.08*np.abs(offsets)[None,:]-.12*np.maximum(-offsets,0)[None,:]
    valid=(positions[...,0]>=0)&(positions[...,0]<=w-1)&(positions[...,1]>=0)&(positions[...,1]<=h-1);score[~valid]=-1e9
    k=len(offsets);pen=.2*(offsets[:,None]-offsets[None,:])**2;best=(-np.inf,None)
    for start in range(k):
        dp=np.full(k,-np.inf);dp[start]=score[0,start];backs=[]
        for i in range(1,len(q)):
            candidates=dp[:,None]-pen;back=np.argmax(candidates,axis=0);dp=candidates[back,np.arange(k)]+score[i];backs.append(back)
        end=int(np.argmax(dp-pen[:,start]));value=float(dp[end]-pen[end,start]);seq=[end]
        for back in reversed(backs):seq.append(int(back[seq[-1]]))
        seq=seq[::-1]
        if value>best[0]:best=(value,seq)
    d=offsets[np.array(best[1])];d=(np.roll(d,1)+2*d+np.roll(d,-1))/4;new=q+n*d[:,None];new=np.vstack([new,new[0]])
    reasons=[];m=raster(new,shape);ratio=float(m.sum()/max(m0.sum(),1));iou=float((m&m0).sum()/max((m|m0).sum(),1))
    if np.any(new<0) or np.any(new>np.array([w-1,h-1])):reasons.append('OUT_OF_BOUNDS')
    if self_intersects(new):reasons.append('SELF_INTERSECTION')
    if inside(p,((w-1)/2,(h-1)/2)) and not inside(new,((w-1)/2,(h-1)/2)):reasons.append('CENTRE_LOST')
    if ndi.label(m,np.ones((3,3)))[1]!=1:reasons.append('COMPONENT_COUNT')
    if not .90<=ratio<=1.20:reasons.append('AREA_RATIO_GUARD')
    if iou<.80:reasons.append('IOU_GUARD')
    rec={'variant':variant,'fallback_to_M1':bool(reasons),'fallback_reasons':reasons,'attempted_area_ratio':ratio,'attempted_iou_to_M1':iou,'offset_min_px':float(d.min()),'offset_max_px':float(d.max()),'offset_mean_px':float(d.mean()),'attempted_contour':new.tolist()}
    return (p.copy() if reasons else new),rec
def diagnostics(image,m,p=None):
    m=np.asarray(m,bool);h,w=m.shape;count=int(m.sum());components=ndi.label(m,np.ones((3,3)))[1]
    cy,cx=(h-1)/2,(w-1)/2
    angles=np.arange(72)*2*np.pi/72;radii=np.arange(1,np.hypot(h,w),.5)
    rayx=np.rint(cx+np.cos(angles[:,None])*radii).astype(int);rayy=np.rint(cy+np.sin(angles[:,None])*radii).astype(int)
    valid=(rayx>=0)&(rayx<w)&(rayy>=0)&(rayy<h)
    bins=(m[np.clip(rayy,0,h-1),np.clip(rayx,0,w-1)]&valid).any(axis=1)
    longest=run=0
    for bit in np.r_[bins,bins]:
        run=0 if bit else run+1;longest=max(longest,run)
    hull=convex_hull_image(m) if count else m;per=float(perimeter_crofton(m,4));hp=float(perimeter_crofton(hull,4));edge=m&~ndi.binary_erosion(m)
    rim=np.zeros(m.shape,bool);rim[0]=rim[-1]=True;rim[:,0]=rim[:,-1]=True
    if p is None:
        contours=find_contours(m.astype(float),.5);p=max(contours,key=len)[:,::-1] if contours else np.empty((0,2))
        if len(p) and not np.array_equal(p[0],p[-1]):p=np.vstack([p,p[0]])
    alignment=consistency=smooth=None
    if len(p)>3:
        q=resample(p);n=normals(q);z,gx,gy,gm,rng,bg=field(image);dx=sample(gx,q);dy=sample(gy,q);g=np.hypot(dx,dy)
        alignment=float(np.mean(np.abs(dx*n[:,0]+dy*n[:,1])/np.maximum(g,1e-8)))
        diff=sample(z,q+n)-sample(z,q-n);eligible=np.abs(diff)>.02*rng
        consistency=float(max(np.mean(diff[eligible]>0),np.mean(diff[eligible]<0))) if eligible.any() else None
        v=np.roll(q,-1,axis=0)-q;theta=np.arctan2(v[:,1],v[:,0]);turn=np.angle(np.exp(1j*(np.roll(theta,-1)-theta)));smooth=float(np.mean(turn**2))
    return {'area_px':count,'area_fraction':float(m.mean()),'perimeter_crofton_px':per,'solidity':float(count/hull.sum()) if hull.any() else None,'convexity_hull_perimeter_ratio':hp/per if per else None,'crop_centre_inside':bool(m[int(round(cy)),int(round(cx))]),'angular_coverage_fraction_72bins':float(bins.mean()),'maximum_angular_gap_degrees':min(longest,72)*5,'connected_components':int(components),'border_contact_pixels':int((m&rim).sum()),'contour_turn_mean_square_rad2':smooth,'normal_gradient_alignment':alignment,'transition_sign_consistency':consistency}
