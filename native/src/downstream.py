"""Native-to-frozen deterministic downstream adapter. No AI reviewer is substituted."""
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours
from common import ROOT
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'frozen'))
import geometry as frozen_geometry
import reserved_validation_rules_v1 as frozen_rules
from morphology.engine import measure

def run_downstream(image,mask):
    m=np.asarray(mask,bool)
    if not m.any():return m,dict(mask_available=False,status=frozen_rules.system_status('UNCERTAIN',False),qc_flags=['NO_MASK'],refinement=None,morphology=None)
    # Native interface adaptation is not a new frozen multimodal polygon response.
    contours=find_contours(m.astype(float),.5)
    if not contours or m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any():
        # No observed closed contour suitable for the frozen polygon refiner.
        # Preserve the mask as-is and require review; do not invent an outside edge.
        return m,dict(mask_available=True,status='HUMAN_REVIEW_REQUIRED_TECHNICAL',qc_flags=['NATIVE_CONTOUR_AT_CROP_EDGE'],refinement=None,morphology=measure(m))
    p=max(contours,key=len)[:,::-1]
    if not np.array_equal(p[0],p[-1]):p=np.vstack([p,p[0]])
    if frozen_geometry.self_intersects(p):
        return m,dict(mask_available=True,status='HUMAN_REVIEW_REQUIRED_TECHNICAL',qc_flags=['NATIVE_CONTOUR_SELF_INTERSECTION'],refinement=None,morphology=measure(m))
    q,ref=frozen_geometry.refine(image,p,'M2');final=frozen_geometry.raster(q,m.shape)
    # Exactly the unchanged post-refinement numerical tests from run_reserved.py.
    flags=[]
    if not .05<=final.mean()<=.95:flags.append('AREA_OUTSIDE_PRESPECIFIED_RANGE')
    if final.sum()<16:flags.append('TOO_FEW_PIXELS')
    if not frozen_geometry.inside(q,((m.shape[1]-1)/2,(m.shape[0]-1)/2)):flags.append('CROP_CENTRE_OUTSIDE')
    if ndi.label(final,np.ones((3,3)))[1]!=1:flags.append('NOT_ONE_COMPONENT')
    status=frozen_rules.system_status('SEGMENTATION_PRODUCED',bool(final.any()),ai_class=None,qc_flags=flags,fallback=ref['fallback_to_M1'])
    return final,dict(mask_available=bool(final.any()),status=status,qc_flags=flags,refinement=ref,morphology=measure(final) if final.any() else None,ai_reviewer='NOT_RUN_NO_NATIVE_SUBSTITUTE',diagnostics=frozen_geometry.diagnostics(image,final,q))
