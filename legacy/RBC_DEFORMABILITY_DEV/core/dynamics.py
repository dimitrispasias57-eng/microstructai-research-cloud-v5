"""Supplied tracked-mask dynamics. No segmentation, classifier or app integration."""
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.util
import math
import sys
import numpy as np
from scipy import ndimage

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK/'RBC_MORPHOLOGY_DEV'))
from morphology.engine import measure
spec = importlib.util.spec_from_file_location('frozen_rbc_temporal', WORK/'RBC_TEMPORAL_ANALYSIS_DEV/core/temporal.py')
temporal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = temporal
spec.loader.exec_module(temporal)
from . import VERSION

metric = temporal.metric
theil_sen = temporal.theil_sen

@dataclass(frozen=True)
class Config:
    track_id: str = ''
    identity_confirmed: bool = False
    um_per_pixel: float | None = None
    calibration_source: str = ''
    mechanical_condition: str = ''
    baseline_window: tuple = (0., 2.)
    load_onset: float | None = None
    rise_window: tuple | None = None
    plateau_window: tuple | None = None
    release: float | None = None
    recovery_window: tuple | None = None
    deformation_rate_window: tuple | None = None
    recovery_rate_window: tuple | None = None
    angular_speed_bound: float | None = None
    shape_oscillatory: bool = False
    orientation_oscillatory: bool = False

def _number(v):
    return isinstance(v, (int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) and math.isfinite(v)

def _validate_times(times):
    if len(times)<3 or not all(_number(v) for v in times):
        raise ValueError('At least three finite timestamps in explicit seconds are required.')
    t=np.asarray(times,dtype=float)
    if np.any(np.diff(t)<=0): raise ValueError('Timestamps must be strictly increasing; duplicates are not sorted away.')
    return t

def _validate_config(c):
    if not isinstance(c,Config) or not c.track_id.strip() or c.identity_confirmed is not True:
        raise ValueError('A stable track ID and explicit identity confirmation are required.')
    if c.um_per_pixel is not None and (not _number(c.um_per_pixel) or c.um_per_pixel<=0 or not c.calibration_source.strip()):
        raise ValueError('Calibration requires a positive finite isotropic scale and source.')
    for key in ('baseline_window','rise_window','plateau_window','recovery_window','deformation_rate_window','recovery_rate_window'):
        w=getattr(c,key)
        if w is not None and (len(w)!=2 or not all(_number(v) for v in w) or w[1]<=w[0]):
            raise ValueError('Invalid declared window: '+key)
    for key in ('load_onset','release','angular_speed_bound'):
        x=getattr(c,key)
        if x is not None and not _number(x): raise ValueError('Invalid '+key)
    if c.angular_speed_bound is not None and c.angular_speed_bound<=0: raise ValueError('Angular bound must be positive.')
    if c.load_onset is not None and c.baseline_window[1]>c.load_onset:
        raise ValueError('Baseline must precede loading.')
    if c.rise_window is not None and (c.load_onset is None or c.rise_window[0]!=c.load_onset):
        raise ValueError('Rise phase must start at declared loading event.')
    if c.plateau_window is not None and c.rise_window is not None and c.plateau_window[0]<c.rise_window[1]:
        raise ValueError('Plateau window must not precede rise completion.')
    if c.release is not None and c.load_onset is not None and c.release<=c.load_onset:
        raise ValueError('Release must follow loading.')
    if c.release is not None and c.plateau_window is not None and c.release<c.plateau_window[1]:
        raise ValueError('Release must follow the declared rise plateau window.')
    if c.recovery_window is not None and (c.release is None or c.recovery_window[0]!=c.release):
        raise ValueError('Recovery phase must start at declared release.')

def masks_to_rows(times,masks,c,tracking_ok=None):
    _validate_config(c); t=_validate_times(times)
    if len(masks)!=len(t): raise ValueError('Mask/timestamp length mismatch.')
    if tracking_ok is None: tracking_ok=[True]*len(t)
    if len(tracking_ok)!=len(t) or any(type(x) is not bool for x in tracking_ok):
        raise ValueError('Tracking flags must be an explicit boolean per frame.')
    rows=[]; shape=None
    for time,mask,tracked in zip(t,masks,tracking_ok):
        row=dict(time_s=float(time),track_id=c.track_id,tracking_ok=tracked,eligible=False,qc=[])
        if not tracked: row['qc'].append('UNSTABLE_TRACKING')
        if mask is None:
            row['qc'].append('MISSING_FRAME'); rows.append(row); continue
        a=np.asarray(mask)
        if a.ndim!=2 or a.dtype.kind not in 'bui' or not a.size or a.size>1_000_000 or np.any((a!=0)&(a!=1)):
            row['qc'].append('INVALID_BINARY_MASK'); rows.append(row); continue
        if shape is None: shape=a.shape
        if a.shape!=shape:
            row['qc'].append('CHANGING_IMAGE_GEOMETRY'); rows.append(row); continue
        b=a.astype(bool)
        if not b.any(): row['qc'].append('EMPTY_MASK')
        elif b[0].any() or b[-1].any() or b[:,0].any() or b[:,-1].any(): row['qc'].append('BORDER_CLIPPED')
        elif ndimage.label(b,structure=np.ones((3,3)))[1]!=1: row['qc'].append('DISCONNECTED_MASK')
        else:
            m=measure(b); row.update(m)
            L,W=m['major_axis_px'],m['minor_axis_px']
            if W<=0: row['qc'].append('DEGENERATE_AXES')
            else:
                row['image_derived_DI']=(L-W)/(L+W)
                row['eligible']=tracked
                row['orientation_eligible']=m['eccentricity']>=.20 and m['orientation_rad'] is not None
                if not row['orientation_eligible']: row['qc'].append('NEAR_ISOTROPIC_ORIENTATION')
                if c.um_per_pixel is not None:
                    for k in ('major_axis','minor_axis','perimeter'): row[k+'_um']=m[k+'_px']*c.um_per_pixel
                    row['area_um2']=m['area_px2']*c.um_per_pixel**2
        rows.append(row)
    return rows

def _window(t,y,window,dt):
    if window is None: return [],['WINDOW_NOT_DECLARED']
    start,end=window
    idx=np.where((t>=start-1e-10)&(t<=end+1e-10))[0].tolist()
    if len(idx)<3 or t[idx[0]]>start+1e-8 or t[idx[-1]]<end-1e-8 or (t[idx[-1]]-t[idx[0]])<.8*(end-start):
        return idx,['INSUFFICIENT_WINDOW_COVERAGE']
    if any(y[i] is None for i in idx) or any(t[j]-t[i]>3*dt+1e-10 for i,j in zip(idx,idx[1:])):
        return idx,['MISSING_OR_LONG_GAP_IN_WINDOW']
    return idx,[]

def _cross(t,y,idx,target,rising):
    a=y[idx[0]]
    if (a>=target if rising else a<=target): return None,'LEFT_CENSORED_CROSSING'
    for i,j in zip(idx,idx[1:]):
        yi,yj=y[i],y[j]
        if (yi<target<=yj if rising else yi>target>=yj):
            return float(t[i]+(target-yi)*(t[j]-t[i])/(yj-yi)),None
    return None,'CROSSING_NOT_OBSERVED'

def _reverse(y,idx,rising):
    d=np.diff([y[i] for i in idx]); total=np.abs(d).sum()
    return float(np.maximum(-d if rising else d,0).sum()/total) if total else 0.

def _slope(t,y,window,dt,reasons=()):
    idx,why=_window(t,y,window,dt); why=list(reasons)+why
    if why: return metric(units='DI/s',reason=';'.join(why),method='Theil-Sen, declared window')
    return metric(float(theil_sen([float(t[i]) for i in idx],[y[i] for i in idx])),units='DI/s',method='Theil-Sen, declared window')

def _period(t,y,declared,dt,units='s'):
    if not declared: return metric(units=units,reason='OSCILLATORY_DESIGN_NOT_DECLARED')
    if any(v is None for v in y) or np.any(np.diff(t)>3*dt+1e-10):
        return metric(units=units,reason='INCOMPLETE_PERIOD_SUPPORT')
    if max(y)-min(y)<.01: return metric(units=units,reason='LOW_DYNAMIC_RANGE')
    centre=float(np.mean(y)); crossings=[]
    for i in range(len(t)-1):
        if y[i]<centre<=y[i+1]: crossings.append(t[i]+(centre-y[i])*(t[i+1]-t[i])/(y[i+1]-y[i]))
    if len(crossings)<3: return metric(units=units,reason='FEWER_THAN_TWO_COMPLETE_CYCLES')
    periods=np.diff(crossings); med=float(np.median(periods))
    if (max(periods)-min(periods))/med>.20: return metric(units=units,reason='NONSTATIONARY_CYCLE_INTERVALS')
    return metric(med,units=units,method='Median upward mean-crossing interval',warnings=['DESCRIPTIVE_PERIOD_NOT_MECHANISM'])

def orientation(t,rows,c,dt):
    raw=[r.get('orientation_rad') if r.get('eligible') and r.get('orientation_eligible') else None for r in rows]
    unwrapped=[None]*len(t); speed=[None]*len(t); reasons=[]
    if c.angular_speed_bound is None: reasons.append('ANGULAR_ALIAS_BOUND_NOT_ESTABLISHED')
    for i,angle in enumerate(raw):
        if angle is None: continue
        if i==0 or raw[i-1] is None or t[i]-t[i-1]>3*dt+1e-10:
            unwrapped[i]=angle; continue
        step=float((angle-raw[i-1]+math.pi/2)%math.pi-math.pi/2)
        interval=t[i]-t[i-1]
        if c.angular_speed_bound is None or c.angular_speed_bound*interval>=math.pi/2:
            unwrapped[i]=angle
            if c.angular_speed_bound is not None: reasons.append('ANGULAR_UNDERSAMPLING')
            continue
        if abs(step)/interval>c.angular_speed_bound+1e-10:
            unwrapped[i]=angle;reasons.append('ANGULAR_BOUND_VIOLATION');continue
        unwrapped[i]=unwrapped[i-1]+step;speed[i]=step/interval
    complete=all(v is not None for v in unwrapped) and all(v is not None for v in speed[1:]) and not reasons
    avg=(unwrapped[-1]-unwrapped[0])/(t[-1]-t[0]) if complete else None
    return dict(axial_angle_rad=raw,unwrapped_angle_rad=unwrapped,angular_velocity_rad_s=speed,
                mean_angular_velocity=metric(avg,units='rad/s',reason=None if complete else ';'.join(reasons or ['INCOMPLETE_ORIENTATION_SUPPORT']),method='Net supported unwrapped axial change / duration'),
                axial_turn_rate=metric(abs(avg)/math.pi if avg is not None else None,units='axial turns/s',reason=None if complete else 'UNQUALIFIED_ORIENTATION',warnings=['NOT_BIOLOGICAL_FLIPPING_FREQUENCY']),
                oscillation_period=_period(t,unwrapped,c.orientation_oscillatory,dt) if complete else metric(units='s',reason='UNQUALIFIED_ORIENTATION'),
                biological_flipping_frequency=metric(units='Hz',reason='REQUIRES_INDEPENDENT_FLIPPING_LABEL_ADAPTER'),
                tank_treading_period=metric(units='s',reason='MEMBRANE_MATERIAL_MOTION_NOT_OBSERVED'),
                translation_speed=metric(units='um/s',reason='COMMON_SPATIAL_COORDINATE_FRAME_NOT_SUPPLIED'))

def analyze_rows(times,rows,c):
    _validate_config(c); t=_validate_times(times)
    if len(rows)!=len(t): raise ValueError('Row/timestamp length mismatch.')
    for ti,row in zip(t,rows):
        if row.get('track_id')!=c.track_id or abs(row.get('time_s',float('inf'))-ti)>1e-10:
            raise ValueError('Track identity or row timestamp mismatch.')
    y=[float(r['image_derived_DI']) if r.get('eligible') and _number(r.get('image_derived_DI')) else None for r in rows]
    if any(v is not None and not 0<=v<1 for v in y): raise ValueError('DI must be in [0,1).')
    dt=float(np.median(np.diff(t)))
    qc=list(dict.fromkeys(q for r in rows for q in r.get('qc',[])))
    finite=[i for i,v in enumerate(y) if v is not None]
    missing=len(finite)!=len(t)
    if missing: qc.append('MISSING_OR_INVALID_FRAME')
    if np.any(np.diff(t)>3*dt+1e-10): qc.append('LONG_TIME_GAP')
    if any(not r.get('tracking_ok',False) for r in rows): qc.append('UNSTABLE_TRACKING')
    areas=[r['area_px2'] for r in rows if r.get('eligible') and _number(r.get('area_px2'))]
    area_change=float((max(areas)-min(areas))/np.median(areas)) if areas and np.median(areas)>0 else None
    if area_change is not None and area_change>.10: qc.append('AREA_CHANGE_REQUIRES_REVIEW')
    block=[x for x in qc if x in ('UNSTABLE_TRACKING','AREA_CHANGE_REQUIRES_REVIEW','CHANGING_IMAGE_GEOMETRY')]
    biological_block=block+([] if c.mechanical_condition.strip() else ['MECHANICAL_CONDITION_NOT_SUPPLIED'])
    bi,bwhy=_window(t,y,c.baseline_window,dt)
    B=float(np.median([y[i] for i in bi])) if not bwhy and not block else None
    m={'baseline':metric(B,units='DI',reason=';'.join(bwhy+block) or None,method='Pre-load window median')}
    peak=max((y[i] for i in finite),default=None)
    pi=next((i for i in finite if y[i]==peak),None)
    observation_warning=['OBSERVED_SUBSET_ONLY'] if missing or 'LONG_TIME_GAP' in qc else []
    m['maximum_observed_DI']=metric(peak,units='DI',method='Maximum valid observed DI',warnings=observation_warning)
    m['delta_observed_DI']=metric(peak-B if peak is not None and B is not None else None,units='DI',reason=None if B is not None else 'BASELINE_UNAVAILABLE',warnings=observation_warning)
    m['time_of_maximum']=metric(float(t[pi]) if pi is not None else None,units='s',method='Earliest observed maximum timestamp',warnings=observation_warning)
    valid_peak_event=pi is not None and c.load_onset is not None and t[pi]>=c.load_onset and not biological_block
    m['time_to_maximum']=metric(float(t[pi]-c.load_onset) if valid_peak_event else None,units='s',reason=None if valid_peak_event else 'LOADING_EVENT_OR_PEAK_UNQUALIFIED',warnings=observation_warning)
    supported=[i for i in range(len(t)-1) if y[i] is not None and y[i+1] is not None and t[i+1]-t[i]<=3*dt+1e-10]
    auc=float(sum((y[i]+y[i+1])*.5*(t[i+1]-t[i]) for i in supported)) if supported else None
    duration=float(sum(t[i+1]-t[i] for i in supported));full=len(supported)==len(t)-1 and not block
    m['observed_segment_auc']=metric(auc,units='DI*s',warnings=[] if full else ['INCOMPLETE_OR_UNQUALIFIED_TRACK'],method='Supported trapezoids only')
    m['integrated_duration']=metric(duration,units='s')
    m['trajectory_auc']=metric(auc if full else None,units='DI*s',reason=None if full else 'INCOMPLETE_OR_UNQUALIFIED_TRACK')
    m['baseline_adjusted_auc']=metric(auc-B*duration if full and B is not None else None,units='DI*s',reason=None if full and B is not None else 'INCOMPLETE_TRACK_OR_BASELINE')
    deformation_block=biological_block.copy(); recovery_block=biological_block.copy()
    if c.load_onset is None: deformation_block.append('LOADING_EVENT_NOT_SUPPLIED')
    elif c.deformation_rate_window is not None and (c.deformation_rate_window[0]<c.load_onset or (c.release is not None and c.deformation_rate_window[1]>c.release)):
        deformation_block.append('RATE_WINDOW_OUTSIDE_LOADING_PHASE')
    if c.release is None: recovery_block.append('RELEASE_EVENT_NOT_SUPPLIED')
    elif c.recovery_rate_window is not None and c.recovery_rate_window[0]<c.release:
        recovery_block.append('RATE_WINDOW_PRECEDES_RELEASE')
    m['deformation_rate']=_slope(t,y,c.deformation_rate_window,dt,deformation_block)
    m['initial_deformation_rate']=dict(m['deformation_rate']) if c.deformation_rate_window is not None and c.deformation_rate_window[0]==c.load_onset else metric(units='DI/s',reason='RATE_WINDOW_NOT_AT_LOADING_ONSET')
    m['recovery_rate']=_slope(t,y,c.recovery_rate_window,dt,recovery_block)
    edge_rates=[(y[i+1]-y[i])/(t[i+1]-t[i]) for i in supported]
    m['maximum_adjacent_rate']=metric(float(max(edge_rates)) if edge_rates and not block else None,units='DI/s',reason='UNQUALIFIED_TRACK' if block else None,warnings=['NOISE_SENSITIVE_ADJACENT_RATE'])
    m['minimum_adjacent_rate']=metric(float(min(edge_rates)) if edge_rates and not block else None,units='DI/s',reason='UNQUALIFIED_TRACK' if block else None,warnings=['NOISE_SENSITIVE_ADJACENT_RATE'])
    # Rise kinetics: no last-value normalization and no fractional times on cycles.
    ri,rwhy=_window(t,y,c.rise_window,dt); pidx,pwhy=_window(t,y,c.plateau_window,dt)
    rwhy+=pwhy+biological_block
    P=float(np.median([y[i] for i in pidx])) if not pwhy else None
    amplitude=P-B if P is not None and B is not None else None
    if amplitude is None or amplitude<.01: rwhy.append('LOW_RANGE_OR_NO_BASELINE')
    if not rwhy:
        spread=max(y[i] for i in pidx)-min(y[i] for i in pidx)
        drift=abs(theil_sen([float(t[i]) for i in pidx],[y[i] for i in pidx]))*(t[pidx[-1]]-t[pidx[0]])
        if spread/amplitude>.10 or drift/amplitude>.05: rwhy.append('RISE_PLATEAU_UNQUALIFIED')
        if _reverse(y,ri,True)>.20: rwhy.append('NONMONOTONIC_RISE_PHASE')
    m['rise_plateau']=metric(P if not rwhy else None,units='DI',reason=';'.join(rwhy) or None)
    for q in (.1,.5,.9):
        value,reason=(None,';'.join(rwhy)) if rwhy else _cross(t,y,ri,B+q*amplitude,True)
        m[f'rise_t{int(100*q)}']=metric(value-c.load_onset if value is not None else None,units='s',reason=reason,method='Fraction of qualified rise amplitude from loading event')
    # Recovery uses a pre-load baseline, not a post-hoc tail-defined target.
    rec,why=_window(t,y,c.recovery_window,dt);why+=biological_block
    amp=y[rec[0]]-B if rec and y[rec[0]] is not None and B is not None else None
    if amp is None or amp<.01: why.append('LOW_RANGE_OR_NO_BASELINE')
    if not why and _reverse(y,rec,False)>.20: why.append('NONMONOTONIC_RECOVERY_PHASE')
    for q in (.1,.5,.9):
        value,reason=(None,';'.join(why)) if why else _cross(t,y,rec,B+(1-q)*amp,False)
        m[f'recovery_t{int(100*q)}']=metric(value-c.release if value is not None else None,units='s',reason=reason,method='Observed fraction of return toward pre-load baseline')
    recovery=None;reason=';'.join(why) or None
    if not why:
        # Find the first sustained entry, not merely the first transient crossing.
        entries=[];lo=B-.05*amp;hi=B+.05*amp
        for i,j in zip(rec,rec[1:]):
            target=hi if y[i]>hi and y[j]<=hi else (lo if y[i]<lo and y[j]>=lo else None)
            if target is not None: entries.append(float(t[i]+(target-y[i])*(t[j]-t[i])/(y[j]-y[i])))
        reason='INCOMPLETE_OR_TRUNCATED_RECOVERY'
        for crossing in entries:
            tail=[i for i in rec if t[i]>=crossing-1e-10]
            if tail and t[rec[-1]]-crossing>=2-1e-10 and all(lo-1e-10<=y[i]<=hi+1e-10 for i in tail):
                recovery=float(crossing-c.release);reason=None;break
    if recovery is None and c.release is not None: qc.append('INCOMPLETE_OR_UNQUALIFIED_RECOVERY')
    m['recovery_time_95']=metric(recovery,units='s',reason=reason,method='95% return to baseline sustained >=2s and through phase end')
    m['shape_oscillation_period']=_period(t,y,c.shape_oscillatory,dt) if not block else metric(units='s',reason='UNQUALIFIED_TRACK')
    if peak is not None and B is not None and peak-B<.01: qc.append('LOW_DYNAMIC_RANGE')
    if any('INSUFFICIENT_WINDOW_COVERAGE' in reason for item in m.values() for reason in item['reason_codes']): qc.append('INSUFFICIENT_DURATION_OR_WINDOW_COVERAGE')
    return dict(version=VERSION,config=asdict(c),time_unit='s',rows=rows,metrics=m,qc=list(dict.fromkeys(qc)),
                area_range_over_median=area_change,orientation=orientation(t,rows,c,dt),
                notes=['DEVELOPMENT; synthetic mathematical validation only.',
                       'DI is MicroStructAI image-derived deformation index, not a LORRCA index.',
                       'No clinical deformability, diagnosis, disease classification or tank-treading inference.'])

def analyze_masks(times,masks,c,tracking_ok=None):
    return analyze_rows(times,masks_to_rows(times,masks,c,tracking_ok),c)
