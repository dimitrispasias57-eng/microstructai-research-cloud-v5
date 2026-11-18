"""Generate repeatable synthetic-only audit artifacts; no dataset access."""
from pathlib import Path
import hashlib
import json
import math
import platform
import statistics
import unittest
import numpy as np
from PIL import Image
from .engine import analyze,Config,METRICS
from .reference import reference_measure,reference_quantile
from .fixtures import fixtures,reference_grid
from .io import csv_bytes,json_bytes,export_bundle
from .plots import overlay,distributions,png_bytes
from . import VERSION

def main():
    root=Path(__file__).resolve().parents[1];out=root/'reports'/'synthetic_evidence';out.mkdir(parents=True,exist_ok=True)
    cases={**fixtures(),**dict(reference_grid())};rows=[];references=[];actuals=[];undefined=0
    for name,a in cases.items():
        if name=='disconnected_invalid':continue
        result=analyze(a,Config(identity_confirmed=True,image_id=name,sample_id='synthetic-case-'+name),{'type':'first-party synthetic'})
        for r in result['instances']:
            if not r['included']:continue
            ref=reference_measure((a==r['instance_id']).tolist());references.append(ref);actuals.append(r)
            for metric in METRICS:
                v=r[metric];expected=ref[metric];missing=v is None or expected is None
                error=None if missing else (abs((v-expected+math.pi/2)%math.pi-math.pi/2) if metric=='orientation_rad' else abs(v-expected))
                tol=0 if metric=='area_px2' else (1e-10 if metric=='orientation_rad' else 1e-10+1e-10*abs(expected or 0))
                passed=v is None and expected is None if missing else error<=tol
                undefined+=int(missing)
                rows.append(dict(case=name,instance_id=r['instance_id'],metric=metric,actual=v,reference=expected,
                    absolute_error=error,relative_error=error/abs(expected) if error is not None and abs(expected)>1e-12 else None,
                    tolerance=tol,pass_check=bool(passed),undefined_match=missing and passed))
    distributions_checked=[]
    for metric in METRICS:
        av=sorted(x[metric] for x in actuals if x[metric] is not None);rv=sorted(x[metric] for x in references if x[metric] is not None)
        max_error=max((abs(a-b) for a,b in zip(av,rv)),default=0)
        agreement=len(av)==len(rv) and all(abs(a-b)<=1e-10+1e-10*abs(b) for a,b in zip(av,rv))
        record={'metric':metric,'finite_count':len(rv),'sorted_max_absolute_error':max_error,'pass_check':agreement}
        if metric!='orientation_rad':
            qa=np.quantile(av,[.25,.5,.75]).tolist();qr=[reference_quantile(rv,q) for q in (.25,.5,.75)]
            record.update(actual_q25_median_q75=qa,reference_q25_median_q75=qr)
            record['pass_check'] &= all(abs(a-b)<=1e-10+1e-10*abs(b) for a,b in zip(qa,qr))
        else:record['note']='Sorted axial values matched for these fixtures; no linear orientation quartiles interpreted.'
        distributions_checked.append(record)
    summary=[]
    for metric in METRICS:
        group=[r for r in rows if r['metric']==metric];finite=[r for r in group if r['absolute_error'] is not None];relative=[r['relative_error'] for r in finite if r['relative_error'] is not None]
        summary.append(dict(metric=metric,comparisons=len(group),passed=sum(r['pass_check'] for r in group),
            median_absolute_error=statistics.median(r['absolute_error'] for r in finite),max_absolute_error=max(r['absolute_error'] for r in finite),
            median_relative_error=statistics.median(relative) if relative else None,max_relative_error=max(relative) if relative else None))
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName('morphology.test_morphology'),unittest.defaultTestLoader.loadTestsFromName('morphology.test_ui')])
    with (out/'test_log.txt').open('w',encoding='utf-8') as log:
        test_result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result=dict(version=VERSION,python=platform.python_version(),platform=platform.platform(),
        synthetic_instances=len(actuals),metric_comparisons=len(rows),metric_passed=sum(r['pass_check'] for r in rows),undefined_status_matches=undefined,
        finite_comparisons=len(rows)-undefined,distribution_checks=distributions_checked,error_summary=summary,
        tests_run=test_result.testsRun,tests_passed=test_result.testsRun-len(test_result.failures)-len(test_result.errors),
        external_mask_test='NOT YET RUN',independent_analyst_review=False,
        sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['morphology/engine.py','morphology/reference.py','protocol/RBC_MORPHOLOGY_METRIC_DEFINITIONS.md']})
    (out/'per_instance_errors.csv').write_bytes(csv_bytes(rows));(out/'metric_error_summary.csv').write_bytes(csv_bytes(summary))
    (out/'VALIDATION_RESULTS.json').write_bytes(json_bytes(result))
    a=fixtures()['different_sizes_orientations'];config=Config(identity_confirmed=True,um_per_pixel=.5,calibration_source='synthetic test scale, not microscope calibration',image_id='synthetic-phenotype',sample_id='synthetic-only')
    demo=analyze(a,config,{'type':'FIRST-PARTY SYNTHETIC'})
    Image.fromarray(a).save(out/'safe_labelled_demo.png')
    images={'instance_overlay.png':png_bytes(overlay(a,config,demo)),'distributions.png':png_bytes(distributions(demo))}
    for name,data in images.items():(out/name).write_bytes(data)
    (out/'demo_bundle.zip').write_bytes(export_bundle(demo,images))
    (out/'demo_instances.csv').write_bytes(csv_bytes(demo['instances']))
    passed=result['metric_passed']==result['metric_comparisons'] and all(r['pass_check'] for r in distributions_checked) and test_result.wasSuccessful()
    print(json.dumps({k:v for k,v in result.items() if k not in ('distribution_checks','error_summary','sha256')},indent=2))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
