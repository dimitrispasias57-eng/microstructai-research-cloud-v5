import io
import json
import math
import unittest
import zipfile
from dataclasses import replace
import numpy as np
from PIL import Image
from morphology.engine import Config,analyze,measure,METRICS,LENGTHS,AREAS,stack_to_labels,combine_samples
from morphology.reference import reference_measure,reference_quantile
from morphology.fixtures import fixtures,reference_grid
from morphology.io import load_mask,load_image,csv_bytes,json_bytes,html_report,export_bundle
from morphology.plots import overlay,distributions,png_bytes

C=Config(identity_confirmed=True,image_id='safe-image',sample_id='safe-sample',donor_id='synthetic-donor')

def encoded(a,fmt='PNG'):
    stream=io.BytesIO();Image.fromarray(a).save(stream,format=fmt);return stream.getvalue()

class MorphologyTests(unittest.TestCase):
    def setUp(self):self.f=fixtures();self.a=self.f['ellipse']
    def result(self,a=None,**kw):return analyze(self.a if a is None else a,replace(C,**kw))
    def test_identity_required(self):
        with self.assertRaises(ValueError):analyze(self.a)
    def test_binary_is_not_cell_identity(self):
        r=self.result(self.f['touching_binary'],mode='components',identity_confirmed=False)
        self.assertEqual(len(r['instances']),1);self.assertIn('unestablished',r['instances'][0]['object_semantics'])
    def test_touching_ids_separate(self):self.assertEqual(self.result(self.f['touching_labelled'])['provenance']['included_count'],2)
    def test_disconnected_invalid(self):
        r=self.result(self.f['disconnected_invalid']);self.assertEqual(r['provenance']['included_count'],0)
        self.assertIsNone(r['instances'][0]['area_px2'])
    def test_roi_does_not_hide_disconnected_original(self):
        r=self.result(self.f['disconnected_invalid'],roi=(0,0,8,8));self.assertEqual(r['provenance']['included_count'],0)
    def test_outside_roi_retained(self):
        r=self.result(roi=(0,0,5,5));self.assertEqual(len(r['instances']),1)
        self.assertIn('OUTSIDE_ROI',r['instances'][0]['exclusion_reasons'])
    def test_roi_clip_flag(self):
        r=self.result(roi=(50,20,90,65));self.assertTrue(r['instances'][0]['roi_clipped'])
        self.assertLess(r['instances'][0]['area_px2'],r['instances'][0]['original_area_px2'])
    def test_border_default_included(self):
        r=self.result(self.f['border_truncated']);self.assertTrue(r['instances'][0]['included'])
    def test_border_explicit_exclusion_retains_metrics(self):
        r=self.result(self.f['border_truncated'],exclude_border=True);row=r['instances'][0]
        self.assertFalse(row['included']);self.assertGreater(row['area_px2'],0)
    def test_minimum_explicit(self):
        r=self.result(minimum_area=10000);self.assertEqual(r['provenance']['included_count'],0)
        self.assertGreater(r['instances'][0]['area_px2'],0)
    def test_single_pixel_undefined(self):
        row=self.result(self.f['single_pixel'])['instances'][0]
        for k in ['aspect_ratio','orientation_rad','eccentricity']:self.assertIsNone(row[k])
        self.assertIn('LOW_PIXEL_SUPPORT',row['qc'])
    def test_line_minor_zero(self):self.assertEqual(self.result(self.f['line'])['instances'][0]['minor_axis_px'],0)
    def test_isotropic_orientation_null(self):self.assertIsNone(self.result(self.f['circle'])['instances'][0]['orientation_rad'])
    def test_holes_not_filled(self):
        row=self.result(self.f['hole'])['instances'][0];self.assertIn('HOLES_PRESENT_NOT_FILLED',row['qc'])
        self.assertEqual(row['area_px2'],int((self.f['hole']>0).sum()))
    def test_circularity_not_clipped(self):
        row=self.result(self.f['single_pixel'])['instances'][0];self.assertGreater(row['circularity'],1)
    def test_stack_adjacent_cells(self):
        a=self.f['touching_labelled'];b=stack_to_labels(np.stack([a==1,a==2]));np.testing.assert_array_equal(a,b)
    def test_stack_overlap_rejected(self):
        with self.assertRaises(ValueError):stack_to_labels(np.stack([self.a,self.a]))
    def test_stack_empty_plane_rejected(self):
        with self.assertRaises(ValueError):stack_to_labels(np.stack([self.a,np.zeros_like(self.a)]))
    def test_stack_multilabel_plane_rejected(self):
        with self.assertRaises(ValueError):stack_to_labels(self.f['touching_labelled'][None,:,:])
    def test_mask_upload_roundtrip(self):
        a,p=load_mask(encoded(self.a),'safe.png');np.testing.assert_array_equal(a,self.a)
        self.assertEqual(len(p['file_sha256']),64)
    def test_tiff_stack_upload(self):
        a=self.f['touching_labelled'];out=io.BytesIO();Image.fromarray((a==1).astype('uint8')).save(out,format='TIFF',save_all=True,append_images=[Image.fromarray((a==2).astype('uint8'))])
        b,p=load_mask(out.getvalue(),'cells.tiff',stack=True);np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):load_mask(out.getvalue(),'cells.tiff')
    def test_rgb_mask_rejected(self):
        with self.assertRaises(ValueError):load_mask(encoded(np.zeros((5,5,3),'uint8')),'rgb.png')
    def test_corrupted_rejected(self):
        with self.assertRaises(ValueError):load_mask(b'not an image','bad.png')
    def test_bytes_limit(self):
        with self.assertRaises(ValueError):load_mask(b'0'*(16*1024*1024+1),'big.png')
    def test_decoded_limit(self):
        with self.assertRaises(ValueError):load_mask(encoded(np.ones((1001,1000),'uint8')),'big.png')
    def test_paired_image_exact_dimensions(self):
        original=load_image(encoded(np.zeros((*self.a.shape,3),'uint8')),self.a.shape);self.assertEqual(original.shape[:2],self.a.shape)
    def test_paired_wrong_dimensions(self):
        with self.assertRaises(ValueError):load_image(encoded(np.zeros((3,3),'uint8')),self.a.shape)
    def test_translation(self):
        a=measure(self.a>0);b=measure(np.pad(self.a,((7,3),(11,2)))>0)
        for k in METRICS:self.assertAlmostEqual(a[k],b[k],places=10)
    def test_rotation(self):
        a=measure(self.a>0);b=measure(np.rot90(self.a)>0)
        for k in METRICS:
            if k=='orientation_rad':self.assertAlmostEqual(abs((a[k]-b[k]+math.pi/2)%math.pi-math.pi/2),math.pi/2)
            else:self.assertAlmostEqual(a[k],b[k],places=10)
    def test_label_permutation(self):
        b=self.a.astype('int64')*9000000000
        a=self.result()['instances'][0];r=self.result(b)['instances'][0]
        for k in METRICS:self.assertEqual(a[k],r[k])
    def test_repeatability(self):self.assertEqual(json_bytes(self.result()),json_bytes(self.result()))
    def test_summary_quantiles_independent(self):
        r=self.result(self.f['different_sizes_orientations']);rows=r['instances']
        for s in r['summaries']:
            if s['metric']=='orientation_rad':continue
            v=[row[s['metric']] for row in rows if row.get(s['metric']) is not None]
            self.assertAlmostEqual(s['median'],reference_quantile(v,.5));self.assertAlmostEqual(s['iqr'],reference_quantile(v,.75)-reference_quantile(v,.25))
    def test_orientation_not_linear_summary(self):
        s=next(x for x in self.result()['summaries'] if x['metric']=='orientation_rad')
        self.assertNotIn('median',s);self.assertIn('axial_resultant',s)
    def test_csv_has_hierarchy_qc_units(self):
        out=csv_bytes(self.result()['instances']).decode('utf-8-sig')
        for word in ['image_id','sample_id','donor_id','instance_id','area_px2','included','exclusion_reasons']:self.assertIn(word,out)
    def test_csv_injection_escaped(self):self.assertIn("'=SUM",csv_bytes([{'sample_id':'=SUM(1,2)'}]).decode())
    def test_html_metadata_escaped(self):
        report=html_report(self.result(sample_id='<script>alert(1)</script>')).decode();self.assertNotIn('<script>',report)
    def test_json_null_not_nan(self):
        r=self.result(self.f['single_pixel']);blob=json_bytes(r)
        self.assertNotIn(b'NaN',blob);self.assertIn(b'null',blob)
    def test_export_bundle(self):
        r=self.result();images={'overlay.png':png_bytes(overlay(self.a,C,r)),'distributions.png':png_bytes(distributions(r))}
        with zipfile.ZipFile(io.BytesIO(export_bundle(r,images))) as z:
            self.assertEqual(len(z.namelist()),6);self.assertEqual(json.loads(z.read('measurements_and_provenance.json'))['provenance']['included_count'],1)
    def test_sample_pool(self):
        a=self.result();b=self.result(image_id='second')
        p=combine_samples([a,b],nonoverlapping_fields_confirmed=True);self.assertEqual(len(p),1);self.assertEqual(len(p[0]['instances']),2)
    def test_pool_duplicate_rejected(self):
        r=self.result()
        with self.assertRaises(ValueError):combine_samples([r,r],nonoverlapping_fields_confirmed=True)
    def test_pool_confirmation_required(self):
        with self.assertRaises(ValueError):combine_samples([self.result()])
    def test_pool_donor_conflict(self):
        with self.assertRaises(ValueError):combine_samples([self.result(),self.result(image_id='second',donor_id='other')],nonoverlapping_fields_confirmed=True)
    def test_pool_missing_id(self):
        with self.assertRaises(ValueError):combine_samples([self.result(sample_id='')],nonoverlapping_fields_confirmed=True)
    def test_pool_missing_provenance(self):
        r=self.result();del r['provenance']['decoded_sha256']
        with self.assertRaises(ValueError):combine_samples([r],nonoverlapping_fields_confirmed=True)
    def test_no_aggregation_or_deformability_claim(self):
        r=self.result();self.assertIn('NOT CALCULATED',r['phenotype']['aggregation']);self.assertIn('DESIGN ONLY',r['phenotype']['deformability'])
    def test_closed_form_rectangle(self):
        a=np.zeros((40,40),bool);a[5:15,6:26]=True;r=measure(a)
        self.assertEqual(r['area_px2'],200);self.assertAlmostEqual(r['convex_area_px2'],200)
        self.assertAlmostEqual(r['perimeter_px'],2*(20+10)-4+2*math.sqrt(2))
        self.assertAlmostEqual(r['major_axis_px'],4*math.sqrt((20**2-1)/12))
        self.assertAlmostEqual(r['minor_axis_px'],4*math.sqrt((10**2-1)/12))
        self.assertEqual(r['orientation_rad'],0);self.assertAlmostEqual(r['solidity'],1);self.assertEqual(r['extent'],1)

def metric_test(key):
    def test(self):
        cases=list(fixtures().items())+list(reference_grid())
        for name,a in cases:
            if name=='disconnected_invalid':continue
            for i in np.unique(a[a>0]):
                with self.subTest(name=name,instance=int(i)):
                    mask=a==i;actual=measure(mask)[key];reference=reference_measure(mask.tolist())[key]
                    if reference is None:self.assertIsNone(actual);continue
                    err=abs((actual-reference+math.pi/2)%math.pi-math.pi/2) if key=='orientation_rad' else abs(actual-reference)
                    tolerance=0 if key=='area_px2' else (1e-10 if key=='orientation_rad' else 1e-10+1e-10*abs(reference))
                    self.assertLessEqual(err,tolerance)
    return test
for key in METRICS:setattr(MorphologyTests,'test_reference_'+key,metric_test(key))

def calibration_test(scale):
    def test(self):
        before=self.result()['instances'][0];after=self.result(um_per_pixel=scale,calibration_source='synthetic calibration')['instances'][0]
        for k in LENGTHS:self.assertAlmostEqual(after[k[:-3]+'_um'],before[k]*scale)
        for k in AREAS:self.assertAlmostEqual(after[k[:-4]+'_um2'],before[k]*scale**2)
        for k in METRICS:self.assertEqual(before[k],after[k])
    return test
for scale in (.25,.5,2):setattr(MorphologyTests,'test_calibration_'+str(scale).replace('.','_'),calibration_test(scale))

def invalid_test(value,keyword='mask'):
    def test(self):
        with self.assertRaises((ValueError,TypeError)):
            self.result(value) if keyword=='mask' else self.result(**{keyword:value})
    return test
for name,value,key in [('empty',np.zeros((4,4),'uint8'),'mask'),('float',np.ones((4,4)),'mask'),
    ('negative',-np.ones((4,4),'int32'),'mask'),('volume',np.ones((3,4,4),'uint8'),'mask'),
    ('roi_invalid',(8,8,2,2),'roi'),('roi_float',(0,0,4.,5),'roi'),('scale_zero',0,'um_per_pixel'),
    ('scale_negative',-.2,'um_per_pixel'),('scale_nan',float('nan'),'um_per_pixel'),('scale_anisotropic',(.2,.3),'um_per_pixel'),
    ('scale_unsourced',.2,'um_per_pixel'),('min_zero',0,'minimum_area')]:
    setattr(MorphologyTests,'test_invalid_'+name,invalid_test(value,key))

if __name__=='__main__':unittest.main()
