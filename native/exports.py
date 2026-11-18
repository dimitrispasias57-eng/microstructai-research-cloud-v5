"""Session-local export formatting, not quantitative scientific calculations."""
import base64,csv,html,io,json,zipfile
from runtime import ROOT,panel,png_bytes,sha_bytes

def rows(records):
    out=[];manifest=json.loads((ROOT/'SCIENTIFIC_MANIFEST.json').read_text())
    for r in records:
        review=r.get('review',{})
        row={'image_id':r['id'],'filename':r['name'],'source_sha256':r['source_sha256'],'mask_sha256':sha_bytes(png_bytes(r['result']['mask'].astype('uint8')*255)),
             'review':review.get('quality','NOT_REVIEWED'),'reviewer':review.get('reviewer',''),'review_saved_utc':review.get('saved_utc',''),
             'review_notes':review.get('notes',''),'routing':r['result']['details']['status'],'qc_flags':';'.join(r['result']['details']['qc_flags']),
             'model_version':manifest['name'],'portable_model_sha256':manifest['files']['model/native_rbc_v1.ts'],
             'mask_available':r['result']['details']['mask_available'],'measurement_status':'UNREVIEWED_EXPLORATORY_NOT_ACCEPTED'}
        if review.get('quality') in ('GOOD','MINOR_ERROR') and r['result']['details']['mask_available']:
            row['measurement_status']='HUMAN_REVIEWED_RESEARCH_MEASUREMENT'
            row.update(r['result']['details']['morphology'] or {})
        out.append(row)
    return out

def csv_bytes(records):
    rr=rows(records);keys=list(dict.fromkeys(k for r in rr for k in r))
    b=io.StringIO(newline='');writer=csv.DictWriter(b,fieldnames=keys);writer.writeheader()
    # Spreadsheet formula injection protection only on user-controlled textual fields.
    for row in rr:
        writer.writerow({k:("'"+v if isinstance(v,str) and v[:1] in ('=','+','-','@','\t','\r') else v) for k,v in row.items()})
    return b.getvalue().encode('utf8')

def report_bytes(records):
    rr=rows(records);accepted=sum(r['measurement_status']=='HUMAN_REVIEWED_RESEARCH_MEASUREMENT' for r in rr)
    sections=[]
    for rec,row in zip(records,rr):
        image=base64.b64encode(png_bytes(panel(rec['image'],rec['result']['mask']))).decode()
        cells=''.join('<tr><th>'+html.escape(str(k))+'</th><td>'+html.escape(str(v))+'</td></tr>' for k,v in row.items())
        sections.append('<section><h2>'+html.escape(rec['name'])+'</h2><img alt="Original, mask, overlay, contour" src="data:image/png;base64,'+image+'"><table>'+cells+'</table></section>')
    return ('<!doctype html><html><meta charset="utf-8"><title>Native RBC research report</title><style>body{font:16px Arial;max-width:1000px;margin:40px auto;color:#172a3a}img{width:768px;max-width:100%;image-rendering:pixelated}td,th{text-align:left;padding:5px;overflow-wrap:anywhere}section{border-top:1px solid #ccc;margin-top:24px}</style><h1>MicroStructAI Native RBC Cloud v1.1</h1><p>Human-supervised Research Use Only. NATIVE_RBC_MODEL_MIXED; INTERNAL_VALIDATION_ONLY. No diagnostic, disease, aggregation or deformability claims. Published training references were algorithmic, not manual ground truth.</p><p>Native centred 96 × 96 DPC crops. No automatic acceptance. Results are in pixels; rejected, uncertain and unreviewed masks do not receive accepted morphology.</p><p>'+str(accepted)+' / '+str(len(rr))+' reviewed usable observations. This denominator is uploaded observations, not independent cells or donors.</p>'+''.join(sections)+'</html>').encode('utf8')

def zip_bytes(records):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('results.csv',csv_bytes(records));z.writestr('report.html',report_bytes(records))
        z.writestr('scientific_provenance.json',(ROOT/'SCIENTIFIC_MANIFEST.json').read_bytes())
        z.writestr('reviews.json',json.dumps([{'id':r['id'],'source_sha256':r['source_sha256'],'revisions':r.get('revisions',[])} for r in records],indent=2))
        for i,r in enumerate(records):
            z.writestr(f'case_{i+1:03d}_mask.png',png_bytes(r['result']['mask'].astype('uint8')*255))
            z.writestr(f'case_{i+1:03d}_panels.png',png_bytes(panel(r['image'],r['result']['mask'])))
    return b.getvalue()
