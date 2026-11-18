"""Isolated supervised native research app. No external inference or image cache."""
import datetime,uuid,platform
import numpy as np
import streamlit as st
from runtime import decode_image,infer,load_model,panel,png_bytes,sha_bytes,verify_science
from exports import csv_bytes,report_bytes,zip_bytes,rows

st.set_page_config(page_title='MicroStructAI · Native RBC',page_icon='🔬',layout='wide')
st.title('MicroStructAI Native RBC Cloud')
st.caption('v1.1 · Native RBC segmentation · Human-supervised Research Use Only · CPU inference')
st.warning('Experimental native model: MIXED evidence. Every mask needs human review. Tested only on centred native 96 × 96 grayscale DPC single-cell crops. Not a full-field detector or a validated brightfield model.')
with st.expander('Evidence, scope and privacy'):
    st.write('Internal test: 454 cells from 5 held-out participants. Blinded audit: 43 GOOD, 6 MINOR_ERROR, 1 MAJOR_ERROR, 0 uncertain. Published algorithmic references were used for training; no independent external native-model validation is claimed.')
    st.write('Images are processed in this app process, without OpenAI, Gemini, or another inference API. On hosted Streamlit, uploads are transmitted to the hosting server. Use only public or authorised non-identifiable research data. Uploaded pixels, masks, reports and reviews are not written to shared application disk or a cross-session cache. Download before ending the session.')
    st.write('Morphology is measured in native pixels. This is not diagnosis, disease classification, aggregation, mechanical deformability or membrane-stiffness measurement. Repeated crops are not independent cells.')
    st.json(verify_science())
    st.caption(f'Host: {platform.system()} · Python {platform.python_version()} · CPU only. Runtime measurements below are not biological validation.')
    try:
        import resource
        if platform.system()=='Linux':st.caption(f'Process peak resident memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MiB. Shared process peak, not a per-session quota or guarantee.')
    except ImportError:
        pass

if 'records' not in st.session_state:st.session_state.records=[]
if 'epoch' not in st.session_state:st.session_state.epoch=0
if st.sidebar.button('Clear this session'):
    epoch=st.session_state.epoch+1
    st.session_state.clear();st.session_state.epoch=epoch;st.rerun()
st.sidebar.caption('Maximum 50 crops per batch; 1 MiB each, 16 MiB total. No silent resizing or cropping.')

@st.cache_resource
def shared_model():
    # Model weights only. Never cache uploaded images, predictions or exports.
    return load_model()

with st.expander('Synthetic software demonstration — not biological validation'):
    from demo import synthetic_image
    st.caption('A first-party artificial ring tests app behaviour only. It is not a real RBC or segmentation-accuracy evidence.')
    demo_name=st.text_input('Demonstration label',value='synthetic_demo')
    st.download_button('Download synthetic upload fixture',png_bytes(synthetic_image()),'synthetic_demo.png','image/png')
    if st.button('Load synthetic demonstration',disabled=bool(st.session_state.records)):
        image=synthetic_image();st.session_state.records=[{'id':str(uuid.uuid4()),'name':demo_name+' (SYNTHETIC)','source_sha256':sha_bytes(png_bytes(image)),'image':image,'result':infer(image,shared_model()),'revisions':[]}]

st.subheader('1 · Upload original cell crops')
uploads=st.file_uploader('Native 96 × 96 DPC crops',type=['png','jpg','jpeg','tif','tiff','bmp'],accept_multiple_files=True,key=f'uploads_{st.session_state.epoch}')
confirmed=st.checkbox('These are centred native 96 × 96 DPC single-cell crops within the tested domain, and I am authorised to upload them.')
st.caption('Colour images with different RGB channels, stacks, other sizes and higher bit depths are rejected. Channel choice, resizing and segmentation settings are not silently inferred.')
if st.session_state.records:st.info('Current batch retained. Export it, then use Clear this session before loading a different batch.')
if st.button('Generate masks',disabled=not (uploads and confirmed) or bool(st.session_state.records),type='primary'):
    try:
        if len(uploads)>50:raise ValueError('Select at most 50 crops.')
        if sum(u.size for u in uploads)>16*1024*1024:raise ValueError('Total upload must not exceed 16 MiB.')
        payloads=[(u.name,u.getvalue()) for u in uploads]
        decoded=[(name,payload,decode_image(payload)) for name,payload in payloads]
        records=[]
        bar=st.progress(0,text='Running local CPU model and frozen refinement')
        model=shared_model()
        for i,(name,payload,image) in enumerate(decoded):
            records.append({'id':str(uuid.uuid4()),'name':name,'source_sha256':sha_bytes(payload),'image':image,'result':infer(image,model),'revisions':[]})
            bar.progress((i+1)/len(decoded))
        st.session_state.records=records
        st.success(f'{len(records)} masks generated. None automatically accepted.')
    except Exception as exc:st.error(str(exc))

records=st.session_state.records
if records:
    st.subheader('2 · Inspect and review')
    selected=st.selectbox('Observation',range(len(records)),format_func=lambda i:f'{i+1}. {records[i]["name"]}')
    r=records[selected];result=r['result']
    original_only=st.checkbox('Original only — hide proposal',key='original_only')
    if original_only:st.image(r['image'],caption='Original native pixels',width=384)
    else:st.image(panel(r['image'],result['mask']),caption='Original · filled mask · overlay · contour (native pixels)',width=1152)
    st.caption(f'Routing: {result["details"]["status"]}. QC flags: {", ".join(result["details"]["qc_flags"]) or "none"}. No AI reviewer was run. QC pass is not acceptance.')
    st.caption(f'Model + refinement + QC elapsed: {result["seconds"]:.3f} s for this observation; excludes upload, review and export. Batch total: {sum(x["result"]["seconds"] for x in records):.3f} s. Shared-host latency varies.')
    with st.form('review_'+r['id']):
        reviewer=st.text_input('Reviewer initials / identity',value=r.get('review',{}).get('reviewer',''))
        quality=st.selectbox('Whole-cell mask quality',['Choose explicitly','GOOD','MINOR_ERROR','MAJOR_ERROR','REVIEWER_UNCERTAIN'])
        st.caption('GOOD: follows the whole visible footprint. MINOR_ERROR: limited deviations. MAJOR_ERROR: missing region, excess background, wrong cell or meaningful shape error. UNCERTAIN: cannot judge reliably. Reject rather than infer an unseen boundary.')
        notes=st.text_area('Observations (optional)')
        inspected=st.checkbox('I personally inspected the original, mask and contour.')
        save=st.form_submit_button('Save review')
        if save:
            if not reviewer.strip() or quality=='Choose explicitly' or not inspected:st.error('Enter reviewer, choose a quality grade and confirm inspection.')
            elif not result['details']['mask_available'] and quality in ('GOOD','MINOR_ERROR'):st.error('A no-mask case cannot be accepted.')
            else:
                review={'quality':quality,'reviewer':reviewer.strip(),'notes':notes,'personally_inspected':True,'saved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision_id':str(uuid.uuid4())}
                r['revisions'].append(review);r['review']=review;st.success('Saved in this session. Export a backup before closing.')
    st.subheader('3 · Reviewed morphology and exports')
    rr=rows(records);accepted=[x for x in rr if x['measurement_status']=='HUMAN_REVIEWED_RESEARCH_MEASUREMENT']
    cols=st.columns(4);cols[0].metric('Uploaded observations',len(records));cols[1].metric('Masks available',sum(bool(x['result']['details']['mask_available']) for x in records));cols[2].metric('Human-approved usable',len(accepted));cols[3].metric('Unreviewed / rejected / uncertain',len(records)-len(accepted))
    st.caption(f'Human-approved coverage: {len(accepted)} / {len(records)} uploaded observations ({100*len(accepted)/len(records):.1f}%). Mask availability is not usable yield; repeated observations are not independent cells.')
    st.dataframe(rr,hide_index=True)
    if accepted:
        st.caption('Descriptive plots of reviewed observations; repeated observations are not independent cells.')
        a,b=st.columns(2)
        a.bar_chart({'area_px2':[x['area_px2'] for x in accepted]},x_label='Reviewed observation',y_label='Area (px²)')
        b.bar_chart({'circularity':[x['circularity'] for x in accepted]},x_label='Reviewed observation',y_label='Circularity')
    else:st.info('Morphology values are withheld until a human marks an available mask GOOD or MINOR_ERROR.')
    st.download_button('Download results CSV',csv_bytes(records),'native_rbc_results.csv','text/csv')
    st.download_button('Download research report',report_bytes(records),'native_rbc_report.html','text/html')
    st.download_button('Download masks, figures and review backup',zip_bytes(records),'native_rbc_session.zip','application/zip')
