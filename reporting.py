"""Memory-only shared presentation and export helpers; no scientific recalculation."""
import base64, csv, html, io, json, zipfile, threading
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from backend import VERSION, json_safe, native, display_u8
PLOT_LOCK=threading.Lock()

def csv_bytes(frame):
    # Neutralise spreadsheet formulas in user-controlled text, leaving numbers typed.
    safe=frame.copy()
    for c in safe.columns:
        safe[c]=safe[c].map(lambda x:"'"+x if isinstance(x,str) and x[:1] in ('=','+','-','@','\t','\r') else x)
    return safe.to_csv(index=False).encode('utf8')

def panel(image,mask):
    from scipy.ndimage import binary_erosion
    a=display_u8(image)
    rgb=np.repeat(a[...,None],3,axis=2) if a.ndim==2 else a[...,:3].copy()
    if rgb.shape[:2]!=mask.shape:raise ValueError('Panel geometry mismatch; refusing an unregistered overlay.')
    overlay=rgb.copy();overlay[mask]=(rgb[mask]*.55+np.array([0,210,255])*.45).astype('uint8')
    edge=rgb.copy();edge[mask&~binary_erosion(mask)]=[255,175,20]
    return np.concatenate([rgb,np.repeat((mask*255).astype('uint8')[...,None],3,2),overlay,edge],axis=1)

def chart_bytes(frame,columns,x=None):
    with PLOT_LOCK:
        fig=Figure(figsize=(9,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
        for c in columns:ax.plot(frame[x] if x else np.arange(1,len(frame)+1),pd.to_numeric(frame[c],errors='coerce'),marker='.',label=c)
        ax.set_xlabel(x or 'Observation (not independent cells)');ax.set_ylabel('Metric value (column units)')
        ax.grid(alpha=.2)
        if columns:ax.legend(fontsize=8)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);return buf.getvalue()

def bundle(module,frame,metadata,limitations,figures=None,extras=None):
    figures=figures or {};extras=extras or {}
    meta={'software':'MicroStructAI Research Cloud','version':VERSION,'module':module,**metadata,'research_use_only':True}
    pictures=''.join('<h2>'+html.escape(n)+'</h2><img src="data:image/png;base64,'+base64.b64encode(b).decode()+'">' for n,b in figures.items())
    report=('<!doctype html><html><meta charset="utf-8"><title>MicroStructAI research report</title><style>body{font:15px Arial;max-width:1100px;margin:35px auto;color:#183143}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #dde5e8}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>MicroStructAI Research Cloud '+VERSION+'</h1><h2>'+html.escape(module)+'</h2><p>Research Use Only. No diagnostic or clinical interpretation.</p><p>'+html.escape(limitations)+'</p><pre>'+html.escape(json.dumps(json_safe(meta),indent=2))+'</pre>'+frame.to_html(index=False,escape=True)+pictures+'</html>').encode('utf8')
    files={'results.csv':csv_bytes(frame),'report.html':report,'provenance.json':json.dumps(json_safe(meta),indent=2,allow_nan=False).encode(),**figures,**extras}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    return {'csv':files['results.csv'],'html':report,'zip':buf.getvalue(),'metadata':meta}
