"""Display-only figures; never modify analysis masks."""
import io
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from .engine import Config

def overlay(labels, config=Config(), result=None, original=None):
    fig=Figure(figsize=(9,6),layout='constrained');ax=fig.subplots()
    a=labels
    if config.mode=='components':a=ndimage.label(a>0,np.ones((3,3)))[0]
    if original is not None:ax.imshow(original)
    else:ax.imshow(np.zeros(a.shape),cmap='gray',vmin=0,vmax=1)
    ids=np.unique(a[a>0]);colours=np.zeros((*a.shape,4))
    statuses={r['instance_id']:r for r in result['instances']} if result else {}
    palette=np.array([(1,.2,.2,.65) if statuses.get(int(i),{}).get('included') is False else ((rank*.618)%1,.72,.88,.45) for rank,i in enumerate(ids)])
    positive=a>0;colours[positive]=palette[np.searchsorted(ids,a[positive])]
    boundaries=np.zeros(a.shape,bool)
    boundaries[:-1,:]|=a[:-1,:]!=a[1:,:];boundaries[1:,:]|=a[1:,:]!=a[:-1,:]
    boundaries[:,:-1]|=a[:,:-1]!=a[:,1:];boundaries[:,1:]|=a[:,1:]!=a[:,:-1]
    boundaries[[0,-1],:]|=positive[[0,-1],:];boundaries[:,[0,-1]]|=positive[:,[0,-1]]
    colours[boundaries&positive]=[1,1,1,.9]
    for i,row in statuses.items():
        if not row['included']:colours[(a==i)&boundaries]=[1,.2,.2,1]
    for i in ids[:80]:
        row=statuses.get(int(i),{});label_color='red' if row.get('included') is False else ('orange' if row.get('qc') else 'yellow')
        y,x=np.nonzero(a==i);ax.text(x.mean(),y.mean(),str(i),color=label_color,fontsize=8,ha='center')
    ax.imshow(colours)
    x0,y0,x1,y1=config.roi or (0,0,a.shape[1],a.shape[0])
    ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#00ff99',linewidth=1.5))
    ax.set(xlim=(-.5,a.shape[1]-.5),ylim=(a.shape[0]-.5,-.5),xlabel='x (pixels)',ylabel='y (pixels)',
           title='Labels • green ROI • red excluded • orange QC • IDs (first 80)')
    return fig

def distributions(result):
    fig=Figure(figsize=(11,7),layout='constrained');axes=fig.subplots(2,3).flat
    calibrated=result['provenance']['settings']['um_per_pixel'] is not None
    metrics=[('area_um2' if calibrated else 'area_px2','Area (µm²)' if calibrated else 'Area (px²)'),
             ('equivalent_diameter_um' if calibrated else 'equivalent_diameter_px','Equivalent diameter (µm)' if calibrated else 'Equivalent diameter (px)'),
             ('aspect_ratio','Aspect ratio (1)'),('circularity','Circularity (1)'),('eccentricity','Eccentricity (1)'),('orientation_rad','Axial orientation (degrees)')]
    rows=[r for r in result['instances'] if r['included']]
    for ax,(key,label) in zip(axes,metrics):
        values=[r[key] for r in rows if r.get(key) is not None]
        if key=='orientation_rad':values=np.degrees(values);bins=np.linspace(-90,90,19)
        else:bins=max(1,min(20,int(np.sqrt(len(values)))))
        if len(values):ax.hist(values,bins=bins,color='#167d9b',edgecolor='white')
        else:ax.text(.5,.5,'NOT EVALUABLE',ha='center',transform=ax.transAxes)
        ax.set(xlabel=label,ylabel='Included instances',title=f'n={len(values)}; missing={len(rows)-len(values)}')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Descriptive distributions — not independent biological replicates')
    return fig

def png_bytes(fig):
    out=io.BytesIO();fig.savefig(out,format='png',dpi=130);return out.getvalue()
