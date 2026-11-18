"""Independent mask arithmetic. No engine, NumPy, SciPy or skimage imports.

New code-path implementation by same developer; not independent analyst review.
"""
import math

def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

def convex_polygon(points):
    ordered=sorted(set(points));lower=[];upper=[]
    for p in ordered:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def reference_measure(mask):
    height=len(mask);width=len(mask[0]);points=[(x,y) for y in range(height) for x in range(width) if mask[y][x]]
    n=len(points)
    if not n:raise ValueError('Empty reference object.')
    meanx=math.fsum(x for x,y in points)/n;meany=math.fsum(y for x,y in points)/n
    xx=math.fsum((x-meanx)**2 for x,y in points)/n
    yy=math.fsum((y-meany)**2 for x,y in points)/n
    xy=math.fsum((x-meanx)*(y-meany) for x,y in points)/n
    delta=math.hypot(xx-yy,2*xy);large=max(0.,(xx+yy+delta)/2);small=max(0.,(xx+yy-delta)/2)
    l=4*math.sqrt(large);w=4*math.sqrt(small)
    theta=(.5*math.atan2(2*xy,xx-yy)+math.pi/2)%math.pi-math.pi/2 if delta>1e-12*max(1.,large) else None
    grid=[[False]*(width+2)]+[[False]+[bool(v) for v in row]+[False] for row in mask]+[[False]*(width+2)]
    segments=[]
    for y in range(height+1):
        for x in range(width+1):
            tl,tr,br,bl=grid[y][x],grid[y][x+1],grid[y+1][x+1],grid[y+1][x]
            count=tl+tr+br+bl
            if count in (1,3):segments.append(math.sqrt(.5))
            elif count==2:segments.append(math.sqrt(2.) if tl==br and tr==bl else 1.)
    perimeter=math.fsum(segments)
    hull=convex_polygon([(x+dx,y+dy) for x,y in points for dx,dy in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))])
    hull_area=abs(math.fsum(a[0]*b[1]-a[1]*b[0] for a,b in zip(hull,hull[1:]+hull[:1])))/2
    box=(max(p[0] for p in points)-min(p[0] for p in points)+1)*(max(p[1] for p in points)-min(p[1] for p in points)+1)
    return dict(area_px2=float(n),perimeter_px=perimeter,equivalent_diameter_px=math.sqrt(4*n/math.pi),
        major_axis_px=l,minor_axis_px=w,aspect_ratio=l/w if w else None,circularity=4*math.pi*n/perimeter**2,
        eccentricity=math.sqrt(max(0.,1-small/large)) if large else None,solidity=n/hull_area,
        orientation_rad=theta,convex_area_px2=hull_area,extent=n/box)

def reference_quantile(values,q):
    v=sorted(values);position=q*(len(v)-1);lo=math.floor(position);hi=math.ceil(position)
    return v[lo]+(position-lo)*(v[hi]-v[lo])
