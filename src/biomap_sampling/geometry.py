# Scopo: finestre native deterministiche per cella, senza target, mosaici o terreno ripetuto tra tile.
# Fasi: maschere indice -> finestre massima copertura -> dispersione -> esclusione footprint -> manifest.
# Input: patch geometriche (tile/grid/row/col/mask), ID/centro cella, seed; output: finestre e diagnostiche.
# Parametri: S64/M128/S144/M64/FULL; esempio select(patches,'M128',cell_id,lon,lat).
# Risorse: CPU NumPy/PyProj; niente embedding o rete. Ripresa: funzione pura, spareggi hash stabili.
# Intersezioni: solo scelta del supporto; nessuna trasformazione dei valori raster. Tolleranza area 1e-6 m2.
from functools import lru_cache
import hashlib
import numpy as np
from pyproj import Transformer,Geod
from scipy.ndimage import binary_erosion

MODES={'S64':(1,8),'M128':(2,8),'S144':(1,12),'M64':(4,4)}
ALIASES={'single_8x8':'S64','multi_2x8x8':'M128','single_12x12':'S144','multi_4x4x4':'M64',
         'full_cell_subset':'FULL-SUBSET','full':'FULL'}
SEED=20260925
GEOD=Geod(ellps='WGS84')


@lru_cache(maxsize=64)
def transformer(first,second):return Transformer.from_crs(first,second,always_xy=True)


def project(first,second,x,y):
    x=np.asarray(x);y=np.asarray(y)
    if x.size==1:
        a,b=transformer(first,second).transform(float(x.reshape(-1)[0]),float(y.reshape(-1)[0]));return np.array([a]),np.array([b])
    return transformer(first,second).transform(x,y)


def distances(lon,lat,x,y):
    x=np.asarray(x);y=np.asarray(y)
    if x.size==1:return np.array([GEOD.inv(lon,lat,float(x.reshape(-1)[0]),float(y.reshape(-1)[0]))[2]])
    return np.asarray(GEOD.inv(np.full(x.shape,lon),np.full(y.shape,lat),x,y)[2])


def unpack(record):
    return np.unpackbits(np.frombuffer(bytes.fromhex(record['mask']),np.uint8),count=record['height']*record['width']).reshape(record['height'],record['width']).astype(bool)


def encode_mask(mask):return np.packbits(mask.reshape(-1)).tobytes().hex()


def rf_positions(mask):return int(binary_erosion(mask,structure=np.ones((5,5),bool),border_value=0).sum())


def xy(grid,rows,cols,center=True):
    a,_,c,_,e,f=grid['transform'][:6];offset=.5 if center else 0
    return c+(np.asarray(cols)+offset)*a,f+(np.asarray(rows)+offset)*e


def clip(poly,axis,bound,greater):
    out=[]
    for p,q in zip(poly,np.roll(poly,-1,axis=0)):
        pin=p[axis]>=bound if greater else p[axis]<=bound
        qin=q[axis]>=bound if greater else q[axis]<=bound
        if pin:out.append(p)
        if pin!=qin:
            alpha=(bound-p[axis])/(q[axis]-p[axis]);out.append(p+alpha*(q-p))
    return np.asarray(out,dtype=np.float64).reshape(-1,2)


def intersection_area(poly,left,bottom,right,top):
    for axis,bound,greater in [(0,left,True),(0,right,False),(1,bottom,True),(1,top,False)]:
        if not len(poly):return 0.
        poly=clip(poly,axis,bound,greater)
    if len(poly)<3:return 0.
    poly=poly-poly[0]
    return abs(float(np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))))/2


def footprints(patch):
    mask=unpack(patch);grid=patch['grid']
    rectangles=[(0,0,*mask.shape)] if mask.all() else [(int(r),int(c),1,1) for r,c in zip(*np.where(mask))]
    result=[]
    for r,c,h,w in rectangles:
        rows=np.array([r,r,r+h,r+h])+patch['row'];cols=np.array([c,c+w,c+w,c])+patch['col']
        x,y=xy(grid,rows,cols,False);result.append((grid['crs'],np.column_stack([x,y])))
    return result


def available(patch,chosen):
    mask=patch['mask'].copy();rr,cc=np.where(mask);grid=patch['grid'];x,y=xy(grid,rr+patch['row'],cc+patch['col'])
    dx,dy=grid['transform'][0]/2,-grid['transform'][4]/2
    for crs,poly in chosen:
        if crs!=grid['crs']:
            px,py=transformer(crs,grid['crs']).transform(poly[:,0],poly[:,1]);poly=np.column_stack([px,py])
        low=poly.min(0);high=poly.max(0)
        possible=np.flatnonzero((x+dx>low[0]+1e-7)&(x-dx<high[0]-1e-7)&(y+dy>low[1]+1e-7)&(y-dy<high[1]-1e-7))
        if crs==grid['crs']:
            mask[rr[possible],cc[possible]]=False
        else:
            for k in possible:
                if intersection_area(poly,x[k]-dx,y[k]-dy,x[k]+dx,y[k]+dy)>1e-6:mask[rr[k],cc[k]]=False
    return mask


def window(mask,row,col,side):
    result=np.zeros((side,side),bool);r0=max(row,0);c0=max(col,0);r1=min(row+side,mask.shape[0]);c1=min(col+side,mask.shape[1])
    if r1>r0 and c1>c0:result[r0-row:r1-row,c0-col:c1-col]=mask[r0:r1,c0:c1]
    return result


def stable(seed,cell,tile,row,col):
    return int.from_bytes(hashlib.blake2b(f'{seed}/{cell}/{tile}/{row}/{col}'.encode(),digest_size=8).digest(),'little')


def record(patch,mask,row,col,source_patch):
    return {'tile':patch['tile'],'grid':patch['grid'],'row':int(row),'col':int(col),'height':mask.shape[0],'width':mask.shape[1],
        'mask':encode_mask(mask),'weight':int(mask.sum()),'source_patch':source_patch,
        'source_index_patch_id':patch.get('source_index_patch_id')}


def select(patches,mode,cell_id,lon,lat,seed=SEED,min_valid_pixels=1):
    mode=ALIASES.get(mode,mode)
    if not patches or not any(p['mask'].any() for p in patches):raise ValueError('Cella senza supporto; nessuna esclusione/imputazione')
    if mode in {'FULL','FULL-SUBSET'}:
        chosen=[record(p,p['mask'],p['row'],p['col'],i) for i,p in enumerate(patches)]
    else:
        number,side=MODES[mode];chosen=[];occupied=[];centers=[]
        for _ in range(number):
            best=None
            for i,p in enumerate(patches):
                mask=available(p,occupied);rr,cc=np.where(mask)
                if not len(rr):continue
                starts_r=rr-side//2;starts_c=cc-side//2
                integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
                r0=np.maximum(starts_r,0);c0=np.maximum(starts_c,0)
                r1=np.minimum(starts_r+side,mask.shape[0]);c1=np.minimum(starts_c+side,mask.shape[1])
                counts=integral[r1,c1]-integral[r0,c1]-integral[r1,c0]+integral[r0,c0]
                take=np.flatnonzero(counts==counts.max())
                x,y=xy(p['grid'],p['row']+starts_r[take]+(side-1)/2,p['col']+starts_c[take]+(side-1)/2)
                ll,la=project(p['grid']['crs'],4326,x,y)
                if centers:
                    distance=np.min([distances(a,b,ll,la) for a,b in centers],axis=0)
                else:distance=-distances(lon,lat,ll,la)
                # Hash only the distance ties: same ordering, avoids hashing hundreds of inferior windows/cell.
                for local in np.flatnonzero(distance==distance.max()):
                    k=take[local]
                    r=int(starts_r[k]);c=int(starts_c[k]);key=(-int(counts[k]),-float(distance[local]),stable(seed,cell_id,p['tile'],r+p['row'],c+p['col']))
                    if best is None or key<best[0]:best=(key,i,r,c,mask,float(ll[local]),float(la[local]))
            if best is None:break
            _,i,r,c,mask,ll,la=best;p=patches[i];selected=record(p,window(mask,r,c,side),p['row']+r,p['col']+c,i)
            chosen.append(selected);occupied.extend(footprints(selected));centers.append((ll,la))
    n=sum(p['weight'] for p in chosen)
    if n<1:raise ValueError('Campionamento vuoto; nessuna cella rimossa')
    rf=sum(rf_positions(unpack(p)) for p in chosen);total=sum(int(p['mask'].sum()) for p in patches)
    centers=[]
    for index,p in enumerate(chosen):
        p['patch_id']=f'{cell_id}/{mode}/{index}'
        rr,cc=np.where(unpack(p));x,y=xy(p['grid'],rr+p['row'],cc+p['col']);ll,la=project(p['grid']['crs'],4326,x,y)
        p['wgs84']=np.column_stack([ll,la]).tolist();centers.append((float(np.mean(ll)),float(np.mean(la))))
    pair_distances=[float(GEOD.inv(*a,*b)[2]) for j,a in enumerate(centers) for b in centers[j+1:]]
    nominal=MODES[mode][0]*MODES[mode][1]**2 if mode in MODES else total
    return {'patches':chosen,'diagnostics':{'selected_pixels':n,'historical_contributions':total,'coverage_contributions':n/total,
        'patches':len(chosen),'partial_mask':any(not unpack(p).all() for p in chosen),'insufficient':n<nominal or n<min_valid_pixels,
        'rf_complete_positions':rf,'rf_complete_fraction':rf/n,'patch_distances_m':pair_distances,
        'centroid_distances_m':[float(GEOD.inv(lon,lat,*x)[2]) for x in centers],
        'multi_tile':len({p['tile'] for p in patches})>1,'multi_utm':len({p['grid']['crs'] for p in patches})>1}}
