# Scopo: provare il supporto fisico V1/V2 prima di pubblicare il reference multi-AOI.
# Ricalcola V1 dalle tile, verifica bordi, scale/maschere e coordinate, misura sovrapposizioni.
import json
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from biomap_v2.common import digest
from biomap_v2.dense.v1_support import aggregate_v1, discover_tiles
from biomap_v2.dense.grid import assign
from .state import MEANS

# L'ordine storico è preferito solo quando riproduce numeri e supporto realmente salvati.
def verify(v1_root, geometry_root, grid, local_ids, expected):
    selected=None
    for edge in ['trunc','floor']:
        recomputed=aggregate_v1(v1_root,grid,edge_op=edge)
        if not set(local_ids)<=set(recomputed.index):continue
        r=recomputed.loc[local_ids]
        if np.array_equal(r.n_valid,expected.n_valid) and np.array_equal(r.n_total,expected.n_total) and np.allclose(r[MEANS],expected[MEANS],atol=1e-6,rtol=0):
            selected=edge;break
    if selected is None:raise ValueError('V1 storico non riprodotto dalle tile: nessun confronto omogeneo possibile')
    originals={t['stem']:t for t in discover_tiles(v1_root)};coords=[]
    for tile in sorted(Path(geometry_root).glob('grid_*')):
        info=json.loads((tile/'geometry.json').read_text());old=originals[tile.name]
        if digest(old['scales'])!=info['v1_scales_sha256'] or digest(old['landmask'])!=info['landmask_sha256']:
            raise ValueError('Maschera/scale V1 diverse dalla geometria legacy')
        with np.load(tile/'geometry.npz') as z:flat=z['flat_indices'];cells=z['cell_ids']
        with rasterio.open(old['landmask']) as ds:tf,crs,shape=ds.transform,ds.crs,ds.shape
        crop=rasterio.Affine(*info['transform']);col0,row0=(~tf)*(crop.c,crop.f)
        if str(crs)!=info['crs'] or not np.allclose([col0,row0],np.rint([col0,row0]),atol=1e-6,rtol=0):raise ValueError('Reticolo non allineato')
        if not np.allclose([crop.a,crop.b,crop.d,crop.e],[tf.a,tf.b,tf.d,tf.e],atol=1e-9,rtol=0):raise ValueError('Pixel size differente')
        rr=flat//info['width']+round(row0);cc=flat%info['width']+round(col0)
        if (rr<0).any() or (cc<0).any() or (rr>=shape[0]).any() or (cc>=shape[1]).any():raise ValueError('Pixel fuori tile')
        scales=np.load(old['scales'],mmap_mode='r')
        if not np.isfinite(scales[rr,cc]).all():raise ValueError('Pixel escluso da V1 presente in V2')
        x,y=tf*(cc+.5,rr+.5);lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(x,y)
        r,c=assign(grid.transform,lon,lat,selected)
        if (r<0).any() or (c<0).any() or (r>=grid.shape[0]).any() or (c>=grid.shape[1]).any() or not np.array_equal(grid.cell_index[r,c],cells):
            raise ValueError('Assegnazione fisica delle celle discordante')
        # Centri a circa 1 cm di tolleranza; nessuna deduplicazione dei contributi storici.
        coords.append(np.column_stack([np.rint(lon*1e7).astype(np.int64),np.rint(lat*1e7).astype(np.int64)]))
    coordinates=np.concatenate(coords);n_unique=len(np.unique(coordinates,axis=0))
    return {'status':'passed','edge_op':selected,'pixel_contributions':len(coordinates),
        'unique_centers_approx_1cm':n_unique,'duplicate_contributions_approx_1cm':len(coordinates)-n_unique,
        'v1_means_atol':1e-6,'policy':'historical multiplicity retained for both encoders'}
