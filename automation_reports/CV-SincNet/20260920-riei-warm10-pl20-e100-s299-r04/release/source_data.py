"""Source-only indexed loads; no-PL mode never indexes unlabeled IQ records."""
from pathlib import Path
import numpy as np
from get_dataset_lab_unlab import _split_domain_indices
from satellite_channel import apply_space_ground_channel


def warm_view(clean,seed=340299):
    """Single fixed moderate Loo realization, applied before time/FFT encoding."""
    z=clean[:,0]+1j*clean[:,1]
    raw=np.stack([z[:,:256],z[:,512:768]],axis=1).reshape(-1,256)
    faded=apply_space_ground_channel(raw,seed=seed,loo_delta2=1.,loo_sigma2=.0049,normalize='rms',fading_mode='sample')
    padded=np.pad(faded,((0,0),(0,256))).reshape(-1,2,512)
    spectrum=np.fft.fft(padded,axis=-1)/np.sqrt(512.)
    result=np.concatenate([padded[:,0],padded[:,1],spectrum[:,0],spectrum[:,1]],axis=1)
    return np.stack([result.real,result.imag],axis=1).astype(np.float32)


def load_source(root,split_dir,use_unlabeled):
    root=Path(root);Path(split_dir).mkdir(parents=True,exist_ok=True)
    lp,up,lv,yL,yU,yV=[],[],[],[],[],[]
    for domain,rx in enumerate([0,1,2,3,5]):
        y=np.load(root/f'rx_{rx}_y.npy').astype(np.int64)
        labeled,validation,unlabeled=_split_domain_indices(y,.1,.1,299+rx,10)
        np.savez(Path(split_dir)/f'rx_{rx}_split.npz',labeled=labeled,validation=validation,unlabeled=unlabeled)
        clean=np.load(root/f'rx_{rx}_clean_x.npy',mmap_mode='r')
        sg=np.load(root/f'rx_{rx}_sg_x.npy',mmap_mode='r')
        labels=np.column_stack([y,np.full(len(y),domain),np.full(len(y),rx),np.arange(len(y))]).astype(np.int64)
        lp.append(np.stack([clean[labeled],sg[labeled]],axis=1));yL.append(labels[labeled])
        lv.append(np.asarray(clean[validation]));yV.append(labels[validation])
        if use_unlabeled:
            up.append(np.stack([clean[unlabeled],sg[unlabeled]],axis=1))
            hidden=labels[unlabeled].copy();hidden[:,0]=-1;yU.append(hidden)
    cat=lambda parts:np.concatenate(parts).astype(np.float32)
    return (cat(lp),np.concatenate(yL),cat(lv),np.concatenate(yV),
            cat(up) if use_unlabeled else None,np.concatenate(yU) if use_unlabeled else None)
