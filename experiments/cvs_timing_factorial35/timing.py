"""Integer receive-window nuisance; reflect edges, preserve input power,no labels."""
import hashlib
import torch
BOUND=16;TRAIN_SEED=202610174101;SOURCE_SEED=202610174102

def warp(raw,offsets):
 if not torch.is_tensor(raw) or raw.device.type!='cpu' or raw.dtype!=torch.float32 or raw.ndim!=3 or raw.shape[1:]!=(2,256) or not len(raw) or not bool(torch.isfinite(raw).all()) or len(offsets)!=len(raw) or any(type(v)is not int or abs(v)>BOUND for v in offsets):raise ValueError('Finite CPU float32 IQ256 and integer offsets[-16,16] required')
 index=torch.arange(256)[None,:]-torch.tensor(offsets,dtype=torch.long)[:,None];index=torch.where(index<0,-index,index);index=torch.where(index>255,510-index,index)
 out=raw.gather(2,index[:,None,:].expand(-1,2,-1));before=raw.double().square().mean((1,2));after=out.double().square().mean((1,2))
 if not bool(torch.isfinite(before).all() and torch.isfinite(after).all()):raise ValueError('Finite power required')
 gain=torch.where(before>0,torch.sqrt(before/after.clamp_min(1e-30)),torch.ones_like(before)).float();out=out*gain[:,None,None]
 error=(out.double().square().mean((1,2))-before).abs()/before.clamp_min(1e-30)
 if not bool(torch.isfinite(out).all()) or float(error.max())>1e-6:raise ValueError('Power-preserving finite timing view failed')
 relative=torch.linalg.vector_norm((out-raw).flatten(1),dim=1)/torch.linalg.vector_norm(raw.flatten(1),dim=1).clamp_min(1e-20)
 return out,dict(offset_samples=list(offsets),power_relative_error=error.tolist(),relative_L2=relative.tolist(),amplitude_restore_gain=gain.tolist(),indices_min=int(index.min()),indices_max=int(index.max()),full_As_reference_raw_dependency=[3,176],padding='reflect_no_wrap',timing_bound=BOUND)

def transform(raw,ids,view,*,seed=TRAIN_SEED):
 if type(seed)is not int or seed not in (TRAIN_SEED,SOURCE_SEED) or not isinstance(view,str) or not view.startswith(('trainL/','sourceV/')) or len(ids)!=len(raw) or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('Private sourceL/sourceV named unique-view stream only')
 offsets=[int.from_bytes(hashlib.blake2b((str(seed)+'\0'+view+'\0'+pid).encode('utf8'),digest_size=8).digest(),'little')%33-BOUND for pid in ids]
 out,info=warp(raw,offsets);info.update(seed=seed,view=view,label_selection=False);return out,info
