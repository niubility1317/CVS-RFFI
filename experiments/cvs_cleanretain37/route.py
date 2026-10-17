"""Retain .75 ordinary supervision, generate all and only .25 practical slots."""
import torch
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
def mask_for_step(step,device):
 if type(step) is not int or step not in range(1,6661):raise ValueError('Ordered actual E30 L step required')
 return (torch.arange(128,device=device)+step).remainder(2)==1

def subset_meta(meta,mask):
 if mask.dtype!=torch.bool or mask.shape!=(128,) or int(mask.sum())!=64 or any(k not in meta or meta[k].shape!=(128,) or meta[k].dtype!=torch.long for k in KEYS):raise ValueError('Original physical metadata for L128 required')
 if set(meta['rx_i'].tolist())-{1,3,4,6,8} or set(meta['day_i'].tolist())-{1,2,3} or set(meta['eq_i'].tolist())!={1}:raise ValueError('Full source RX/day/eq only; no target or U')
 return {k:meta[k][mask.to(meta[k].device)] for k in KEYS}

def compose(batch,ordinary,y,domain,mask):
 if ordinary.shape!=(128,2,256) or ordinary.dtype!=torch.float32 or y.shape!=(128,) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or domain.shape!=(128,) or domain.dtype!=torch.long or not bool(((domain>=0)&(domain<15)).all()) or mask.dtype!=torch.bool or mask.shape!=(128,) or int(mask.sum())!=64:raise ValueError('Real L128/domain and64 supervised slots required')
 if batch.x.shape!=(128,2,256) or not bool(torch.isfinite(batch.x).all()) or not bool(torch.isfinite(ordinary).all()) or not torch.equal(batch.y,torch.cat((y[mask],y[mask]))) or not torch.equal(batch.d_raw,torch.cat((domain[mask],domain[mask]))):raise ValueError('Each generated satellite must keep its own physical TX/domain label')
 secondary=ordinary.clone();secondary[mask]=batch.x[64:]
 for name in ('nuisance','nuisance_valid'):
  value=getattr(batch,name,None)
  if value is not None:
   if value.ndim<1 or value.shape[0]!=128:raise ValueError('Native nuisance alignment differs')
   expanded=torch.zeros((256,*value.shape[1:]),dtype=value.dtype,device=value.device);indices=mask.nonzero().flatten().to(value.device)+128;expanded[indices]=value[64:];setattr(batch,name,expanded)
 batch.x=torch.cat((ordinary,secondary));batch.y=torch.cat((y,y));batch.d_raw=torch.cat((domain,domain));batch.clean_batch_size=128;batch.total_batch_size=256
 return batch
