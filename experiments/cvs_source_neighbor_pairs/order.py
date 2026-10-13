"""Source physical-neighbor pairs: allowed metadata only, no TX or labels."""
import torch
from experiments.cvs_source_gate_audit.gates import FIELDS

def validate(meta):
 if set(meta)!=FIELDS:raise ValueError('Only physical RX/day/eq/sig/base_index metadata is allowed')
 n=len(meta['base_index'])
 if n<2 or n%2:raise ValueError('Pair study requires an even allowed pool')
 if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.long or v.device.type!='cpu' or v.ndim!=1 or len(v)!=n for v in meta.values()):raise ValueError('Invalid physical metadata tensors')
 if len(torch.unique(meta['base_index']))!=n or (meta['base_index']<0).any() or (meta['sig_i']<0).any():raise ValueError('Duplicate/negative physical indices')
 return n

def physical_pairs(meta):
 n=validate(meta);pairs=torch.argsort(meta['base_index']).reshape(-1,2);a,b=pairs.unbind(1)
 same=all(torch.equal(meta[k][a],meta[k][b]) for k in ('rx_i','day_i','eq_i'))
 gap=meta['base_index'][b]-meta['base_index'][a];sig=meta['sig_i'][b]-meta['sig_i'][a]
 if not same or not ((gap>=1)&(gap<=2)&(sig==gap)).all():raise ValueError('Allowed pool does not form disjoint physical neighbors')
 return pairs

def order(meta,seed,sampling):
 n=validate(meta);gen=torch.Generator().manual_seed(seed)
 if sampling=='RANDOM':return torch.randperm(n,generator=gen)
 if sampling!='PAIRS':raise ValueError('Unregistered sampling method')
 pairs=physical_pairs(meta);pairs=pairs[torch.randperm(n//2,generator=gen)]
 # Random orientation within each pair; only this private generator is consumed.
 flips=torch.randint(2,(n//2,),generator=gen).bool();pairs[flips]=pairs[flips].flip(1)
 result=pairs.reshape(-1)
 if not torch.equal(result.sort().values,torch.arange(n)):raise AssertionError('Each allowed physical packet must appear exactly once')
 return result
