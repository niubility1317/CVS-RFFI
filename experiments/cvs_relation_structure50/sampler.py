"""Only permitted U metadata; native random seed schedule preserved."""
import torch
from torch.utils.data import Sampler,DataLoader,RandomSampler
from experiments.cvs_trusted_correction.order import order,validate,physical_pairs

def metadata(dataset):
 if not getattr(dataset,'hide',False) or 'unlabeled' not in dataset.split_source or len(dataset)!=56700:raise ValueError('Only existing hidden source U_s is allowed')
 # Do not read IQ, labels, tx_i, true_tx_i or domain-specific class identities.
 meta={k:torch.tensor([int(getattr(i,k)) for i in dataset.index],dtype=torch.long) for k in ('rx_i','day_i','eq_i','sig_i')}
 meta['base_index']=torch.tensor(dataset.selected.tolist(),dtype=torch.long)
 validate(meta);physical_pairs(meta);return meta

class PhysicalOrderSampler(Sampler):
 def __init__(self,meta,mode):
  self.meta=meta;self.mode=mode;self.n=validate(meta);self.history=[]
  if mode not in ('RANDOM','PAIRS'):raise ValueError('Unknown U order')
  if mode=='PAIRS':physical_pairs(meta)
 def __len__(self):return self.n
 def __iter__(self):
  # Match native RandomSampler(generator=None), including lazy seed timing.
  seed=int(torch.empty((),dtype=torch.int64).random_().item())
  indices=order(self.meta,seed,self.mode);record=dict(iterator=len(self.history)+1,pool_count=self.n,generated_count=len(indices),delivered_count=0,physical_packets_once=True,pair_metadata_verified=self.mode=='PAIRS',U_truth_read=False,IQ_read_for_order=False);self.history.append(record)
  for ix in indices.tolist():
   record['delivered_count']+=1;yield ix

def replace(loader,mode):
 if loader.batch_size!=256 or loader.drop_last or loader.generator is not None or not isinstance(loader.sampler,RandomSampler) or loader.sampler.replacement or loader.sampler.generator is not None:raise ValueError('Native U loader/random seed contract differs')
 sampler=PhysicalOrderSampler(metadata(loader.dataset),mode)
 kw=dict(batch_size=loader.batch_size,sampler=sampler,num_workers=loader.num_workers,pin_memory=loader.pin_memory,drop_last=loader.drop_last,collate_fn=loader.collate_fn,worker_init_fn=loader.worker_init_fn,persistent_workers=loader.persistent_workers,timeout=loader.timeout)
 if loader.num_workers:
  kw['prefetch_factor']=loader.prefetch_factor;kw['multiprocessing_context']=loader.multiprocessing_context
 result=DataLoader(loader.dataset,**kw)
 if len(result)!=222:raise ValueError('Unchanged U step budget required')
 return result
