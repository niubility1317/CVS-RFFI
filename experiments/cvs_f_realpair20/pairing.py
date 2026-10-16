"""Sample intact true-label source L observations. No signal synthesis or fitting."""
from collections import Counter,defaultdict
import numpy as np
import torch
from . import design as d
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
FIT=(3,4,6,8)
def key(meta,i=None):return tuple(int(meta[k] if i is None else meta[k][i]) for k in KEYS)

class Bank:
    def __init__(self,entries):
        self.entries=entries;self.groups=defaultdict(list);self.index={};self.calls=0;self.packets=0;self.coverage=Counter();self.unique=set()
        if len(entries)!=5040:raise ValueError('Exact filtered L5040 bank required')
        for i,(x,y,domain,meta) in enumerate(entries):
            pid=key(meta);eq,rx,day,_,base=pid
            if (x.shape!=(2,256) or not bool(torch.isfinite(x).all()) or int(y) not in range(6) or rx not in FIT or day not in (1,2,3) or eq!=1
                or int(domain)!=(3,4,6,8).index(rx)*3+3+day-1 or pid in self.index):raise ValueError('Finite distinct original fit-L IQ/label/domain required')
            self.index[pid]=i;self.groups[(int(y),rx,day)].append(i)
        if len({p[-1] for p in self.index})!=5040 or set(self.groups)!={(t,r,day) for t in range(6) for r in FIT for day in (1,2,3)} or any(len(g)!=70 for g in self.groups.values()):
            raise ValueError('Complete6TX x4RX x3day x70 legal L bank required')
    @classmethod
    def from_dataset(cls,dataset,domain_label_map):
        if len(dataset)!=5040 or bool(getattr(dataset,'hide',False)) or getattr(dataset,'transform',None) is not None:raise ValueError('Filtered unaugmented visible L dataset only')
        # Dataset domains refer to the original full RX/day lookup. Native
        # training remaps them through data_ctx before any model forward.
        mapping={int(k):int(v) for k,v in domain_label_map.items()};entries=[]
        for i in range(len(dataset)):
            x,y,raw_domain,meta=dataset[i]
            if int(raw_domain) not in mapping:raise ValueError('Raw partner domain missing from original native label map')
            entries.append((x.detach().cpu().clone(),int(y),mapping[int(raw_domain)],{k:int(meta[k]) for k in KEYS}))
        return cls(entries)
    def select(self,labels,meta,step,mode,primary_x):
        if mode not in d.ARMS or step!=self.calls+1 or step not in range(1,4441):raise ValueError('One ordered source partner draw per update')
        labels=labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)
        if len(labels)!=128:raise ValueError('Original L128 anchor batch required')
        if primary_x.shape!=(128,2,256):raise ValueError('Original L IQ128 required')
        primary=[]
        for j in range(128):
            i=self.index.get(key(meta,j))
            if i is None:raise ValueError('Primary outside the fit L bank')
            primary.append(self.entries[i][0])
        if not torch.equal(primary_x.detach().cpu(),torch.stack(primary)):raise ValueError('Bank and actual native primary preprocessing differ')
        random=np.random.default_rng(np.random.SeedSequence([d.PAIR_SEED,step]));u=random.random((128,2));selected=[];records=[]
        for j,y in enumerate(labels):
            pid=key(meta,j);i=self.index.get(pid)
            if i is None or int(y)!=self.entries[i][1]:raise ValueError('Primary packet must belong to identical legal L bank with true label')
            _,rx,day,_,_=pid;others=[r for r in FIT if r!=rx];target=rx if mode=='SAME_RX' else others[min(2,int(u[j,0]*3))]
            candidates=[k for k in self.groups[(int(y),target,day)] if k!=i]
            chosen=candidates[min(len(candidates)-1,int(u[j,1]*len(candidates)))];entry=self.entries[chosen];other=key(entry[3])
            if other==pid or entry[1]!=int(y) or other[2]!=day or ((other[1]==rx)!=(mode=='SAME_RX')):raise ValueError('Real sameTX/day distinct-packet RX pairing differs')
            selected.append(entry);records.append(dict(primary=list(pid),partner=list(other),TX=int(y),mode=mode))
            self.coverage[(int(y),rx,target,day)]+=1;self.unique.add(other)
        self.calls=step;self.packets+=128
        return torch.stack([e[0] for e in selected]),torch.tensor([e[2] for e in selected],dtype=torch.long),records
    def summary(self):
        return dict(calls=self.calls,packets=self.packets,unique_partner_packets=len(self.unique),bank_packets=5040,bank_IQ_reads=5040,
            strata=[dict(TX=k[0],primary_RX=k[1],partner_RX=k[2],day=k[3],count=v) for k,v in sorted(self.coverage.items())],
            U_used=False,V_used=False,held_RX_used=False,target_access=False,IQ_mixing=False,synthetic_hardware=False)
