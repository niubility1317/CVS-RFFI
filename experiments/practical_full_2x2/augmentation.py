"""Training readers: original L IQ in the auxiliary slot, with matched CE dose."""
from .cache import Reader
from .compat import tensor

class BatchAugment:
    def __init__(self,c):
        self.c=c;self.indices=None;self.epoch=None
        self.reader=Reader(c) if c['satellite_train_augmentation'] else None
    def __call__(self,x,progress,batch_index,stage=1):
        if self.indices is None or len(self.indices)!=len(x):raise ValueError('L physical index missing')
        if self.reader is None:return x,dict(applied=False,applied_count=0,scenario='unchanged_L',probability=0.,scene_counts={-1:len(x)})
        a,m=self.reader.get(self.indices,self.epoch,stage,progress,self.c['epochs'])
        return tensor(a,x.device),m

class PAMFPractical:
    def __init__(self,c):
        import numpy as np
        from pathlib import Path
        self.c=c;self.reader=Reader(c) if c['channel_augmentation'] else None
        with np.load(Path(c['source_cache'])/'L_s_metadata.npz',allow_pickle=False) as m:
            self.lookup={tuple(int(m[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')):i for i in range(len(m['ids']))}
    def indices(self,meta):
        values={k:v.detach().cpu().tolist() for k,v in meta.items() if k in ('eq_i','rx_i','day_i','sig_i','base_index')}
        if set(values)!=set(('eq_i','rx_i','day_i','sig_i','base_index')):raise ValueError('Physical metadata missing')
        return [self.lookup[tuple(int(values[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))] for i in range(len(values['eq_i']))]
    def __call__(self,x,meta,epoch,step):
        if self.reader is None:return x
        a,_=self.reader.get(self.indices(meta),epoch,1,(epoch-1)/50.,50)
        return tensor(a,x.device)

class PAMFConcat(PAMFPractical):
    def prepare(self,raw,meta,epoch,step):
        self.raw=raw;self.meta=meta;self.epoch=epoch
    def __call__(self,ordinary,progress,batch_index,stage=1):
        if self.reader is None:return self.raw,dict(applied=False,applied_count=0)
        a,m=self.reader.get(self.indices(self.meta),self.epoch,1,(self.epoch-1)/50.,50)
        return tensor(a,ordinary.device),m
