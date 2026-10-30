"""Source-L-only cross-receiver/day episodes, with no OE identity input."""
import numpy as np

SOURCE_RX=(1,3,4,6,8)
ARMS=('S0','S1','M')


def _vector(value,name,length=None):
    a=np.asarray(value)
    if a.ndim!=1 or a.dtype!=np.int64 or (length is not None and len(a)!=length):
        raise ValueError(name+' must be a one-dimensional int64 metadata vector')
    return a.copy()


def active_slots(sample,arm):
    if arm not in ARMS:raise ValueError('arm must be S0, S1 or M')
    order=np.asarray(sample['slot_order'])
    if order.dtype!=np.int64 or order.shape!=(6,) or set(order.tolist())!=set(range(6)):
        raise ValueError('Six source-class slots must form a permutation')
    held=sample['held_class']
    if type(held) is not int or held not in range(6):raise ValueError('held_class must be a source class')
    return order[order!=held].copy() if arm=='M' else order.copy()


class EpisodeSampler:
    def __init__(self,y,rx,day,ids,oe_domain,seed=2026103017,slot_seed=2026103018):
        self.y=_vector(y,'source-L y');n=len(self.y)
        self.rx=_vector(rx,'source-L rx',n);self.day=_vector(day,'source-L day',n)
        ids=np.asarray(ids)
        if ids.shape!=(n,) or ids.dtype.kind!='U' or len(np.unique(ids))!=n or np.any(ids==''):
            raise ValueError('Unique nonempty Unicode source-L opaque IDs required')
        self.ids=ids.copy()
        if set(self.y.tolist())!=set(range(6)) or set(self.rx.tolist())!=set(SOURCE_RX) or set(self.day.tolist())!={1,2,3}:
            raise ValueError('Only the known six-class/five-source-RX/three-day L contract is supported')
        self.cells={}
        for c in range(6):
            for r in SOURCE_RX:
                for d in (1,2,3):
                    rows=np.flatnonzero((self.y==c)&(self.rx==r)&(self.day==d)).astype(np.int64)
                    if len(rows)<16:raise ValueError('Each source-L class/RX/day cell requires at least 16 records, without padding')
                    if len(rows)!=70:raise ValueError('Known source-L contract requires exactly 70 records per cell')
                    self.cells[c,r,d]=rows
        domain=np.asarray(oe_domain)
        if domain.ndim!=1 or domain.dtype.kind not in 'iuU':raise ValueError('OE accepts only a visible one-dimensional domain vector, never identity/role dictionaries')
        values=np.unique(domain)
        if len(values)!=15 or (domain.dtype.kind=='U' and np.any(values=='')):
            raise ValueError('The fifteen nonempty visible OE RX/day domains are required')
        self.oe_rows=[np.flatnonzero(domain==v).astype(np.int64) for v in values]
        if type(seed) is not int or seed<0 or type(slot_seed) is not int or slot_seed<0:
            raise ValueError('Nonnegative integer sampler and slot seeds required')
        self.seed=seed;self.slot_seed=slot_seed
        self.support_pools={(c,r,d):np.concatenate([self.cells[c,other_r,other_d] for other_r in SOURCE_RX if other_r!=r for other_d in (1,2,3) if other_d!=d]) for c in range(6) for r in SOURCE_RX for d in (1,2,3)}

    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('step must be a positive integer')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
        slot_rng=np.random.default_rng(np.random.SeedSequence([self.slot_seed,step]))
        held_class=(step-1)%6;held_rx=SOURCE_RX[((step-1)//6)%5];held_day=1+((step-1)//30)%3
        native=rng.integers(len(self.y),size=128,dtype=np.int64)
        support=np.stack([rng.choice(self.support_pools[c,held_rx,held_day],16,replace=False) for c in range(6)])
        query=np.stack([rng.choice(self.cells[c,held_rx,held_day],16,replace=False) for c in range(6)])
        domains=rng.integers(15,size=64);oe=np.empty(64,np.int64)
        for d,rows in enumerate(self.oe_rows):
            selected=domains==d
            oe[selected]=rows[rng.integers(len(rows),size=int(selected.sum()))]
        return dict(native_L=native,support=support,query=query,OE=oe,slot_order=slot_rng.permutation(6).astype(np.int64),
                    held_class=held_class,held_rx=held_rx,held_day=held_day)
