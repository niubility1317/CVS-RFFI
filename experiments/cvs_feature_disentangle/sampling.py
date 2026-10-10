"""Private, source-labelled relationship stream; never replaces native sampling."""
from collections import Counter, defaultdict, deque
from itertools import combinations
import copy
import numpy as np
import torch
from torch.utils.data import default_collate


def records_from_dataset(dataset):
    if getattr(dataset, 'hide', False) or getattr(dataset, 'tx_label_visible', True) is False:
        raise ValueError('Relationship stream requires visible source L labels')
    if getattr(dataset, 'transform', None) is not None or getattr(getattr(dataset, 'base', None), 'transform', None) is not None:
        raise ValueError('Relationship stream requires deterministic clean IQ dataset')
    if getattr(getattr(dataset, 'base', dataset), 'crop_mode', 'center') == 'random':
        raise ValueError('Random dataset cropping is outside the private relationship stream')
    records = []
    for local, item in enumerate(dataset.index):
        get = lambda key: item[key] if isinstance(item, dict) else getattr(item, key)
        base = int(dataset.selected[local]) if hasattr(dataset, 'selected') else local
        rx, day, eq, sig = (int(get(k)) for k in ('rx_i','day_i','eq_i','sig_i'))
        collection = getattr(item, 'collection_id', None) if not isinstance(item, dict) else item.get('collection_id')
        records.append(dict(index=local, y=int(get('tx_i')), rx=rx, day=day,
                            pid=(rx,day,eq,sig,base), time_index=sig, collection=collection))
    if len({r['pid'] for r in records}) != len(records):
        raise ValueError('Source L contains repeated physical identities')
    return records


class RelationStream:
    """One balanced B48 every invocation, with all 30 blocks fairly rotated.

    Each cell queue exhausts before reshuffling. At a cycle boundary repeats
    already present in the current batch are deferred, never duplicated.
    sig_i is used as an order proxy for dispersal, not claimed as independent
    acquisition time. No DataLoader iterator/base-seed/global RNG is involved.
    """
    def __init__(self, dataset, seed, mode='balanced', txs=range(6), receivers=None, days=None):
        if mode not in ('balanced','random'): raise ValueError(mode)
        self.dataset, self.mode = dataset, mode
        self.records = records_from_dataset(dataset)
        self.rng = np.random.default_rng(int(seed))
        self.txs = tuple(txs)
        self.receivers = tuple(sorted({r['rx'] for r in self.records}) if receivers is None else receivers)
        self.days = tuple(sorted({r['day'] for r in self.records}) if days is None else days)
        if len(self.txs)!=6 or len(self.receivers)!=5 or len(self.days)!=3: raise ValueError('B48 requires six TX, five RX, three days')
        self.cells = defaultdict(list)
        for r in self.records:
            if r['y'] in self.txs and r['rx'] in self.receivers and r['day'] in self.days:
                self.cells[(r['y'],r['rx'],r['day'])].append(r['index'])
        self.blocks = [(r1,r2,d) for r1,r2 in combinations(self.receivers,2) for d in self.days]
        self.block_queue = deque(); self.queues = {}; self.cycles = Counter()
        self.coverage = Counter(); self.pid_counts = Counter(); self.last_seen = {}
        self.calls=0; self.appearances=0; self.max_wait=0; self.skipped_blocks=Counter(); self.insufficient_cells=Counter()
        self.last_step=0; self.last={}
        self.random_queue=deque()
        if mode=='random' and len(self.records)<48: raise ValueError('Random B48 needs 48 unique L packets')

    def _permutation(self, indices):
        # Spread each group of four over order quartiles when available. This
        # preserves a full no-replacement cycle with randomized within-bin order.
        ordered=sorted(indices,key=lambda i: (str(self.records[i]['collection']),self.records[i]['time_index']))
        bins=[list(self.rng.permutation(x)) for x in np.array_split(ordered,4)]
        out=[]
        while any(bins):
            for b in self.rng.permutation(4):
                if bins[int(b)]: out.append(int(bins[int(b)].pop()))
        return out

    def _take(self,key,indices,count):
        queue=self.queues.setdefault(key,deque())
        selected=[]; postponed=[]
        while len(selected)<count:
            if not queue:
                order=self.rng.permutation(indices).tolist() if key==('random',) else self._permutation(indices)
                queue.extend(order); self.cycles[key]+=1
            i=queue.popleft()
            if i in selected: postponed.append(i)
            else: selected.append(i)
        queue.extendleft(reversed(postponed))
        return selected

    def next_indices(self,step):
        step=int(step); self.last_step=step
        if self.mode=='random':
            indices=self._take(('random',),list(range(len(self.records))),48); block=None
        else:
            indices=None; block=None
            for _ in range(len(self.blocks)):
                if not self.block_queue: self.block_queue.extend(self.blocks[int(i)] for i in self.rng.permutation(len(self.blocks)))
                candidate=self.block_queue.popleft(); r1,r2,day=candidate
                keys=[(y,r,day) for r in (r1,r2) for y in self.txs]
                bad=[k for k in keys if len(self.cells[k])<4]
                if bad:
                    self.skipped_blocks[candidate]+=1
                    self.insufficient_cells.update(bad); continue
                indices=[i for k in keys for i in self._take(k,self.cells[k],4)]
                block=candidate; break
            if indices is None:
                self.last=dict(step=step,skipped=True,reason='no_supported_relationship_block')
                return None
        pids=[self.records[i]['pid'] for i in indices]
        if len(indices)!=48 or len(set(pids))!=48: raise AssertionError('B48 duplicate physical package')
        self.calls+=1; self.appearances+=48; self.pid_counts.update(pids)
        if block is not None:
            wait=step-self.last_seen.get(block,0); self.max_wait=max(self.max_wait,wait)
            self.last_seen[block]=step; self.coverage[block]+=1
        self.last=dict(step=step,block=block,skipped=False,unique_pid=48,indices=indices)
        return indices

    def next_batch(self,step,device=None):
        indices=self.next_indices(step)
        if indices is None: return None
        batch=default_collate([self.dataset[i] for i in indices])
        if device is not None:
            batch=[v.to(device) if torch.is_tensor(v) else v for v in batch]
        return tuple(batch)

    def summary(self,step=None):
        step=self.last_step if step is None else int(step)
        waits=[step-self.last_seen.get(b,0) for b in self.blocks]
        return dict(mode=self.mode,calls=self.calls,packet_appearances=self.appearances,
                    unique_pid=len(self.pid_counts),repeat_appearances=self.appearances-len(self.pid_counts),
                    repeat_rate=(self.appearances-len(self.pid_counts))/self.appearances if self.appearances else None,
                    blocks_covered=len(self.coverage),blocks_total=30,
                    block_coverage={str(b):self.coverage[b] for b in self.blocks},
                    max_wait_native_steps=max([self.max_wait]+waits),
                    skipped_blocks={str(k):v for k,v in self.skipped_blocks.items()},
                    insufficient_cells={str(k):v for k,v in self.insufficient_cells.items()},
                    acquisition_metadata='collection_id' if all(r['collection'] is not None for r in self.records) else 'collection unknown; sig_i order proxy used, independence not established',
                    cell_cycles={str(k):v for k,v in self.cycles.items()})

    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.bit_generator.state, queues=self.queues,cycles=self.cycles,
          block_queue=self.block_queue,coverage=self.coverage,pid_counts=self.pid_counts,last_seen=self.last_seen,
          calls=self.calls,appearances=self.appearances,max_wait=self.max_wait,skipped_blocks=self.skipped_blocks,
          insufficient_cells=self.insufficient_cells,last_step=self.last_step,last=self.last))

    def load_state_dict(self,state):
        state=copy.deepcopy(state); self.rng.bit_generator.state=state.pop('rng')
        for key,value in state.items(): setattr(self,key,value)
