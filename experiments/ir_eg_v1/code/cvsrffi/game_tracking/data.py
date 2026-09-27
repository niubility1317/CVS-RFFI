"""Source boundary: the learner never receives hidden TX metadata."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl

def opaque_id(item):
    # Builder-only: no physical TX indices are propagated through the ID text.
    raw = f'{item.tx_i}/{item.rx_i}/{item.day_i}/{item.eq_i}/{item.sig_i}'
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

class SourceRoleDataset(Dataset):
    def __init__(self, base, indices, role, domains):
        self.base, self.indices, self.role, self.domains = base, list(indices), role, domains
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        k = self.indices[i]
        x, y, _, raw = self.base[k]
        key = (int(raw['rx_i']), int(raw['day_i']))
        meta = {n: int(raw[n]) for n in ('rx_i','day_i','eq_i','sig_i')}
        meta['sample_id'] = opaque_id(self.base.index[k])
        # Opaque ordering index preserves the historical temporal rule; not a TX field.
        meta['base_index'] = k
        meta['role'] = self.role
        if self.role == 'unlabeled':
            y = -1
        else:
            meta['capture_group'] = f'{raw["tx_i"]}:{raw["rx_i"]}:{raw["day_i"]}'
        return x, y, self.domains[key], meta

@dataclass
class SourceData:
    train: Dataset
    unlabeled: Dataset
    val: Dataset
    domains: dict
    info: dict
    def loader(self, role, batch, *, seed=0, shuffle=False, workers=0, drop_last=False):
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(getattr(self, role), batch_size=batch, shuffle=shuffle,
                          num_workers=workers, drop_last=drop_last, generator=gen,
                          persistent_workers=False)

    def unlabeled_epoch_loader(self,batch,*,epoch_index,steps,seed,workers=0):
        # Permute contiguous windows once, then rotate through that order over
        # epochs. This covers the full U pool while retaining temporal neighbors;
        # it never groups/sorts/stratifies using hidden TX labels.
        blocks=[list(range(i,min(i+batch,len(self.unlabeled)))) for i in range(0,len(self.unlabeled),batch)]
        if not blocks: raise ValueError('Empty unlabeled pool')
        if len(blocks)>1 and len(blocks[-1])==1:
            blocks[-2].extend(blocks.pop())
        order=torch.randperm(len(blocks),generator=torch.Generator().manual_seed(seed+810019)).tolist()
        chosen=[blocks[order[(epoch_index*steps+i)%len(blocks)]] for i in range(steps)]
        return DataLoader(self.unlabeled,batch_sampler=chosen,num_workers=workers,
                          generator=torch.Generator().manual_seed(seed+epoch_index),persistent_workers=False)

def build_source(args):
    if args.game_synthetic:
        return synthetic_source(args)
    raw = load_wisig_compact_pkl(args.wisig_pkl)
    rxs = [int(v) for v in args.wisig_train_rxs.split(',')]
    days = [int(v) for v in args.wisig_train_days.split(',')]
    base = WiSigCompactDataset(raw, out_len=args.wisig_out_len, crop_mode='center', normalize=True,
                              equalized=int(args.wisig_equalized), day_keep=days, rx_keep=rxs,
                              domain='rx_day', seed=args.game_split_seed)
    if not len(base):
        raise ValueError('Empty source data')
    if {it.rx_i for it in base.index} != set(rxs) or {it.day_i for it in base.index} != set(days):
        raise ValueError('Actual source physical RX/day IDs differ from contract')
    groups = defaultdict(list)
    for k, it in enumerate(base.index):
        groups[(it.tx_i,it.rx_i,it.day_i,it.eq_i)].append(k)
    parts = [[], [], []]
    # Historical contiguous-in-signal-index split with current legal proportions.
    # Split seed is recorded and shared; this rule itself has no stochastic step.
    for key, indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda i: base.index[i].sig_i)
        n = len(indices)
        nl, nu = round(.07*n), round(.63*n)
        if nl < 1 or nu < 1 or n-nl-nu < 1:
            raise ValueError('Capture cannot satisfy L/U/V roles')
        parts[0].extend(indices[:nl]); parts[1].extend(indices[nl:nl+nu]); parts[2].extend(indices[nl+nu:])
    domains = {key: i for i, key in enumerate(sorted({(it.rx_i,it.day_i) for it in base.index}))}
    datasets = [SourceRoleDataset(base, idx, role, domains) for idx,role in zip(parts,('train','unlabeled','val'))]
    ids = [{opaque_id(base.index[k]) for k in idx} for idx in parts]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError('Physical role overlap')
    args.num_classes = len(raw['tx_list'])
    info = {'schema':'core90_game_source_roles_v1','source_rxs':rxs,'source_days':days,
            'num_classes':args.num_classes,'domain_kind':'rx_day','domain_map':[[*k,v] for k,v in domains.items()],
            'counts':dict(zip(('L_s','U_s','V'),map(len,parts))), 'ratios':[.07,.63,.30],
            'split_rule':'per_tx_rx_day_eq_contiguous_signal_index','split_seed':args.game_split_seed,
            'role_ids':{r:sorted(v) for r,v in zip(('L_s','U_s','V'),ids)},
            'checkpoint_init':'scratch_only','target_access_before_freeze':False,
            'capture_group_granularity':'whole_tx_rx_day_container; no signal-window split'}
    return SourceData(*datasets, domains, info)

class SyntheticRole(Dataset):
    def __init__(self, role, seed):
        self.role = role
        gen = torch.Generator().manual_seed(seed)
        self.items = []
        for tx in range(6):
            for rx in range(3):
                for day in range(3):
                    for sig in range(2):
                        x = torch.randn(2,256,generator=gen) * .2
                        t = torch.arange(256.)
                        x[0] += torch.sin(t*(.05+.02*tx))
                        x[1] += torch.cos(t*(.05+.02*tx)) + rx*.1
                        meta = dict(rx_i=rx,day_i=day,eq_i=1,sig_i=sig,base_index=len(self.items),
                                    sample_id=f'{role}-{len(self.items)}',role=role)
                        if role != 'unlabeled':
                            meta['capture_group'] = f'{tx}:{rx}:{day}'
                        self.items.append((x,-1 if role=='unlabeled' else tx,rx*3+day,meta))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def synthetic_source(args):
    args.num_classes = 6
    domains = {(r,d):r*3+d for r in range(3) for d in range(3)}
    return SourceData(SyntheticRole('train',1),SyntheticRole('unlabeled',2),SyntheticRole('val',3),domains,
                      {'synthetic':True,'num_classes':6,'domain_kind':'rx_day','counts':{'L_s':108,'U_s':108,'V':108},
                       'checkpoint_init':'scratch_only','target_access_before_freeze':False})

def audit_indices(source, per_capture=4):
    """Only labeled source metadata; whole capture containers stay on one side."""
    records = []
    for i in range(len(source.train)):
        # Read metadata from known builder structures; avoid materializing IQ for indexing.
        if isinstance(source.train, SourceRoleDataset):
            item = source.train.base.index[source.train.indices[i]]
            tx, rx, day = item.tx_i, item.rx_i, item.day_i
        else:
            _, tx, _, meta = source.train[i]
            rx, day = meta['rx_i'], meta['day_i']
        records.append((i,tx,rx,day))
    txs = sorted({r[1] for r in records}); days = sorted({r[3] for r in records})
    if len(txs) < 2 or len(days) < 2:
        raise ValueError('Independent source audit needs multiple TX capture groups and days')
    fit_tx = set(txs[:len(txs)//2]); fit_days = set(days[:-1])
    groups = defaultdict(list)
    for i,t,r,d in records: groups[(t,r,d)].append(i)
    result = dict(domain_fit=[],domain_monitor=[],capability_fit=[],capability_monitor=[])
    for (t,r,d), idx in sorted(groups.items()):
        chosen = idx[:per_capture]
        result['domain_fit' if t in fit_tx else 'domain_monitor'].extend(chosen)
        result['capability_fit' if d in fit_days else 'capability_monitor'].extend(chosen)
    return result


def labeled_source_records(source):
    """Metadata-only view of L_s. Never inspect U_s to construct TX groups."""
    rows = []
    for i in range(len(source.train)):
        if isinstance(source.train, SourceRoleDataset):
            item = source.train.base.index[source.train.indices[i]]
            tx, rx, day = int(item.tx_i), int(item.rx_i), int(item.day_i)
            # Selection identity is the legal L_s index; opaque IDs are read by loader.
            identity = str(source.train.indices[i])
        else:
            _, tx, _, meta = source.train[i]
            tx, rx, day = int(tx), int(meta['rx_i']), int(meta['day_i'])
            identity = str(meta['sample_id'])
        rows.append(dict(index=i,tx=tx,rx=rx,day=day,capture_group=f'{tx}:{rx}:{day}',
                         sample_id=identity,role='train'))
    return rows


def balanced_reference_indices(records, samples_per_group=1, seed=0):
    """Choose every available capture container, without global RNG mutation.

    Whole-container coverage is reported, not invented independent acquisitions.
    Caller compares available coverage with the source contract when required.
    """
    import hashlib
    if samples_per_group < 1 or not records:
        raise ValueError('nonempty labeled source reference and positive samples required')
    groups = defaultdict(list)
    identities = set()
    for row in records:
        if row.get('role') != 'train' or int(row.get('tx',-1)) < 0 or not row.get('capture_group'):
            raise ValueError('reference requires legal labeled source capture metadata')
        if row['sample_id'] in identities:
            raise ValueError('duplicate reference sample identity')
        identities.add(row['sample_id'])
        groups[(int(row['tx']),int(row['rx']),int(row['day']))].append(row)
    chosen=[]
    for key,values in sorted(groups.items()):
        order=sorted(values,key=lambda r:hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).digest())
        chosen.extend(r['index'] for r in order[:samples_per_group])
    coverage=dict(valid=True,groups=len(groups),samples=len(chosen),
                  domains=len({(r,d) for t,r,d in groups}),txs=len({t for t,r,d in groups}),
                  rxs=sorted({r for t,r,d in groups}),days=sorted({d for t,r,d in groups}),
                  group_counts={f'{t}:{r}:{d}':min(len(v),samples_per_group) for (t,r,d),v in groups.items()},
                  grouping_kind='whole_tx_rx_day_container',independent_repeat_claim=False)
    return chosen,coverage


def audit_indices_v2(source, per_capture=4, seed=0):
    records=labeled_source_records(source)
    lag,_=balanced_reference_indices(records,per_capture,seed)
    reference,_=balanced_reference_indices(records,1,seed+1)
    txs=sorted({r['tx'] for r in records});days=sorted({r['day'] for r in records})
    if len(txs)<2 or len(days)<2:
        raise ValueError('source diagnostics need multiple TX and day containers')
    fit_tx=set(txs[:len(txs)//2]);fit_days=set(days[:-1])
    by_index={r['index']:r for r in records}
    result=dict(lag_reference=lag,gradient_reference=reference)
    for name,predicate in [('domain_fit',lambda r:r['tx'] in fit_tx),
                           ('domain_monitor',lambda r:r['tx'] not in fit_tx),
                           ('capability_fit',lambda r:r['day'] in fit_days),
                           ('capability_monitor',lambda r:r['day'] not in fit_days)]:
        result[name]=[i for i in lag if predicate(by_index[i])]
    result['response_fit']=[i for i in reference if by_index[i]['tx'] in fit_tx]
    result['response_monitor']=[i for i in reference if by_index[i]['tx'] not in fit_tx]
    return result
