"""Screen a CE-head batch proxy against complete source-gradient relations."""
import json
import sys
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.source_branch_gradients import _relation
from . import source_component_diagnostic as base

OFFSETS = tuple(range(8))
_snapshots = []


class ProxyObjective(base.ComponentObjective):
    def guide(self, labeled, student):
        named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        ce = getattr(student, '_diagnostic_identity_loss', None)
        if not torch.is_tensor(ce) or not ce.requires_grad or set(self.components) != set(base.PARTS):
            raise ValueError('Actual ordinary L CE and all native U components required')
        def grad(value):
            return tuple(None if x is None else x.detach().clone() for x in
                         torch.autograd.grad(value, [p for _,p in named], retain_graph=True, allow_unused=True))
        snapshot = dict(named=[(n,p.detach()) for n,p in named],
                        L_CE=grad(ce), L_full=grad(labeled),
                        U_CE=grad(self.components['pseudo_CE']), U_full=grad(self.pending_gradient_loss))
        result = super().guide(labeled, student)
        _snapshots.append(snapshot)
        return result


def summarize(records):
    result = {}
    for setting in sorted({r['setting'] for r in records}):
        rows = sorted((r for r in records if r['setting'] == setting), key=lambda x:x['offset'])
        if len(rows)<3 or len({r['offset'] for r in rows})!=len(rows):
            raise ValueError('Three or more distinct batches per setting required')
        shapes=[(n,tuple(p.shape)) for n,p in rows[0]['named']]
        if any([(n,tuple(p.shape)) for n,p in r['named']] != shapes for r in rows):
            raise ValueError('Gradient parameter identities differ')
        pairs=[]; ranking=[]
        for left in rows:
            options=[]
            for right in rows:
                if left is right:continue
                proxy=_relation(left['named'],left['L_CE'],right['U_CE'],'cls_head')
                actual={group:_relation(left['named'],left['L_full'],right['U_full'],group)
                        for group in base.original.GROUPS}
                item=dict(labeled_offset=left['offset'],unlabeled_offset=right['offset'],
                          proxy_CE_head=proxy,actual_full_L_vs_raw_U=actual)
                pairs.append(item)
                if proxy['cosine'] is not None:options.append(item)
            if len(options)<2:raise ValueError('At least two defined U proxy candidates required per labeled batch')
            options.sort(key=lambda x:(x['proxy_CE_head']['cosine'],x['unlabeled_offset']))
            ranking.append(dict(labeled_offset=left['offset'],lowest=options[0],highest=options[-1],
                                defined_candidates=len(options)))
        result[setting]=dict(cross_batch_pairs=pairs,per_L_extremes=ranking)
    return result


def collect(runtime):
    if _snapshots:raise ValueError('A single diagnostic execution per process is required')
    previous=base.ComponentObjective,base.original.BATCH_OFFSETS
    base.ComponentObjective,base.original.BATCH_OFFSETS=ProxyObjective,OFFSETS
    try:
        result=base.collect(runtime)
    finally:
        base.ComponentObjective,base.original.BATCH_OFFSETS=previous
    if len(result['rows']) != len(_snapshots):
        raise ValueError('Every fixed batch requires one gradient snapshot')
    for snap,row in zip(_snapshots,result['rows']):
        snap.update(setting=row['setting'],offset=row['batch_offset'])
    result['proxy_screen']=summarize(_snapshots)
    _snapshots.clear()
    result.update(status='SOURCE_CE_HEAD_PROXY_SCREEN_COMPLETE',fixed_offsets=list(OFFSETS),
        interpretation='Each original E20 L128 and U256 source batch is computed independently. '
        'For each L, rank other U batches by ordinary L CE versus native U pseudo-CE head gradient; '
        'compare full-L versus raw full-U gradients within each parameter group. These dependent '
        'cross-batch relations do not simulate recombined U64 groups, MixStyle or contrast after '
        'regrouping, later AdamW moments, a training intervention, or target accuracy.')
    return result


if __name__ == '__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python -m experiments.pamf_repl_attribution_factorial.source_proxy_diagnostic RUNTIME_JSON')
    print(json.dumps(collect(read(sys.argv[1])),ensure_ascii=False,allow_nan=False),flush=True)
