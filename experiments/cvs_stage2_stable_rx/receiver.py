"""Cross-TX/day additive RX potential, frozen full-vector relation targets.

Receiver identity is used only to build source relations. No target environment
lookup or inference correction is defined. The TX x RX interaction is retained.
"""
import time
import torch
from . import design as d
from experiments.cvs_stage2_geometry.receiver import (
    SCHEMA, _source, _build_groups, _check_features, _frozen, FOLDS,
    block_potentials, vector_reliability, trusted_covariance, load_receiver)
from experiments.cvs_stage2_geometry.identity_screen import estimate_rx


@torch.no_grad()
def fit_receiver(cache,seed):
    started=time.perf_counter()
    data=_source(cache);parent=cache['parent_feature'].detach().cpu().clone()
    _check_features(parent,len(data['ids']))
    groups,blocks,decisions,_=_build_groups(data)
    days=sorted(data['day'].unique().tolist());rxs=sorted(data['rx'].unique().tolist())
    if len(days)<2 or len(rxs)<2:raise ValueError('Cross-day multi-RX source groups required')
    # Equal complete cells are essential to the additive interpretation; reject
    # missing groups instead of silently mixing unequal TX/day distributions.
    if len(blocks)!=6*len(days):raise ValueError('Expected one complete quality-collapsed TX/day block')
    for block in blocks:
        if sorted(groups[g]['rx'] for g in block['group_ids'])!=rxs:
            raise ValueError('Unbalanced receiver support in a TX/day block')
    for group in groups:group['parent_feature_mean']=parent[group['indices']].double().mean(0)
    means=torch.stack([g['parent_feature_mean'] for g in groups]);folds=[]
    for held_tx in FOLDS:
        fit_tx=[y for y in range(6) if y not in held_tx]
        for held_day in days:
            index=len(folds)
            offsets=estimate_rx(means,groups,blocks,held_day,fit_tx)
            raw_matrix=torch.stack([offsets[r] for r in rxs])
            singular=torch.linalg.svdvals(raw_matrix-raw_matrix.mean(0))
            rank=int((singular>max(1e-12,float(singular[0])*1e-10)).sum())
            if rank>len(rxs)-1:raise ValueError('Centered RX estimator rank exceeds RX count minus one')
            held_blocks=[b for b in blocks if b['tx'] in held_tx and b['day']==held_day]
            for block in held_blocks:
                gids=block['group_ids']
                geometry=block_potentials(means[gids],torch.stack([offsets[groups[g]['rx']] for g in gids]))
                block.update({k:v for k,v in geometry.items() if k not in ('raw_potential','potential','target_mean')})
                block['fold']=index
                for pos,g in enumerate(gids):
                    groups[g].update(fold=index,raw_potential=geometry['raw_potential'][pos],
                        potential=geometry['potential'][pos],target_feature_mean=geometry['target_mean'][pos])
            # Empty basis requests the established full160 scalar MSE. There is
            # no rank-2 projection; subspace diagnostics are marked N/A below.
            reliability=vector_reliability(groups,held_blocks,torch.empty(160,0))
            reliability.update(mse_subspace=None,mse_outside_subspace=None,
                               subspace_diagnostic='N/A_no_truncated_basis')
            fit_indices=[i for i,(y,day) in enumerate(zip(data['y'].tolist(),data['day'].tolist()))
                         if y in fit_tx and day!=held_day]
            folds.append(dict(fold=index,fit_tx=fit_tx,heldout_tx=list(held_tx),heldout_day=held_day,
                fit_days=[day for day in days if day!=held_day],rank=rank,rx_offsets=offsets,
                fit_ids=len(fit_indices),fit_source_ids=[data['ids'][i] for i in fit_indices],
                fit_method='equal_RX_centered_TX_day_cells_fixed_shrink04',optimizer_steps=0,**reliability))
    payload=dict(schema=SCHEMA,method=d.METHOD,seed=int(seed),feature_dim=160,state_dim=40,
        groups=groups,blocks=blocks,folds=folds,grouping_decisions=decisions,
        source_count=len(parent),source_ids=data['ids'],source_y=data['y'],source_rx=data['rx'],
        source_day=data['day'],parent_feature=parent,parent_feature_bytes=parent.numel()*parent.element_size(),
        target_access=False,optimizer_steps=0,max_rank=len(rxs)-1,rank_truncation=False,shrink=.4,
        alpha_max=.25,max_pair_norm=2.,reliability='scalar_full160_equal_block_pair_after_kappa_before_alpha',
        fit_seconds=time.perf_counter()-started,
        diagnostics=dict(legal_groups=len(groups),condition_blocks=len(blocks),folds=len(folds),
                         unique_physical_ids=len(parent),legal_group_ids=sum(g['count'] for g in groups)))
    payload['diagnostics']['trusted_covariance']=trusted_covariance(payload)
    return _frozen(payload)
