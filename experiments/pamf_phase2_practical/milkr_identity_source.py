"""Adapter-TX/RX-held source calibration; frozen F has seen all six TX.

Source validation simulates identities absent from the new linear encoder fit.
It is explicitly not a never-seen-by-F or target-blind confirmation claim.
"""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr.paired_cca import PairedCCA,RANKS
from .milkr.source_class_metric import within_class_covariances,fuse_metric
from .milkr import nested_fisher,signature_kernel
from .milkr.support_calibrated import LAMBDAS

STRENGTHS=(0.,.5,.9)
SOURCE_K=(1,5,20)


def source_train_indices(rx_all,y_all,rx,held_tx):
    train=np.flatnonzero((rx_all!=rx)&~np.isin(y_all,held_tx))
    require(len(train)>0 and not np.isin(y_all[train],held_tx).any() and (rx_all[train]!=rx).all(),'Source TX/RX held-out leakage')
    return train


def truncate(cca,rank):
    return PairedCCA([(mx,my,px[:,:rank],py[:,:rank]) for mx,my,px,py in cca.parameters],rank)


def scores(labels,pred,old=4):
    accuracy=np.asarray([(pred[labels==c]==c).mean() for c in range(6)])
    a=float(accuracy[:old].mean());b=float(accuracy[old:].mean())
    return dict(H=2*a*b/(a+b) if a+b else 0.,new_accuracy=b,old_accuracy=a)


def summarize(rows):
    out=[]
    for k in SOURCE_K:
        for sp in (0,1):
            for qp in (0,1):
                selected=[r for r in rows if (r['k'],r['S'],r['Q'])==(k,sp,qp)]
                require(len(selected)==5,'Incomplete adapter-heldout source matrix')
                out.append(dict(k=k,S=sp,Q=qp,**{name:float(np.mean([r[name] for r in selected])) for name in ('H','new_accuracy','old_accuracy')}))
    return out


def assess(summary,control):
    ref={(r['k'],r['S'],r['Q']):r for r in control};value={(r['k'],r['S'],r['Q']):r for r in summary}
    gain=float(np.mean([100*(r['H']-ref[r['k'],r['S'],r['Q']]['H']) for r in summary]))
    new=float(np.mean([100*(r['new_accuracy']-ref[r['k'],r['S'],r['Q']]['new_accuracy']) for r in summary]))
    raw=min(100*(value[k,sp,0]['H']-ref[k,sp,0]['H']) for k in SOURCE_K for sp in (0,1))
    k1new=[100*(value[1,sp,0]['new_accuracy']-ref[1,sp,0]['new_accuracy']) for sp in (0,1)]
    query=[100*(value[k,sp,1]['H']-value[k,sp,0]['H']) for k in SOURCE_K for sp in (0,1)]
    average_query=[float(np.mean([100*(value[k,sp,1]['H']-value[k,sp,0]['H']) for k in SOURCE_K])) for sp in (0,1)]
    eligible=gain>=.5 and new>=1. and raw>=.25 and min(k1new)>=1. and min(query)>=0 and min(average_query)>0
    return dict(source_H=float(np.mean([r['H'] for r in summary])),source_gain_pp=gain,new_macro_gain_pp=new,
        min_K_Q0_gain_pp=float(raw),K1_new_Q0_gain_pp_by_S=k1new,min_K_Q1_minus_Q0_pp=float(min(query)),
        Q1_minus_Q0_pp_by_S=average_query,eligible=bool(eligible))


def calibrate(rt):
    if rt.get('source_identity_grid_runtime'):
        from .milkr_identity_reuse_source import calibrate as reuse_grid
        return reuse_grid(rt)
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();prior=read(rt['prior_source_selection'])
    require(prior['status']=='SOURCE_FROZEN' and prior['target_reads']==0 and prior['query_reads']==0,'Original source-only kernel rule required')
    default_rule=dict(prior['K1_rule'],outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    grid=[dict(config=list(config),ridge=ridge,outer_folds=3,outer_strengths=[0.,.25,.5,.75],ridge_reference_classes=6) for config in signature_kernel.CONFIGS for ridge in LAMBDAS]
    old_index=next(i for i,rule in enumerate(grid) if rule['config']==default_rule['config'] and rule['ridge']==default_rule['ridge'])
    reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
    require(rt['source_cache_epochs']==[91,92],'Fixed two source simulation epochs required')
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source physical alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('IDENTITY_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks]
    rx_all=np.tile(m['rx_i'],2);source_y=np.tile(m['y'],2);folds=[];frames={(rank,strength):[] for rank in RANKS for strength in STRENGTHS};middle={key:[] for key in frames}
    for fold,rx in enumerate(sorted(np.unique(rx_all).tolist())):
        held_tx=[fold%6,(fold+1)%6];old_tx=[c for c in range(6) if c not in held_tx];train=source_train_indices(rx_all,source_y,rx,held_tx)
        training=[subset(f,train) for f in features];maximum=PairedCCA.fit(*training,96)
        rng=np.random.default_rng(rt['source_proxy_seed']+rx*100);class_order=old_tx+held_tx
        groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in class_order]
        require(all(len(g)==52 for g in groups),'Insufficient source held-RX physical count')
        positions=np.concatenate(groups);held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
        folds.append(dict(receiver=int(rx),held_TX_indices=held_tx,train_TX_indices=old_tx,training_observations=len(train),eval_epoch=91))
        for rank in RANKS:
            shadow=truncate(maximum,rank);cov=within_class_covariances(shadow,*training,source_y[train],allow_subset=True)
            for strength in STRENGTHS:
                encoder=fuse_metric(shadow,cov,strength);mapped=[encoder.transform(subset(f,positions),restored=bool(path)) for path,f in enumerate(features)]
                frames[rank,strength].append(dict(rx=int(rx),mapped=mapped,ids=[str(m['ids'][int(p)]) for p in positions],held=held))
                for k in (5,20):
                    sup=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);ys=np.repeat(np.arange(6),k)
                    for sp in (0,1):
                        state=nested_fisher.fit(subset(mapped[sp],sup),ys,[str(m['ids'][int(positions[i])]) for i in sup],old_count=4,fixed=default_rule)
                        for qp in (0,1):middle[rank,strength].append(dict(rx=int(rx),k=k,S=sp,Q=qp,**scores(qy,state.score(subset(mapped[qp],held)).argmax(1))))
            print('IDENTITY_REP_COMPLETE',rx,rank,flush=True)
    # K1 source kernels are recalibrated for each representation, not inherited
    # from the earlier pre-CCA geometry. All tuning remains on source labels.
    candidates=[];control=None
    for representation,items in frames.items():
        rank,strength=representation
        for index,rule in enumerate(grid):
            rows=list(middle[representation])
            for item in items:
                sup=np.arange(6)*52;y=np.arange(6);qy=np.repeat(np.arange(6),32)
                for sp in (0,1):
                    state=signature_kernel.fit(subset(item['mapped'][sp],sup),y,[item['ids'][i] for i in sup],old_count=4,fixed=rule,conditional_loo=False)
                    for qp in (0,1):rows.append(dict(rx=item['rx'],k=1,S=sp,Q=qp,**scores(qy,state.score(subset(item['mapped'][qp],item['held'])).argmax(1))))
            summary=summarize(rows);record=dict(rank=rank,strength=strength,rule_index=index,K1_rule=rule,summary=summary)
            candidates.append(record)
            if rank==16 and strength==0. and index==old_index:control=summary
        print('IDENTITY_K1_GRID_COMPLETE',rank,strength,flush=True)
    require(control is not None,'Missing fixed R14 algorithm source control')
    for candidate in candidates:candidate.update(assess(candidate['summary'],control))
    passing=[c for c in candidates if c['eligible']]
    ordering=lambda c:(c['source_H'],c['new_macro_gain_pp'],c['K1_rule']['ridge'],-c['rank'],-c['strength'],-c['rule_index'])
    best=max(passing,key=ordering) if passing else None;quality=[]
    for representation in frames:
        local=[c for c in candidates if (c['rank'],c['strength'])==representation];eligible=[c for c in local if c['eligible']]
        chosen=max(eligible or local,key=ordering)
        quality.append({key:value for key,value in chosen.items() if key!='summary'})
    write(Path(rt['run_root'])/'source_calibration_grid.json',dict(candidates=candidates,source_only=True,target_reads=0))
    if best:
        final=PairedCCA.fit(*features,best['rank']);cov=within_class_covariances(final,*features,source_y)
        fused=fuse_metric(final,cov,best['strength']);fused.save(Path(rt['run_root'])/'source_identity_cca.npz')
        loaded=PairedCCA.load(Path(rt['run_root'])/'source_identity_cca.npz');small=subset(features[0],np.arange(12))
        require(all(np.array_equal(value,loaded.transform(small,restored=False)[key]) for key,value in fused.transform(small,restored=False).items()),'Identity-held source encoder reload differs')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=best['K1_rule'] if best else dict(default_rule,ridge_reference_classes=6),
        target_eligible=best is not None,scientific_verdict='PASS_ADAPTER_IDENTITY_HELDOUT_GATE' if best else 'NO_SOURCE_CANDIDATE_PASSES_IDENTITY_HELDOUT_GATE',
        source_quality=quality,source_summary=best['summary'] if best else control,selected_rank=best['rank'] if best else None,selected_strength=best['strength'] if best else None,
        source_RX_TX_folds=folds,source_validation_K=list(SOURCE_K),source_validation_epochs=[91],source_fit_epochs=[91,92],
        source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),source_candidates=len(candidates),K1_grid_per_representation=len(grid),
        source_RX_training_excludes_held_RX=True,source_TX_training_excludes_held_TX=True,source_support_held_physical_disjoint=True,source_numerical_smoke='PASS',
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,seconds=time.perf_counter()-started,
        scope='adapter TX/RX heldout only; F already saw all six; source old4/new2 not target old6/new20 confirmation; current targetK1 ridge scaled by allocated support class count/6, never query count'))
