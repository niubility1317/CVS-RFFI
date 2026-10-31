"""Receiver-held source-only selection of a canonical paired feature space."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr.source_bridge import FeatureBridge
from .milkr.paired_cca import PairedCCA,RANKS
from .milkr import nested_fisher
from .milkr.support_calibrated import target_and_score


def aggregate(records):
    result=[]
    for mode in ('R13',)+RANKS:
        for k in (1,5,10,20):
            for sp in (0,1):
                for qp in (0,1):
                    values=[r['H'] for r in records if (r['mode'],r['k'],r['S'],r['Q'])==(mode,k,sp,qp)]
                    require(len(values)==10,'Incomplete source held-RX policy matrix')
                    result.append(dict(mode=mode,k=k,S=sp,Q=qp,H=float(np.mean(values))))
    return result


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();prior=read(rt['prior_source_selection'])
    require(prior['status']=='SOURCE_FROZEN' and prior['target_reads']==0 and prior['query_reads']==0,'Original source-only rule required')
    rule=dict(prior['K1_rule'],outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata
    banks=[[],[]]
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source pair physical alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('CCA_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks]
    rx_all=np.tile(m['rx_i'],2);records=[]
    for rx in sorted(np.unique(rx_all).tolist()):
        train=np.flatnonzero(rx_all!=rx);raw_train=subset(features[0],train);rest_train=subset(features[1],train)
        adapters={'R13':FeatureBridge.fit(raw_train,rest_train)}
        adapters.update({rank:PairedCCA.fit(raw_train,rest_train,rank) for rank in RANKS})
        for ep in range(2):
            rng=np.random.default_rng(rt['source_proxy_seed']+rx*100+ep)
            groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in range(6)]
            require(all(len(g)==52 for g in groups),'Insufficient source held-RX physical count')
            positions=np.concatenate(groups)+ep*len(m['ids']);pair=[subset(f,positions) for f in features]
            held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
            for mode,adapter in adapters.items():
                mapped=[adapter.transform(pair[0]),pair[1]] if mode=='R13' else [adapter.transform(f,restored=bool(path)) for path,f in enumerate(pair)]
                for k in (1,5,10,20):
                    support=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);sy=np.repeat(np.arange(6),k)
                    ids=[str(m['ids'][int(positions[i]%len(m['ids']))]) for i in support]
                    for sp in (0,1):
                        state=nested_fisher.fit(subset(mapped[sp],support),sy,ids,old_count=3,fixed=rule)
                        for qp in (0,1):
                            pred=state.score(subset(mapped[qp],held)).argmax(1)
                            records.append(dict(mode=mode,rx=int(rx),ep=ep,k=k,S=sp,Q=qp,H=target_and_score(qy,pred,3)))
            print('CCA_HELD_COMPLETE',rx,ep,flush=True)
    summary=aggregate(records);quality=[]
    for rank in RANKS:
        mean=lambda mode,sp=None,qp=None:float(np.mean([r['H'] for r in summary if r['mode']==mode and (sp is None or r['S']==sp) and (qp is None or r['Q']==qp)]))
        gains=[100*(mean(rank,sp,0)-mean('R13',sp,0)) for sp in (0,1)]
        query_gains=[100*(mean(rank,sp,1)-mean(rank,sp,0)) for sp in (0,1)]
        gain=100*(mean(rank)-mean('R13'))
        eligible=gain>=.5 and min(gains)>=-.5 and min(query_gains)>=.5
        quality.append(dict(rank=rank,mean_H=mean(rank),four_path_gain_pp=gain,Q0_gain_pp_by_S=gains,Q1_gain_pp_by_S=query_gains,eligible=bool(eligible)))
    passing=[r for r in quality if r['eligible']];best=max(passing,key=lambda r:(r['mean_H'],-r['rank'])) if passing else None
    if best is not None:
        final=PairedCCA.fit(features[0],features[1],best['rank']);final.save(Path(rt['run_root'])/'source_cca.npz')
        loaded=PairedCCA.load(Path(rt['run_root'])/'source_cca.npz');small=subset(features[0],np.arange(12))
        require(all(np.array_equal(value,loaded.transform(small,restored=False)[key]) for key,value in final.transform(small,restored=False).items()),'CCA checkpoint reload differs')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rule,source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),
        source_summary=summary,source_quality=quality,selected_rank=best['rank'] if best else None,target_eligible=best is not None,
        scientific_verdict='PASS_SOURCE_FOUR_PATH_GATE' if best else 'NO_RANK_PASSES_SOURCE_FOUR_PATH_GATE',target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,
        seconds=time.perf_counter()-started,source_numerical_smoke='PASS',source_RX_training_excludes_held_RX=True,source_support_held_physical_disjoint=True,
        scope='source known-six only; paired CCA trained excluding each held RX, no unseen novel identity confirmation; final source-only parameter checkpoint only at target'))
