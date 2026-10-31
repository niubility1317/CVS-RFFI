"""Asymmetric source group-count proxy for a support-OOF H decision rule."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr_cca_source import reuse_source_checkpoint
from .milkr.paired_cca import PairedCCA
from .milkr.balanced_h import fit_grid,KAPPAS
from .milkr.support_calibrated import target_and_score


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();reuse_source_checkpoint(rt,'source_cca_provenance.json')
    provenance=read(Path(rt['run_root'])/'source_cca_provenance.json');require(provenance['selected_rank']==16,'Fixed R14 source rank16 required')
    rule=provenance['K1_rule'];reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source pair alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('H_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks];rx_all=np.tile(m['rx_i'],2);records=[]
    for rx in sorted(np.unique(rx_all).tolist()):
        train=np.flatnonzero(rx_all!=rx);adapter=PairedCCA.fit(subset(features[0],train),subset(features[1],train),16)
        for ep in range(2):
            rng=np.random.default_rng(rt['source_proxy_seed']+rx*100+ep);class_order=rng.permutation(6)
            groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in class_order]
            require(all(len(g)==52 for g in groups),'Insufficient source held-RX physical count')
            positions=np.concatenate(groups)+ep*len(m['ids'])
            mapped=[adapter.transform(subset(f,positions),restored=bool(path)) for path,f in enumerate(features)]
            held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
            for k in (1,5,10,20):
                support=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);sy=np.repeat(np.arange(6),k)
                ids=[str(m['ids'][int(positions[i]%len(m['ids']))]) for i in support]
                for old in (2,3,4,5):
                    for sp in (0,1):
                        states=fit_grid(subset(mapped[sp],support),sy,ids,old_count=old,fixed=rule)
                        for kappa,state in states.items():
                            for qp in (0,1):
                                pred=state.score(subset(mapped[qp],held)).argmax(1)
                                records.append(dict(kappa=kappa,rx=int(rx),ep=ep,k=k,old=old,S=sp,Q=qp,H=target_and_score(qy,pred,old)))
            print('H_HELD_COMPLETE',rx,ep,flush=True)
    summary=[]
    for kap in KAPPAS:
        for k in (1,5,10,20):
            for sp in (0,1):
                for qp in (0,1):
                    values=[r['H'] for r in records if (r['kappa'],r['k'],r['S'],r['Q'])==(kap,k,sp,qp)]
                    require(len(values)==40,'Incomplete asymmetric source policy matrix')
                    summary.append(dict(kappa=kap,k=k,S=sp,Q=qp,H=float(np.mean(values))))
    mean=lambda kap,sp=None,qp=None,k=None:float(np.mean([r['H'] for r in summary if r['kappa']==kap and (sp is None or r['S']==sp) and (qp is None or r['Q']==qp) and (k is None or r['k']==k)]))
    quality=[]
    for kap in KAPPAS[:-1]:
        gain=100*(mean(kap)-mean('none'));raw=[100*(mean(kap,sp,0)-mean('none',sp,0)) for sp in (0,1)];qgain=[100*(mean(kap,sp,1)-mean(kap,sp,0)) for sp in (0,1)]
        minimum=min(100*(mean(kap,k=k)-mean('none',k=k)) for k in (1,5,10,20))
        eligible=gain>=.5 and min(raw)>=-.5 and min(qgain)>0 and minimum>=-.5
        quality.append(dict(kappa=kap,source_H=mean(kap),source_gain_pp=gain,Q0_gain_pp_by_S=raw,Q1_minus_Q0_pp_by_S=qgain,min_K_gain_pp=minimum,eligible=bool(eligible)))
    passing=[r for r in quality if r['eligible']];best=max(passing,key=lambda r:(r['source_H'],r['kappa'])) if passing else None
    fixed=dict(rule,H_bias_kappa=best['kappa'] if best else 'none')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=fixed,target_eligible=best is not None,
        scientific_verdict='PASS_ASYMMETRIC_SOURCE_H_GATE' if best else 'NO_SHRINKAGE_PASSES_SOURCE_H_GATE',source_summary=summary,source_quality=quality,
        source_old_class_counts=[2,3,4,5],source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),
        selected_rank=16,source_numerical_smoke='PASS',source_RX_training_excludes_held_RX=True,source_support_held_physical_disjoint=True,
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,source_final_CCA_new_fit=False,seconds=time.perf_counter()-started,
        scope='known-source6 with asymmetric group counts; no target query roles/quotas/fitting; K1 and B unchanged R14; current labeled support only at target'))
