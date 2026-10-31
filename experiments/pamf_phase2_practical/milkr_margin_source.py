"""Source-only max-margin head diagnostic; no target evidence selects C."""
from pathlib import Path
import time
import numpy as np
import sklearn
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr_cca_source import reuse_source_checkpoint
from .milkr.paired_cca import PairedCCA
from .milkr.kernel_margin import fit_grid,CS
from .milkr.support_calibrated import target_and_score


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();reuse_source_checkpoint(rt,'source_cca_provenance.json')
    provenance=read(Path(rt['run_root'])/'source_cca_provenance.json');require(provenance['selected_rank']==16,'Fixed sourceCCA16 required')
    rule=provenance['K1_rule'];reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source pair alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('MARGIN_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks];rx_all=np.tile(m['rx_i'],2);records=[]
    for rx in sorted(np.unique(rx_all).tolist()):
        train=np.flatnonzero(rx_all!=rx);adapter=PairedCCA.fit(subset(features[0],train),subset(features[1],train),16)
        for ep in range(2):
            rng=np.random.default_rng(rt['source_proxy_seed']+rx*100+ep);class_order=rng.permutation(6)
            groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in class_order]
            require(all(len(g)==52 for g in groups),'Insufficient source held-RX physical count')
            positions=np.concatenate(groups)+ep*len(m['ids']);mapped=[adapter.transform(subset(f,positions),restored=bool(path)) for path,f in enumerate(features)]
            held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
            for k in (1,5,10,20):
                support=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);sy=np.repeat(np.arange(6),k);ids=[str(m['ids'][int(positions[i]%len(m['ids']))]) for i in support]
                for old in (2,3,4,5):
                    for sp in (0,1):
                        states=fit_grid(subset(mapped[sp],support),sy,ids,old_count=old,fixed=rule)
                        for C,state in states.items():
                            for qp in (0,1):records.append(dict(C=C,rx=int(rx),ep=ep,k=k,old=old,S=sp,Q=qp,H=target_and_score(qy,state.score(subset(mapped[qp],held)).argmax(1),old)))
            print('MARGIN_HELD_COMPLETE',rx,ep,flush=True)
    summary=[]
    for C in ('ridge',)+CS:
        for k in (1,5,10,20):
            for sp in (0,1):
                for qp in (0,1):
                    values=[r['H'] for r in records if (r['C'],r['k'],r['S'],r['Q'])==(C,k,sp,qp)];require(len(values)==40,'Incomplete source margin matrix')
                    summary.append(dict(C=C,k=k,S=sp,Q=qp,H=float(np.mean(values))))
    mean=lambda C,sp=None,qp=None,k=None:float(np.mean([r['H'] for r in summary if r['C']==C and (sp is None or r['S']==sp) and (qp is None or r['Q']==qp) and (k is None or r['k']==k)]))
    quality=[]
    for C in CS:
        gain=100*(mean(C)-mean('ridge'));raw=[100*(mean(C,sp,0)-mean('ridge',sp,0)) for sp in (0,1)];inc=[100*(mean(C,sp,1)-mean(C,sp,0)) for sp in (0,1)]
        minimum=min(100*(mean(C,k=k)-mean('ridge',k=k)) for k in (1,5,10,20));eligible=gain>=.5 and min(raw)>=-.5 and min(inc)>0 and minimum>=-.5
        quality.append(dict(C=C,source_H=mean(C),source_gain_pp=gain,Q0_gain_pp_by_S=raw,Q1_minus_Q0_pp_by_S=inc,min_K_gain_pp=minimum,eligible=bool(eligible)))
    passing=[r for r in quality if r['eligible']];best=max(passing,key=lambda r:(r['source_H'],-r['C'])) if passing else None
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=dict(rule,margin_C=best['C'] if best else None),target_eligible=best is not None,
        scientific_verdict='PASS_SOURCE_MARGIN_GATE' if best else 'NO_C_PASSES_SOURCE_MARGIN_GATE',source_summary=summary,source_quality=quality,sklearn_version=sklearn.__version__,
        source_old_class_counts=[2,3,4,5],source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),selected_rank=16,source_numerical_smoke='PASS',
        source_RX_training_excludes_held_RX=True,source_support_held_physical_disjoint=True,target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,source_final_CCA_new_fit=False,seconds=time.perf_counter()-started,
        scope='explicit kernel-C-SVC objective variant; default neural head unchanged; source six allknownF, not novel confirmation; no query-role/quota/statistics/fitting'))
