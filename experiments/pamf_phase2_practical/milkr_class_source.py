"""Source class-noise projection selection with an explicit K1 requirement."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr_cca_source import reuse_source_checkpoint
from .milkr.paired_cca import PairedCCA
from .milkr.source_class_metric import within_class_covariances,fuse_metric,STRENGTHS
from .milkr import nested_fisher
from .milkr.support_calibrated import target_and_score

SOURCE_K=(1,5,20)


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();reuse_source_checkpoint(rt,'source_cca_provenance.json')
    provenance=read(Path(rt['run_root'])/'source_cca_provenance.json');require(provenance['selected_rank']==16,'Fixed R14 source CCA16 required')
    rule=provenance['K1_rule'];reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
    require(rt['source_cache_epochs']==[91,92],'Fixed two source simulation epochs required')
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source physical pair alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('CLASS_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks]
    rx_all=np.tile(m['rx_i'],2);source_y=np.tile(m['y'],2);records=[]
    for rx in sorted(np.unique(rx_all).tolist()):
        train=np.flatnonzero(rx_all!=rx);training=[subset(f,train) for f in features]
        shadow=PairedCCA.fit(training[0],training[1],16);covariances=within_class_covariances(shadow,*training,source_y[train])
        rng=np.random.default_rng(rt['source_proxy_seed']+rx*100);class_order=rng.permutation(6)
        groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in class_order]
        require(all(len(g)==52 for g in groups),'Insufficient held sourceRX physical samples')
        positions=np.concatenate(groups);held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
        for strength in STRENGTHS:
            encoder=fuse_metric(shadow,covariances,strength);mapped=[encoder.transform(subset(f,positions),restored=bool(path)) for path,f in enumerate(features)]
            for k in SOURCE_K:
                support=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);sy=np.repeat(np.arange(6),k);ids=[str(m['ids'][int(positions[i])]) for i in support]
                for sp in (0,1):
                    state=nested_fisher.fit(subset(mapped[sp],support),sy,ids,old_count=3,fixed=rule)
                    for qp in (0,1):records.append(dict(strength=strength,rx=int(rx),k=k,S=sp,Q=qp,H=target_and_score(qy,state.score(subset(mapped[qp],held)).argmax(1),3)))
        print('CLASS_HELD_COMPLETE',rx,flush=True)
    summary=[]
    for strength in STRENGTHS:
        for k in SOURCE_K:
            for sp in (0,1):
                for qp in (0,1):
                    values=[r['H'] for r in records if (r['strength'],r['k'],r['S'],r['Q'])==(strength,k,sp,qp)];require(len(values)==5,'Incomplete source class-noise matrix')
                    summary.append(dict(strength=strength,k=k,S=sp,Q=qp,H=float(np.mean(values))))
    mean=lambda strength,sp=None,qp=None,k=None:float(np.mean([r['H'] for r in summary if r['strength']==strength and (sp is None or r['S']==sp) and (qp is None or r['Q']==qp) and (k is None or r['k']==k)]))
    quality=[]
    for strength in STRENGTHS[1:]:
        gain=100*(mean(strength)-mean(0.));k1=[100*(mean(strength,sp,0,1)-mean(0.,sp,0,1)) for sp in (0,1)]
        minimum=min(100*(mean(strength,sp,0,k)-mean(0.,sp,0,k)) for sp in (0,1) for k in SOURCE_K)
        qgain=[100*(mean(strength,sp,1)-mean(strength,sp,0)) for sp in (0,1)]
        min_q=min(100*(mean(strength,sp,1,k)-mean(strength,sp,0,k)) for sp in (0,1) for k in SOURCE_K)
        eligible=gain>=.5 and min(k1)>=.5 and minimum>=-.25 and min(qgain)>0 and min_q>=0
        quality.append(dict(strength=strength,source_H=mean(strength),source_gain_pp=gain,K1_Q0_gain_pp_by_S=k1,min_K_Q0_gain_pp=minimum,Q1_minus_Q0_pp_by_S=qgain,min_K_Q1_minus_Q0_pp=min_q,eligible=bool(eligible)))
    passing=[r for r in quality if r['eligible']];best=max(passing,key=lambda r:(r['source_H'],-r['strength'])) if passing else None
    if best:
        original=PairedCCA.load(Path(rt['run_root'])/'source_cca.npz');full_covariances=within_class_covariances(original,*features,source_y)
        fused=fuse_metric(original,full_covariances,best['strength']);fused.save(Path(rt['run_root'])/'source_class_cca.npz')
        loaded=PairedCCA.load(Path(rt['run_root'])/'source_class_cca.npz');small=subset(features[0],np.arange(12))
        require(all(np.array_equal(value,loaded.transform(small,restored=False)[key]) for key,value in fused.transform(small,restored=False).items()),'Fused encoder checkpoint reload differs')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=dict(rule,source_class_metric_strength=best['strength'] if best else 0.),target_eligible=best is not None,
        scientific_verdict='PASS_SOURCE_CLASS_K1_GATE' if best else 'NO_STRENGTH_PASSES_SOURCE_CLASS_K1_GATE',source_summary=summary,source_quality=quality,selected_rank=16,
        source_validation_K=list(SOURCE_K),source_validation_epochs=[91],source_fit_epochs=[91,92],source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),
        source_metric_label_roles='L_s only; no additional labels',source_final_encoder='source_class_cca.npz' if best else None,source_numerical_smoke='PASS',
        source_RX_training_excludes_held_RX=True,source_support_held_physical_disjoint=True,target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,seconds=time.perf_counter()-started,
        scope='source-supervised linear encoder parameters only; no source covariance/means/prototypes/examples in target; known-six proxy not novel confirmation; targetK10 interpolates sourceK5/20'))
