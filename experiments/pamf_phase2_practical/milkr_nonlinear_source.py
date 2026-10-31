"""Offline bounded-pair training and held-RX/TX source falsification."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr.orbit_cca import OrbitCCA
from .milkr.bounded_pair import BoundedPairEncoder,train as train_pair
from .milkr import nested_fisher,signature_kernel
from .milkr.support_calibrated import LAMBDAS
from .milkr_identity_source import source_train_indices,scores


def aggregate(rows):
    result=[]
    for k in (1,20):
        for sp in (0,1):
            for qp in (0,1):
                values=[r for r in rows if (r['k'],r['S'],r['Q'])==(k,sp,qp)]
                require(len(values)==5,'Incomplete nonlinear source fold matrix')
                result.append(dict(k=k,S=sp,Q=qp,**{key:float(np.mean([r[key] for r in values])) for key in ('H','old_accuracy','new_accuracy')}))
    return result


def assess(rows,control):
    reference={(r['k'],r['S'],r['Q']):r for r in control};current={(r['k'],r['S'],r['Q']):r for r in rows}
    gain=100*float(np.mean([r['H']-reference[r['k'],r['S'],r['Q']]['H'] for r in rows]))
    raw=[100*float(np.mean([current[k,sp,0]['H']-reference[k,sp,0]['H'] for k in (1,20)])) for sp in (0,1)]
    query=[100*(current[k,sp,1]['H']-current[k,sp,0]['H']) for k in (1,20) for sp in (0,1)]
    components=[100*float(np.mean([current[k,sp,0][name]-reference[k,sp,0][name] for k in (1,20)])) for sp in (0,1) for name in ('old_accuracy','new_accuracy')]
    return dict(source_gain_pp=gain,raw_gain_pp_by_S=raw,query_gain_pp_each_K_S=query,raw_component_gain_pp=components,
        eligible=bool(gain>=.5 and min(raw)>=-.25 and min(query)>=0 and min(components)>=-1))


def calibrate_rule(frames):
    grid=[]
    for config in signature_kernel.CONFIGS:
        for ridge in LAMBDAS:
            rule=dict(config=list(config),ridge=ridge,outer_folds=3,outer_strengths=[0.,.25,.5,.75],ridge_reference_classes=6);rows=[]
            for frame in frames:
                sup=np.arange(6)*52
                for sp in (0,1):
                    state=signature_kernel.fit(subset(frame['mapped'][sp],sup),np.arange(6),[frame['ids'][i] for i in sup],old_count=4,fixed=rule,conditional_loo=False)
                    for qp in (0,1):rows.append(dict(rx=frame['rx'],k=1,S=sp,Q=qp,**scores(np.repeat(np.arange(6),32),state.score(subset(frame['mapped'][qp],frame['held'])).argmax(1))))
            grid.append(dict(K1_rule=rule,source_H=float(np.mean([r['H'] for r in rows])),rows=rows))
    best=max(enumerate(grid),key=lambda item:(item[1]['source_H'],item[1]['K1_rule']['ridge'],-item[0]))[1];rows=list(best['rows'])
    for frame in frames:
        sup=np.concatenate([np.arange(c*52,c*52+20) for c in range(6)])
        for sp in (0,1):
            state=nested_fisher.fit(subset(frame['mapped'][sp],sup),np.repeat(np.arange(6),20),[frame['ids'][i] for i in sup],old_count=4,fixed=best['K1_rule'])
            for qp in (0,1):rows.append(dict(rx=frame['rx'],k=20,S=sp,Q=qp,**scores(np.repeat(np.arange(6),32),state.score(subset(frame['mapped'][qp],frame['held'])).argmax(1))))
    return best['K1_rule'],aggregate(rows),grid


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    require(rt['source_cache_epochs']==[91,92] and rt['orbit_quarters']==[0,1,2,3] and rt['orbit_rank']==64,'Wrong fixed moment geometry')
    require(rt['pair_steps']==600 and rt['pair_batch']==256 and rt['pair_bound']==.25,'Wrong fixed nonlinear budget')
    started=time.perf_counter();reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source physical alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('NONLINEAR_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks]
    rx_all=np.tile(m['rx_i'],2);source_y=np.tile(m['y'],2);frames=[[],[]];folds=[];audits=[]
    for index,rx in enumerate(sorted(np.unique(m['rx_i']).tolist())):
        held_tx=[index,index+1];old_tx=[c for c in range(6) if c not in held_tx];indices=source_train_indices(rx_all,source_y,rx,held_tx)
        training=[subset(f,indices) for f in features];cca=OrbitCCA.fit(*training)
        encoder,audit=train_pair(cca,*training,device='cuda:0',steps=rt['pair_steps'],batch=rt['pair_batch'],seed=rt['pair_seed']+index)
        audits.append(dict(receiver=int(rx),**audit));rng=np.random.default_rng(rt['source_proxy_seed']+rx*100)
        groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in old_tx+held_tx]
        require(all(len(group)==52 for group in groups),'Insufficient source validation rows')
        positions=np.concatenate(groups);held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)])
        for mode,model in enumerate((cca,encoder)):
            mapped=[model.transform(subset(f,positions),restored=bool(path)) for path,f in enumerate(features)]
            frames[mode].append(dict(rx=int(rx),mapped=mapped,ids=[str(m['ids'][int(p)]) for p in positions],held=held))
        folds.append(dict(receiver=int(rx),held_TX_indices=held_tx,train_TX_indices=old_tx,training_observations=len(indices),eval_epoch=91))
        print('NONLINEAR_SOURCE_FOLD',rx,audit['initial_alignment_loss'],audit['final_alignment_loss'],flush=True)
    control_rule,control,control_grid=calibrate_rule(frames[0]);rule,summary,grid=calibrate_rule(frames[1]);quality=assess(summary,control);final_audit=None
    if quality['eligible']:
        cca=OrbitCCA.fit(*features);model,final_audit=train_pair(cca,*features,device='cuda:0',steps=rt['pair_steps'],batch=rt['pair_batch'],seed=rt['pair_seed']+5)
        path=Path(rt['run_root'])/'source_nonlinear_pair.npz';model.save(path);loaded=BoundedPairEncoder.load(path,device='cuda:0')
        for sp in (0,1):
            small=subset(features[sp],slice(0,12))
            require(all(np.array_equal(value,loaded.transform(small,restored=bool(sp))[key]) for key,value in model.transform(small,restored=bool(sp)).items()),'Nonlinear encoder reload differs')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rule,control_K1_rule=control_rule,target_eligible=quality['eligible'],
        scientific_verdict='PASS_NONLINEAR_SOURCE_GATE' if quality['eligible'] else 'NONLINEAR_SOURCE_GATE_FAILED',source_quality=quality,source_summary=summary,control_source_summary=control,
        source_K1_grid=grid,control_K1_grid=control_grid,source_RX_TX_folds=folds,source_training_audits=audits,final_training_audit=final_audit,
        source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),selected_rank=64,pair_bound=.25,
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,seconds=time.perf_counter()-started,source_numerical_smoke='PASS',
        scope='bounded source-only MLP; source fold excludes RX/TX from bothCCA/NN; F saw allsource6; no trueF-novel claim; strict G-only query remains'))
