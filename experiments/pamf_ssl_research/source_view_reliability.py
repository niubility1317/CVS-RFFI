"""Frozen original-E20 source-view learnability;no fitted state or hidden U truth."""
import torch
from torch.nn import functional as F


def summarize(raw,view,role,truth=None,thresholds=(.8,.9,.95)):
    if role not in ('U_s','V') or (role=='U_s' and truth is not None):raise ValueError('Only source V truth or hidden-label U allowed')
    if (raw.ndim!=2 or raw.shape!=view.shape or raw.shape[1]!=6 or not len(raw)
        or not bool(torch.isfinite(raw).all() and torch.isfinite(view).all())
        or bool((raw<0).any() or (view<0).any())
        or not torch.allclose(raw.sum(1),torch.ones(len(raw)),atol=1e-5,rtol=1e-5)
        or not torch.allclose(view.sum(1),torch.ones(len(view)),atol=1e-5,rtol=1e-5)):
        raise ValueError('Finite normalized same-record source probabilities required')
    if role=='V' and (truth is None or truth.shape!=(len(raw),) or bool(((truth<0)|(truth>=6)).any())):
        raise ValueError('Matching source V labels required')
    conf,pseudo=raw.max(1);aug_conf,aug_pred=view.max(1);accepted=conf>=.95;agree=pseudo==aug_pred
    logp=view.clamp_min(torch.finfo(view.dtype).tiny).log()
    hard=F.nll_loss(logp,pseudo,reduction='none');smooth=.99*hard-.01*logp.mean(1)
    def counts(mask):
        n=int(mask.sum());good=int((mask&(pseudo==truth)).sum()) if truth is not None else None
        return dict(accepted=n,coverage_percent=100*n/len(raw),raw_pseudo_correct=good,
            raw_pseudo_precision_percent=None if not n or good is None else 100*good/n,
            same_input_view_agreement=int((mask&agree).sum()),
            view_prediction_correct=int((mask&(aug_pred==truth)).sum()) if truth is not None else None,
            hard_CE_to_raw_pseudo_mean=None if not n else float(hard[mask].mean()),
            LS01_CE_to_raw_pseudo_mean=None if not n else float(smooth[mask].mean()))
    gates=[]
    for threshold in thresholds:
        if not 0<threshold<=1:raise ValueError('Valid fixed diagnostic threshold required')
        for require_agreement in (False,True):
            mask=accepted&(aug_conf>=threshold)&(agree if require_agreement else torch.ones_like(agree))
            gates.append(dict(augmented_confidence_threshold=threshold,require_raw_label_agreement=require_agreement,**counts(mask)))
    return dict(samples=len(raw),role=role,raw_confidence_threshold=.95,raw_accepted=counts(accepted),
        all_view_prediction_correct=int((aug_pred==truth).sum()) if truth is not None else None,gates=gates,
        U_truth_read=False,label_purity_is_raw_label_correctness_not_view_easiness=True)


def collect(rt,plan,analysis_commit):
    import gc
    import time
    from collections import Counter
    from pathlib import Path
    import numpy as np
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    activate(rt);numerical_policy();torch.set_num_threads(2)
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.t1_practical import Practical
    from experiments.pamf_ssl_research.hard_pl_early_train import source_parents
    from experiments.pamf_ssl_research.hard_pl_joint import validate
    from experiments.pamf_ssl_research.source_geometry import fixed_indices
    from leo_practical.channel import stable_seed
    if (plan['model_seeds']!=[2026092701,2026092702,2026092703] or plan['profiles']!=['ORIGINAL_F21_E20']
        or plan['source_phases']!=[[21,4441],[41,8881]] or plan['roles']!=['U_s','V']
        or plan['augmented_gate_grid']!=[.8,.9,.95] or plan['source_updates']!=0):
        raise ValueError('Fixed original-E20 source-only diagnostic plan required')
    rows=[];start=time.monotonic()
    for seed in plan['model_seeds']:
        selected=[r for r in rt['rows'] if r['model_seed']==seed]
        if len(selected)!=1:raise ValueError('One fixed Joint source template per seed required')
        c=read(selected[0]['train_config']);parents,contract=source_parents(c,validate_config=validate);parent=parents[0]
        payload=torch.load(Path(parent['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        reader=reader_from_payload(dict(c,parent_source=None),payload,'cpu')
        if reader.training or any(p.requires_grad for p in reader.parameters()):raise ValueError('Frozen eval default-head reader required')
        for role in plan['roles']:
            data=SourceDataset(c['source_cache'],role,contract);ix=fixed_indices(data.metadata,role);x=data.x[ix]
            truth=torch.tensor(data.metadata['y'][ix].tolist(),dtype=torch.long) if role=='V' else None
            if len(x)!=(960 if role=='U_s' else 720):raise ValueError('Fixed existing source cohorts required')
            meta={key:torch.tensor(data.metadata[key][ix].tolist(),dtype=torch.long) for key in ('eq_i','rx_i','day_i','sig_i','base_index')}
            def probabilities(values):
                with torch.inference_mode():return torch.cat([reader.classify_features(reader.features(values[b:b+256])).softmax(1) for b in range(0,len(values),256)])
            raw=probabilities(x);rows.append(dict(model_seed=seed,view='raw',source_phase=None,**summarize(raw,raw,role,truth,tuple(plan['augmented_gate_grid']))))
            for epoch,first_step in plan['source_phases']:
                generator=Practical(c);values=[];scenes=[]
                for begin in range(0,len(x),128):
                    step=first_step+begin//128;batch={k:v[begin:begin+128] for k,v in meta.items()}
                    values.append(generator(x[begin:begin+128],batch,epoch,step))
                    namespace='FcrossRX/train/e%d/step%d/L_S'%(epoch,step)
                    for i in range(len(values[-1])):
                        pid=tuple(int(batch[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
                        rng=np.random.default_rng(stable_seed(20261016421,namespace,'scene',pid))
                        scenes.append(str(rng.choice(('practical_low_suburban','practical_mid_urban','practical_low_urban'))))
                if Counter(scenes)!=generator.coverage or sum(generator.coverage.values())!=len(x):raise ValueError('Actual native As scene coverage differs')
                augmented=torch.cat(values);q=probabilities(augmented);strata=[]
                dimensions=[('scene',scenes),('RX',meta['rx_i'].tolist()),('day',meta['day_i'].tolist())]
                if truth is not None:dimensions.append(('TX',truth.tolist()))
                for dimension,keys in dimensions:
                    for key in sorted(set(keys)):
                        mask=torch.tensor([v==key for v in keys],dtype=torch.bool)
                        strata.append(dict(dimension=dimension,stratum=key,**summarize(raw[mask],q[mask],role,None if truth is None else truth[mask],tuple(plan['augmented_gate_grid']))))
                rows.append(dict(model_seed=seed,view='native_As_mix',source_phase=[epoch,first_step],scene_coverage=dict(generator.coverage),
                    strata=strata,**summarize(raw,q,role,truth,tuple(plan['augmented_gate_grid']))))
            del data
        del reader,payload;gc.collect()
    return dict(status='VERIFIED_FROZEN_SOURCE_VIEW_RELIABILITY',run_id=rt['run_id'],analysis_commit=analysis_commit,
        observed_unix=time.time(),seconds=time.monotonic()-start,plan=plan,rows=rows,
        model_updates=0,normalization_updates=0,EMA_updates=0,prototype_updates=0,head_fit_updates=0,
        GPU_workers=0,U_truth_reads=0,target_reads=0,G_training_reads=0,
        interpretation='Fixed originalE20 source-only view learnability.Raw pseudo-label truth correctness is unchanged by simulating the same physical TX;augmented default-head difficulty is a different quantity.U truth stays hidden,no target benefit or physical identity-recovery claim.No healthy recipe/checkpoint/queue is changed.')
