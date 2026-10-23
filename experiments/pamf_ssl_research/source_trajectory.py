"""Compare only preregistered frozen source checkpoints;never fit or select a model."""


def collect(contexts,plan,phase,analysis_commit):
    from pathlib import Path
    import gc
    import time
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    rt=contexts['joint'];activate(rt);numerical_policy()
    import torch
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_ssl_research.hard_pl_early_train import source_parents
    from experiments.pamf_ssl_research.hard_pl_train import source_parents as old_parents
    from experiments.pamf_ssl_research.hard_pl import validate as old_validate
    from experiments.pamf_ssl_research.hard_pl_early import validate as early_validate
    from experiments.pamf_ssl_research.hard_pl_guard import validate as guard_validate
    from experiments.pamf_ssl_research.hard_pl_joint import validate as joint_validate
    from experiments.pamf_ssl_research.source_geometry import LEVELS,fixed_indices,capture_deployed,describe
    torch.set_num_threads(2)
    def config(key,method,seed):
        rows=[r for r in contexts[key]['rows'] if r['method']==method and r['model_seed']==seed]
        if len(rows)!=1:raise ValueError('Exactly one registered same-seed source row required')
        return read(rows[0]['train_config'])
    profiles=plan['profiles_by_phase'][phase];seeds=plan['model_seeds']
    # Metadata readiness precedes checkpoint loads;waiting does not restart or alter live jobs.
    for seed in seeds:
        for profile in profiles:
            if profile in ('F21_E20','F21_E30'):continue
            key,method={'EARLY_NAMS_E30':('early','HPL_E20_NAMS2'),'GUARD_NAMS_E30':('guard','HPL_E20_NAMS6G'),
                        'JOINT_E30':('joint','HPL_E20_NAMS_JOINT6G')}[profile]
            c=config(key,method,seed);path=Path(c['output_root'])/'segment30/complete.json'
            if not path.is_file() or not read(path).get('actual_adapter_SSL'):
                return dict(status='WAIT_FIXED_SOURCE_STAGE',profile=profile,model_seed=seed,target_reads=0,U_truth_reads=0)
    def load(profile,seed,expected):
        key,method,validator=('old','HPL2',old_validate) if profile=='F21_E30' else ('joint','HPL_E20_NAMS_JOINT6G',joint_validate) if profile in ('F21_E20','JOINT_E30') else ('early','HPL_E20_NAMS2',early_validate) if profile=='EARLY_NAMS_E30' else ('guard','HPL_E20_NAMS6G',guard_validate)
        c=config(key,method,seed)
        parents,contract=(old_parents(c,validate_config=validator) if profile=='F21_E30' else source_parents(c,validate_config=validator))
        if contract!=expected:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if profile in ('F21_E20','F21_E30'):
            done=parents[-1];folder=Path(done['folder']);cfg=done['config']
        else:
            folder=Path(c['output_root'])/'segment30';done=read(folder/'complete.json');cfg=done['config'];parent=parents[-1]['folder']
            if (done['status']!='SOURCE_FROZEN' or done['epoch']!=10 or done['updates']!=2220
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
                or cfg['run_id']!=c['run_id'] or cfg['row_id']!=c['row_id'] or cfg['model_seed']!=seed
                or cfg['epoch_offset']!=20 or cfg['step_offset']!=4440 or cfg['parent_source']!=parent
                or read(cfg['source_contract'])!=expected or done['actual_adapter_SSL']['hard_PL_calls']!=2220):
                raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:fixed own E20+10 source stage')
        payload=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (payload['run_id']!=cfg['run_id'] or payload['candidate_id']!=cfg['row_id'] or payload['epoch']!=done['epoch']
            or payload['a1_ema_successful_updates']!=done['updates'] or payload['args']['teacher_ckpt']
            or payload['args']['use_ema_teacher'] or payload['args']['use_unlabeled']
            or payload['args']['baseline_ckpt']!=('' if cfg.get('parent_source') is None else cfg['parent_source']+'/final_ssdg.pth')):
            raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:actual fixed source payload')
        return reader_from_payload(dict(cfg,parent_source=None),payload,'cpu'),cfg
    rows=[];start=time.monotonic()
    for seed in seeds:
        base_c=config('joint','HPL_E20_NAMS_JOINT6G',seed);parents,contract=source_parents(base_c,validate_config=joint_validate)
        reference,_=load('F21_E20',seed,contract);data=SourceDataset(base_c['source_cache'],'L_s',contract)
        features,logits=capture_deployed(reference,data.x[:2])
        if logits.shape!=(2,6):raise ValueError('Actual source-only checkpoint/default-head smoke failed')
        del data
        cohorts={}
        for role in ('U_s','V'):
            data=SourceDataset(base_c['source_cache'],role,contract);ix=fixed_indices(data.metadata,role)
            x=data.x[ix];rx=torch.tensor(data.metadata['rx_i'][ix].tolist(),dtype=torch.long)
            q=[]
            for begin in range(0,len(x),256):
                _,logits=capture_deployed(reference,x[begin:begin+256]);q.append(logits.softmax(1))
            q=torch.cat(q);accepted=q.max(1).values>=.95 if role=='U_s' else torch.ones(len(x),dtype=torch.bool)
            labels=q.argmax(1) if role=='U_s' else torch.tensor(data.metadata['y'][ix].tolist(),dtype=torch.long)
            cohorts[role]=dict(x=x,rx=rx,labels=labels,accepted=accepted,reference_q=q);del data
        del reference;gc.collect()
        for profile in profiles:
            reader,c=load(profile,seed,contract)
            for role,cohort in cohorts.items():
                values={name:[] for name in LEVELS};probabilities=[];x=cohort['x']
                for begin in range(0,len(x),256):
                    features,logits=capture_deployed(reader,x[begin:begin+256]);probabilities.append(logits.softmax(1))
                    for name in LEVELS:values[name].append(features[name])
                features={name:torch.cat(parts) for name,parts in values.items()};q=torch.cat(probabilities)
                y=cohort['labels'];accepted=cohort['accepted'];pred=q.argmax(1);confidence=q.max(1).values
                rows.append(dict(profile=profile,model_seed=seed,role=role,source_epoch=20 if profile=='F21_E20' else 30,
                    source_run_id=c['run_id'],source_row_id=c['row_id'],source_code_commit=c.get('code_commit'),
                    samples=len(x),common_reference_selected=int(accepted.sum()),own_acceptance=int((confidence>=.95).sum()),
                    common_selected_prediction_agreement_or_V_correct=int(((pred==y)&accepted).sum()),
                    mean_common_reference_label_CE=float(-q[accepted,y[accepted]].clamp_min(1e-12).log().mean()) if accepted.any() else None,
                    labels='Fixed originalE20 raw pseudo TX;U truth hidden' if role=='U_s' else 'Fixed sourceV true TX,diagnostic only,all720',
                    **describe(features,y,cohort['rx'],accepted)))
            del reader;gc.collect()
        del cohorts
    return dict(status='VERIFIED_SOURCE_GEOMETRY_TRAJECTORY_PHASE_'+phase,analysis_code_commit=analysis_commit,
        run_context=rt['run_id'],phase=phase,profiles=profiles,model_seeds=seeds,rows=rows,seconds=time.monotonic()-start,
        U_samples_per_seed=960,V_samples_per_seed=720,U_reference='Same-seed originalF21E20 raw eval argmax/confidence.95,fixed for every comparator',
        GPU_workers=0,model_updates=0,normalization_updates=0,EMA_updates=0,prototype_updates=0,head_fit_updates=0,
        target_reads=0,U_truth_reads=0,live_jobs_changed=False,source_V_role_changed=False,
        interpretation='Raw deployed eval features,without run-time style perturbation;fixed source cohort and U reference mask/classes across profiles.Two descriptive normalizations and actual additive branch norms;not fitted nuisance identification,U actual accuracy,statistical significance,target success or a selection rule.')
