"""LoRa and Orthogonal core transplants from the preceding satellite B states."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import atomic, read, require, write

POINTS = [[k,n] for k in (5,10) for n in (5,10)]
PROFILES = ('protonet_cda','mrior_sda','twostage','poster_ft','tweak_m1')


def parent_gpu_drained(state):
    return state.get('pending')==0 and state.get('active')==[]


def orthogonal_anchor(saved,profile,old_x,old_y):
    import torch
    from . import group2_cil_bridge as cil
    if profile=='protonet_cda':
        # MoPC's affine bridge uses different features from the original C bridge.
        original=cil.from_protonet(saved['model'],saved['old_prototypes'])
        orth=cil.make_anchor(original,old_x,old_y,method='orthogonal')
    else:
        orth=copy.deepcopy(saved['anchor']);orth.method='orthogonal'
    require(orth.prototypes.shape==orth.model.fc.weight.shape, 'Orthogonal support dimensions differ')
    orth.model.classifier_weight=torch.nn.Parameter(orth.prototypes.detach().clone(),requires_grad=False)
    orth.protocol_metadata.update(orthogonal_bridge='support_prototypes_v1',
         C_old_weights='B_encoder_old_support_prototypes',B_old_decision_preserved_at_C_entry=False)
    return orth


def make_plan(parent_config, root, release):
    parent = read(parent_config); root=Path(root).resolve(); release=Path(release).resolve()
    require(not root.exists() and release.is_dir() and parent['entry_module']=='satellite_mix3',
            'Wrong parent or existing output')
    jobs=[]
    for old in parent['jobs']:
        if old.get('group')!='hybrid': continue
        require(old['profile'] in PROFILES and old['points']==POINTS and old['methods']==['mopc_hr']
                and old.get('save_B_for_mopc') is True, 'Wrong parent B task')
        ident=old['id'].replace('-hybrid-', '-savedb-c-')
        jobs.append(dict(id=ident, profile=old['profile'], seed=old['seed'], capsule=old['capsule'],
                         parent=old, output=str(root/'predictions'/ident), score=str(root/'scores'/ident)))
    require(len(jobs)==945 and len({j['id'] for j in jobs})==945, 'Wrong saved B matrix')
    return dict(run_root=str(root), release_root=str(release), parent_config=str(Path(parent_config).resolve()),
                parent_run=parent['run_root'], runtime_file=parent['runtimes']['hybrid'],
                allowed_gpus=list(range(8)), max_per_gpu=6, minimum_free_MiB=6000,
                worker_reserve_MiB=3000, jobs=jobs)


def verify_parent(job):
    folder=Path(job['parent']['output']); done=read(folder/'complete.json'); idx=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False
            and done.get('query_fit') is False and done.get('profile')==job['profile']
            and done.get('model_seed')==job['seed'] and idx.get('capsule')==job['capsule']
            and idx.get('model_seed')==job['seed'] and done.get('rows')==4,
            'Parent B completion or identity differs')
    require({(r['k'],r['new_count']) for r in idx['rows']}=={tuple(p) for p in POINTS}
            and all(r.get('input_condition')=='satellite_received' and r.get('stage')=='C'
                    and (folder/r['prediction']).is_file() for r in idx['rows'])
            and all((folder/f'B-K{k}.pt').is_file() for k in (5,10)), 'Parent B rows absent')
    return folder


def run_worker(config,job,*,smoke_only=False):
    import numpy as np
    import torch
    from . import group2_cil_bridge as cil, group2_lora_bridge as lora
    from .episodic_experiment import split_rows
    from .external_execution import seed_for
    from .fast_transfer import to_tensor
    from .group1 import row
    from .group1_source import embedding
    from .group2_b_state import load_b_state
    from .group2_c_backbone_worker import support_only, score_indexed
    from .io import activate
    parent=verify_parent(job);rt=read(config['runtime_file'])
    require(not rt.get('clean_root') and rt.get('input_condition')=='satellite_received','Wrong satellite input')
    activate(rt);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cpu' if smoke_only else rt.get('device','cuda:0')
    cap=Path(rt['data_root'])/job['capsule']; manifest=read(cap/'manifest.json')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE',
            'Wrong capsule protocol')
    lookup={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    out=Path(job['output']);rows=[]
    if not smoke_only:out.mkdir(parents=True,exist_ok=False)
    bank=None if smoke_only else np.load(cap/'received.npy',mmap_mode='r',allow_pickle=False)
    for k in (5,10):
        require((k,0) in lookup and all((k,n) in lookup for n in (5,10)), 'Missing split')
        old_iq,old_labels,old_ids,_=support_only(cap,lookup[k,0]['split_id'])
        old_x=to_tensor(old_iq,device,np.float32)
        old_y=to_tensor(old_labels,device,np.int64).long()
        bseed=seed_for(job['seed'],job['capsule'],k,'B')
        saved=load_b_state(parent/f'B-K{k}.pt',job['parent'],k,lookup[k,0]['split_id'],bseed,device)
        model=saved['model'].eval()
        if job['profile']=='tweak_m1':
            from .external_additional import native_features
            extractor=native_features
        else:
            method=row(job['profile'])['method']
            extractor=lambda model,x: embedding(model,method,x)
        gallery=lora.build_gallery(model,extractor,old_x,old_y,old_count=6)
        orth=orthogonal_anchor(saved,job['profile'],old_x,old_y)
        with torch.no_grad():
            observed=orth.model(old_x)
            require(torch.isfinite(observed['features']).all() and torch.isfinite(observed['tx_logits']).all(),
                    'Saved B support smoke failed')
        for n in (5,10):
            spec=lookup[k,n]
            support,labels,ids,_=support_only(cap,spec['split_id']);mask=labels<6
            require(np.array_equal(old_ids,ids[mask]),'B/C support differs')
            new_x=to_tensor(support[~mask],device,np.float32)
            new_y=to_tensor(labels[~mask],device,np.int64).long()
            options={'epochs':1} if smoke_only else {}
            states={'lora':lora.register_lora(gallery,new_x,new_y),
                    'orthogonal':cil.register(orth,new_x,new_y,seed=seed_for(job['seed'],job['capsule'],k,n,'C'),**options)}
            if smoke_only:
                for state in states.values():
                    scores=state.score(new_x[:1])
                    require(tuple(scores.shape)==(1,6+n) and bool(torch.isfinite(scores).all()),
                            'Support-only registered model smoke failed')
                print('SUPPORT_ONLY_SMOKE_PASS',job['profile'],job['capsule'],flush=True)
                return
            for method,state in states.items():
                query=split_rows(cap,spec)
                pred,restricted=score_indexed(state,bank,query,count=6+n,old_count=6,
                    batch_size=1 if job['profile']=='tweak_m1' else rt.get('batch_size',128),device=device)
                name=f'C-K{k}-N{n}-{method}.npz'
                np.savez_compressed(out/name,query_indices=query,
                                    predicted_indices=pred,old_restricted_indices=restricted)
                suffix='lora_balanced_v1' if method=='lora' else 'orthogonal_support_prototypes_v1'
                rows.append(dict(split_id=spec['split_id'],stage='C',arm=job['profile']+'__'+suffix,
                    input_path='S0Q0' if job['profile']=='tweak_m1' else 'P0',input_condition='satellite_received',
                    prediction=name,k=k,new_count=n,registered_classes=spec['registered_classes'],query_count=len(pred),
                    query_records_per_decision=1 if job['profile']=='tweak_m1' else None,
                    claim='DA_backbone_with_CIL_core_transplant_not_native_source_architecture',
                    resources=dict(B_state_ref=str(parent/f'B-K{k}.pt'),B_reused=True,
                      C_protocol=getattr(state,'protocol_metadata',{}))))
                print('SAVED_B_C_PREDICTED',job['profile'],method,k,n,len(pred),flush=True)
    require(len(rows)==8,'Incomplete C predictions')
    write(out/'predictions.json',dict(capsule_id=manifest['capsule_id'],capsule=job['capsule'],
          profile=job['profile'],model_seed=job['seed'],rows=rows))
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],
          profile=job['profile'],model_seed=job['seed'],rows=len(rows),truth_read=False,query_fit=False))


def verify_prediction(job):
    folder=Path(job['output']);done=read(folder/'complete.json');idx=read(folder/'predictions.json')
    require(done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False
            and done.get('query_fit') is False and done.get('profile')==job['profile']
            and done.get('model_seed')==job['seed'] and idx.get('capsule')==job['capsule'],
            'Wrong saved B C completion')
    expected={(k,n,job['profile']+'__'+method) for k,n in POINTS
              for method in ('lora_balanced_v1','orthogonal_support_prototypes_v1')}
    rows=idx['rows'];actual=[(r['k'],r['new_count'],r['arm']) for r in rows]
    require(len(rows)==len(expected)==done['rows'] and set(actual)==expected and
            all(r['stage']=='C' and r.get('input_condition')=='satellite_received'
                and (folder/r['prediction']).is_file() and r['resources']['B_reused']
                for r in rows), 'Wrong saved B C rows')
    if job['profile']=='tweak_m1':
        require(all(r.get('query_records_per_decision')==1 for r in rows),'Tweak query must be single-record')


def run_score(config,job):
    from .episodic_score import run
    verify_prediction(job)
    run(read(config['runtime_file']),dict(prediction=job['output'],output=job['score']))


def dispatch(config,config_file):
    from .clean_scheduler import choose
    from .runner import occupancy
    root=Path(config['run_root']); require(not root.exists(),'Run root exists')
    root.mkdir(parents=True);control=root/'scheduler';control.mkdir();(control/'logs').mkdir()
    write(control/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    pending=list(config['jobs']);active={};queue=[];scorers={};done=[];scored=[];failed=[];excluded=[];unscored=[];stopped=False
    while pending or active or queue or scorers:
        if (control/'USER_STOP.json').is_file():stopped=True;pending.clear()
        for key,item in list(active.items()):
            code=item['proc'].poll()
            if code is None:continue
            item['log'].close();job=item['job'];del active[key]
            try:
                if code==0:verify_prediction(job);done.append(key);queue.append(job);continue
            except (OSError,ValueError,KeyError,json.JSONDecodeError):pass
            failed.append(dict(id=key,stage='prediction',exit_code=code));pending.clear()
        for key,item in list(scorers.items()):
            code=item['proc'].poll()
            if code is None:continue
            item['log'].close();del scorers[key]
            try:ok=code==0 and read(Path(item['job']['score'])/'complete.json').get('status')=='SCORED_COMPLETE'
            except (OSError,ValueError,KeyError,json.JSONDecodeError):ok=False
            if ok:scored.append(key)
            else:failed.append(dict(id=key,stage='score',exit_code=code));pending.clear()
        while queue and len(scorers)<2 and not failed:
            job=queue.pop(0);log=None
            try:
                log=(control/'logs'/(job['id']+'-score.log')).open('x')
                proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.satellite_saved_b_c','score',
                    '--config',config_file,'--job-id',job['id']],cwd=config['release_root'],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                scorers[job['id']]=dict(job=job,proc=proc,log=log)
            except OSError as exc:
                if log is not None:log.close()
                failed.append(dict(id=job['id'],stage='score_launch',error=str(exc)));pending.clear()
        ready=[]
        try:
            parent_state=read(Path(config['parent_run'])/'scheduler'/'state.json')
            # The old dispatcher cannot share an atomic GPU admission lock.
            # Only start after it has no queued or active GPU work. Scoring may continue.
            drained=parent_gpu_drained(parent_state)
            if drained:
                for job in list(pending):
                    marker=Path(job['parent']['output'])/'numerical_failure.json'
                    if marker.is_file():
                        data=read(marker)
                        require(data.get('status')=='MOPC_NUMERICAL_FAILURE' and
                                all(data.get(k)==job.get(k) for k in ('profile','seed','capsule')),
                                'Bad numerical marker')
                        excluded.append(job['id']);pending.remove(job)
                ready=[j for j in pending if (Path(j['parent']['output'])/'complete.json').is_file()]
                require(len(ready)==len(pending),'Parent drained with missing B states')
        except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
            failed.append(dict(stage='parent',error=str(exc)));pending.clear()
        if ready and not failed and not stopped:
            entries=[dict(id=j['id']) for j in ready]
            try:
                admissions=choose(entries,{k:dict(pid=v['proc'].pid,gpu=v['gpu']) for k,v in active.items()},
                    occupancy(),config['allowed_gpus'],config['minimum_free_MiB'],config['max_per_gpu'],
                    worker_reserve=config['worker_reserve_MiB'],one_per_snapshot=True)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                failed.append(dict(stage='occupancy',error=str(exc)));pending.clear();admissions=[]
            for entry,gpu in admissions:
                job=next(j for j in ready if j['id']==entry['id'])
                log=None
                try:
                    log=(control/'logs'/(job['id']+'.log')).open('x')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.satellite_saved_b_c','worker',
                        '--config',config_file,'--job-id',job['id']],cwd=config['release_root'],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                except OSError as exc:
                    if log is not None:log.close()
                    failed.append(dict(id=job['id'],stage='worker_launch',error=str(exc)));pending.clear();break
                active[job['id']]=dict(job=job,proc=proc,log=log,gpu=gpu);pending.remove(job)
        if failed and queue:
            unscored.extend(j['id'] for j in queue);queue.clear()
        atomic(control/'state.json',dict(status='FAILED' if failed else 'STOPPED' if stopped else 'RUNNING',
            completed=len(done),scored=len(scored),pending=len(pending),ready=len(ready),
            active=[dict(id=k,pid=v['proc'].pid,gpu=v['gpu']) for k,v in active.items()],
            score_pending=len(queue),score_active=list(scorers),unscored=unscored,excluded=excluded,failed=failed))
        if pending or active or queue or scorers:time.sleep(5)
    if failed:raise RuntimeError('Failure retained')
    if not stopped:write(control/'complete.json',dict(status='COMPLETED_WITH_EXCLUSIONS' if excluded else 'SCORED_COMPLETE',
                         scored=len(scored),excluded=excluded,truth_last=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','smoke','dispatch','worker','score'))
    p.add_argument('--config',required=True);p.add_argument('--parent-config')
    p.add_argument('--run-root');p.add_argument('--release-root');p.add_argument('--job-id');a=p.parse_args()
    if a.action=='prepare':
        path=Path(a.config).resolve();require(not path.exists() and not path.parent.exists(),'Existing configuration')
        config=make_plan(a.parent_config,a.run_root,a.release_root)
        path.parent.mkdir(parents=True);write(path,config);print(json.dumps(dict(status='PREPARED',jobs=len(config['jobs']))));return
    config=read(a.config)
    if a.action=='smoke':
        for profile in PROFILES:
            choices=[j for j in config['jobs'] if j['profile']==profile and
                     (Path(j['parent']['output'])/'complete.json').is_file()]
            require(choices,'No completed B for support-only smoke: '+profile)
            run_worker(config,choices[0],smoke_only=True)
        return
    if a.action=='dispatch':dispatch(config,a.config);return
    selected=[j for j in config['jobs'] if j['id']==a.job_id];require(len(selected)==1,'Unknown job')
    (run_worker if a.action=='worker' else run_score)(config,selected[0])


if __name__=='__main__':main()
