"""Prepare a fixed identity-only pilot; never load features, scores or weights."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path, PurePosixPath

from run_d92_registration_diagnostic import (ROOT, CHANNEL, DIAGNOSTIC_CONFIG,
    CONFIG_NAMES, IDENTITY_KEYS, KS, NEW_COUNTS, require, validate_selection, validate_spec, read)

RUN = '20260930-phase2-d92-registration-diagnostic-m2-r01'
RELEASE = 'd92_registration_diagnostic_20260930_r01'
SPEC = 'configs/d92_registration_diagnostic_20260930.json'
PARENT_SPEC = 'configs/d92_branch_local_ridge_support_20260929.json'
MANIFEST_KEYS = {'schema','capsule_id','checkpoint_sha256','splits'}
SPLIT_KEYS = (IDENTITY_KEYS - {'new_count'}) | {'support_indices','support_ids','support_labels'}


def select_identities(manifest, cohort, checkpoint_digests, support_seed):
    """Validate the complete producer identity matrix before selecting 40 cells."""
    require(set(manifest) == MANIFEST_KEYS and manifest['schema']=='d92_branch_support_features_v1', 'Support identity manifest schema mismatch')
    require(manifest['capsule_id']==cohort['capsule_id'] and manifest['checkpoint_sha256'] in checkpoint_digests, 'Support identity manifest lineage mismatch')
    matrix=cohort['matrix']; pairs=sorted(itertools.product(matrix['receivers'],matrix['scenarios']))[:2]
    expected=set(itertools.product(matrix['receivers'],matrix['scenarios'],matrix['ks'],matrix['new_counts'],matrix['support_seeds']))
    require(len(manifest['splits'])==cohort['expected_split_count']==len(expected), 'Incomplete producer identity matrix')
    seen=set(); ids=set(); selected=[]; old_mapping=None; old_physical={}
    for row in manifest['splits']:
        require(set(row)==SPLIT_KEYS, 'Support split identity schema mismatch')
        require(isinstance(row['split_id'],str) and row['split_id'] and row['split_id'] not in ids, 'Duplicate split ID')
        classes=row['registered_classes']; count=len(classes)-6
        require(isinstance(classes,list) and all(isinstance(c,str) and c for c in classes)
            and len(classes)==len(set(classes)) and count in NEW_COUNTS, 'Invalid registered class mapping')
        old=classes[:6]
        if old_mapping is None: old_mapping=old
        require(old==old_mapping, 'Producer old-class mapping drift')
        require(type(row['k']) is int and type(row['support_seed']) is int, 'Invalid numeric split identity')
        cell=(row['receiver'],row['scenario'],row['k'],count,row['support_seed'])
        require(cell in expected and cell not in seen, 'Missing/duplicate producer matrix cell')
        physical=row['support_ids']; labels=row['support_labels']; indices=row['support_indices']; k=row['k']
        require(len(physical)==len(labels)==len(indices)==len(classes)*k, 'Physical support count mismatch')
        require(all(isinstance(v,str) and v for v in physical) and len(set(physical))==len(physical), 'Invalid physical support IDs')
        require(all(type(v) is int and v>=0 for v in indices) and len(set(indices))==len(indices), 'Invalid physical support indices')
        require(all(type(v) is int and 0<=v<len(classes) for v in labels)
            and all(labels.count(c)==k for c in range(len(classes))), 'Physical K/class label mismatch')
        if (row['receiver'],row['scenario']) in pairs and row['support_seed']==support_seed:
            old_ids={pid:classes[label] for pid,label in zip(physical,labels) if label<6}
            key=(row['receiver'],row['scenario'],k,row['support_seed'])
            require(old_physical.setdefault(key,old_ids)==old_ids, 'Selected old physical support differs across new counts')
            selected.append({**{name:deepcopy(row[name]) for name in IDENTITY_KEYS-{'new_count'}},'new_count':count})
        seen.add(cell);ids.add(row['split_id'])
    require(seen==expected, 'Incomplete producer identity coverage')
    selected.sort(key=lambda x:(x['receiver'],x['scenario'],x['k'],x['new_count'],x['support_seed'],x['split_id']))
    selection=dict(receiver_scenes=[list(p) for p in pairs],support_seed=support_seed,
        ks=KS.copy(),new_counts=NEW_COUNTS.copy(),splits=selected)
    validate_selection(selection,matrix)
    return selection


def documents(parent, identity_manifests, *, commit):
    """Pure transformation. The caller owns writing/registering returned documents."""
    require(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit), 'Explicit preparation commit required')
    require(set(identity_manifests)=={'rx3','rx1'} and set(parent['probe']['cohorts'])=={'rx3','rx1'}, 'Both identity manifests required')
    available=sorted({r['seeds']['model'] for r in parent['rows']})
    require(len(available)>=2 and all(type(s) is int for s in available), 'At least two fixed source models required')
    rows=parent['rows']; pairs=[(r['cohort'],r['seeds']['model']) for r in rows]
    require(len(pairs)==len(set(pairs)) and set(pairs)==set(itertools.product(('rx3','rx1'),available)), 'Incomplete source model/cohort matrix')
    common=set.intersection(*(set(c['matrix']['support_seeds']) for c in parent['probe']['cohorts'].values()))
    require(bool(common) and all(type(s) is int for s in common), 'No valid common support seed')
    seed=min(common); chosen=available[:2]; spec=deepcopy(parent)
    remote=str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release=str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    spec.update(run_id=RUN,group_id='d92-local-ridge-registration-diagnostic',
        display_name='LocalRidge旧类重拟合与新增类别竞争support机制诊断',
        description='固定原LocalRidge；B0旧类拟合、C0全注册拟合、固定C0分数旧列离线限制；确定性160 parent pilot。',
        parent_run_ids=[parent['run_id']],status='PLANNED',tags=['d92','support-only','registration-diagnostic','practical-residual'],
        authorization='User authorized support-only mechanism diagnostic; root is sole launch owner.',
        notes=['No new candidate, optimizer, adapter or performance gate.',
            'Selected practical_high and practical_low_urban on first receiver per cohort; practical_mid and remaining receivers excluded.',
            'Two smallest original model seeds and smallest common support seed selected from identities before results.',
            'True K1 numerical-only; OOF pools held records per parent; proxy anchors averaged within parent.'])
    spec['code'].update(cwd=release,commit=commit)
    spec['execution'].update(remote_run_root=remote,remote_log_root=remote,cpu_lanes=2,blas_threads_per_lane=2,
        export_device=None,export_batch_size=None,gpu_policy='CPU only; two lanes with two BLAS threads each; no encoder/checkpoint loading',
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_registration_diagnostic.py --spec {SPEC}')
    spec['permissions'].update(regime='fixed_local_ridge_support_registration_diagnostic',
        truth_use='Held legal support labels only after fixed B0/C0/C_old predictions; no query access')
    spec['checkpoint'].update(selection_rule='Two smallest fixed source model seeds, final200, selected from metadata before results',
        current_auxiliary_training='Original frozen LocalRidge B0/C0 heads only; no optimizer or inherited target-trained state',runtime_checkpoint_reload=False)
    spec['data'].update(split_id='Explicit selected 40 support splits per cohort; complete original producer matrix retained',
        support_seeds=[seed],channel=deepcopy(CHANNEL))
    spec['metrics_plan'].update(metric_names=['B0_old_accuracy','C_old_old_accuracy','C0_old_accuracy','C0_new_accuracy',
        'signed_old_head_change','additional_new_competition_loss','six_correctness_transitions','resource_cost'],
        interpretation='Descriptive support mechanism diagnostic only; no performance gate or method choice; complete pilot required before summary.',
        prediction_ref='Held legal support predictions only; no query',scorer_ref=None)
    for key in ('controls','candidate','max_factorizations','algorithm_config'):
        spec['probe'].pop(key,None)
    spec['probe'].update(algorithm=deepcopy(DIAGNOSTIC_CONFIG),channel=deepcopy(CHANNEL),
        available_model_seeds=available,selected_model_seeds=chosen,model_rows=4,total_episodes=160,
        expected_parents=160,expected_true_k1_parents=40,expected_oof_parents=120,
        expected_proxy_anchors=1400,expected_sequence_paths=1760,expected_head_fits=3168,
        query_access=False,reuse_support_cache=True,diagnostics=['physical_oof','support_oneshot_proxy'])
    outputs={}
    for name,co in spec['probe']['cohorts'].items():
        digests={r['expected_checkpoint_sha256'] for r in rows if r['cohort']==name}
        co['selection']=select_identities(identity_manifests[name],co,digests,seed)
        co.update(selected_split_count=40,evaluation_config=release+'/'+CONFIG_NAMES[name])
        outputs[CONFIG_NAMES[name]]=dict(algorithm=deepcopy(DIAGNOSTIC_CONFIG),producer_matrix=deepcopy(co['matrix']),selection=deepcopy(co['selection']))
    spec['rows']=[deepcopy(r) for r in rows if r['seeds']['model'] in chosen]
    for row in spec['rows']:
        out=remote+'/'+row['row_id'];co=spec['probe']['cohorts'][row['cohort']]
        row['seeds']['support']=seed
        row.update(method=DIAGNOSTIC_CONFIG['method'],purpose='Identify old-head change versus new-class competition on held support',
            output_root=out,log_path=out+'/probe.log',config_ref=SPEC,resolved_config_ref=out+'/probe/startup.json',
            scenario=','.join(p[1] for p in co['selection']['receiver_scenes'])+'; residual/post_sync/noeq',
            optimizer='None; original frozen LocalRidge head fits only',budget_ref='40 parents;10 K1 numerical-only;30 OOF;350 proxy anchors;440 paths;792 head fits',
            seed_notes='Original model/split/data/augmentation lineage retained; explicit selected support seed; deterministic evaluation.',
            command=f'{spec["code"]["environment"]} -u {release}/tools/run_d92_registration_diagnostic.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
    validate_spec(spec);outputs[SPEC]=spec
    return outputs


def parse_manifests(values):
    parsed={}
    for value in values:
        name,sep,path=value.partition('=')
        require(sep and name in ('rx3','rx1') and name not in parsed and path, 'Use --identity-manifests rx3=path rx1=path')
        parsed[name]=read(Path(path))
    require(set(parsed)=={'rx3','rx1'}, 'Both cohort identity manifests required')
    return parsed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity-manifests',nargs=2,required=True)
    p.add_argument('--parent-spec',type=Path,default=ROOT/PARENT_SPEC);p.add_argument('--commit',required=True)
    a=p.parse_args();out=documents(read(a.parent_spec),parse_manifests(a.identity_manifests),commit=a.commit)
    if any((ROOT/name).exists() for name in out):raise FileExistsError('Preparation exists; reconcile without overwrite')
    for name,value in out.items():
        with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',files=list(out))))
