"""Explicit-row CPU support supervisor; no global four-row budget assumption."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
from cvsrffi.d92_conditional_joint_local_ridge import FROZEN_CONFIG
from run_d92_branch_support_probe import read, write, launch

PROBE_CONFIG=deepcopy(FROZEN_CONFIG)
SCHEMA='d92_conditional_joint_local_ridge_v1'
METHOD='D92-ConditionalJointLocalRidge-v1'
STATUS='CONDITIONAL_JOINT_PROBE_COMPLETE'
CHANNEL=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000)
SCENARIOS=['practical_high','practical_mid','practical_low_urban']
KS=[1,5,10,20]
NEW_COUNTS=[0,2,5,10,20]
IDENTITY_FIELDS=('split_id','receiver','scenario','k','support_seed','registered_classes','new_count')


def require(condition,message):
    if not condition: raise ValueError(message)


def validate_selection(selection):
    require(set(selection)=={'receiver_scenes','support_seed','ks','new_counts','splits'},'Selection schema mismatch')
    pairs=selection['receiver_scenes']
    require(isinstance(pairs,list) and pairs and all(isinstance(p,(list,tuple)) and len(p)==2
        and all(isinstance(v,str) and v for v in p) and p[1] in SCENARIOS for p in pairs)
        and len({tuple(p) for p in pairs})==len(pairs),'Explicit unique receiver-scene pairs required')
    seed=selection['support_seed']
    require(type(seed) is int and seed>=0 and selection['ks']==KS and selection['new_counts']==NEW_COUNTS,'Fixed K/new axes mismatch')
    expected={(rx,scene,k,n,seed) for rx,scene in pairs for k in KS for n in NEW_COUNTS}
    cells=set();ids=set();old=None
    for item in selection['splits']:
        require(set(item)==set(IDENTITY_FIELDS),'Selected identity fields mismatch')
        require(all(type(item[k]) is int for k in ('k','new_count','support_seed')),'Invalid numeric split identity')
        cell=tuple(item[k] for k in ('receiver','scenario','k','new_count','support_seed'))
        classes=item['registered_classes'];sid=item['split_id']
        require(isinstance(sid,str) and sid and sid not in ids and cell in expected and cell not in cells,'Duplicate/unexpected selection')
        require(isinstance(classes,list) and len(classes)==6+item['new_count'] and all(isinstance(v,str) and v for v in classes)
            and len(set(classes))==len(classes),'Six-old registered class contract mismatch')
        if old is None: old=classes[:6]
        require(classes[:6]==old,'Old class registry differs across selected parents')
        cells.add(cell);ids.add(sid)
    require(cells==expected,'Incomplete declared K by new-count matrix')
    return ids


def selected_tasks(tasks,selection,old):
    wanted=validate_selection(selection)
    lookup={s['split_id']:(s,positions,labels) for s,positions,labels in tasks}
    require(len(lookup)==len(tasks) and wanted<=set(lookup),'Missing or duplicate producer split')
    chosen=[];old_maps={}
    for identity in selection['splits']:
        task=lookup[identity['split_id']];split=task[0]
        projected={k:deepcopy(split[k]) for k in IDENTITY_FIELDS if k!='new_count'}
        projected['new_count']=len(split['registered_classes'])-len(old)
        require(projected==identity and set(old)<=set(split['registered_classes']),'Selected/producer identity mismatch')
        mapping={pid:split['registered_classes'][y] for pid,y in zip(split['support_ids'],split['support_labels'])}
        old_map={pid:label for pid,label in mapping.items() if label in old}
        key=tuple(split[k] for k in ('receiver','scenario','k','support_seed'))
        require(old_maps.setdefault(key,old_map)==old_map,'Paired old physical support changed across new counts')
        chosen.append(task)
    return chosen


def selection_budget(selection):
    """Bounds derive only from declared physical folds and fixed 4x12 budget."""
    validate_selection(selection)
    exact=dict(episodes=len(selection['splits']),k1_episodes=0,oof_episodes=0,proxy_anchor_count=0,
        sequence_paths=0,baseline_head_fit_count=0,candidate_preparation_count=0,candidate_stage_count=0,diagnostic_fit_count=0)
    maximum=dict(trained_conditional_stage_count=0,optimizer_steps=0,optimizer_iterations=0,
        trial_count=0,trial_attempt_count=0,inner_objective_evaluation_count=0,inner_head_fit_count=0,
        inner_factorization_count=0,final_head_fit_count=0,final_factorization_count=0,
        prior_head_fit_count=0,prior_factorization_count=0,baseline_factorization_count=0,
        head_fit_count=0,factorization_count=0,projection_factorization_count=0,
        residual_factorization_count=0,spectral_diagnostic_count=0)
    for split in selection['splits']:
        k=split['k'];has_new=split['new_count']>0;stages=1+int(has_new)
        if k==1: exact['k1_episodes']+=1;train_ks=[1]
        else:
            exact['oof_episodes']+=1;exact['proxy_anchor_count']+=k
            folds=min(k,3)
            train_ks=[k-len(range(fold,k,folds)) for fold in range(folds)]+[1]*k
        for train_k in train_ks:
            exact['sequence_paths']+=1
            for key in ('baseline_head_fit_count','candidate_preparation_count','candidate_stage_count'):exact[key]+=stages
            inner=min(train_k,3) if train_k>1 else 0
            informative=stages if inner else 0
            maximum['trained_conditional_stage_count']+=informative
            maximum['optimizer_steps']+=4*informative;maximum['optimizer_iterations']+=4*informative
            maximum['trial_count']+=48*informative;maximum['trial_attempt_count']+=48*informative
            maximum['inner_objective_evaluation_count']+=49*informative
            maximum['inner_head_fit_count']+=49*inner*stages
            maximum['inner_factorization_count']+=49*inner*(1+2*int(has_new))
            maximum['final_head_fit_count']+=stages;maximum['final_factorization_count']+=1+2*int(has_new)
            maximum['prior_head_fit_count']+=inner*int(has_new);maximum['prior_factorization_count']+=inner*int(has_new)
            conditional=(49*inner+1)*int(has_new)
            maximum['projection_factorization_count']+=conditional
            maximum['residual_factorization_count']+=(49*inner+1)*stages
            maximum['spectral_diagnostic_count']+=3*conditional
    maximum['baseline_factorization_count']=exact['baseline_head_fit_count']
    maximum['head_fit_count']=sum(maximum[k] for k in ('inner_head_fit_count','final_head_fit_count','prior_head_fit_count'))+exact['baseline_head_fit_count']
    maximum['factorization_count']=sum(maximum[k] for k in ('baseline_factorization_count','inner_factorization_count','final_factorization_count','prior_factorization_count'))
    return dict(exact=exact,maximum=maximum)


def budget_for_spec(spec):
    per_row={r['row_id']:selection_budget(spec['probe']['cohorts'][r['cohort']]['selection']) for r in spec['rows']}
    total={kind:{key:sum(value[kind][key] for value in per_row.values()) for key in next(iter(per_row.values()))[kind]}
           for kind in ('exact','maximum')}
    return dict(per_row=per_row,total=total)


def validate_spec(spec):
    rows=spec['rows'];probe=spec['probe'];execution=spec['execution'];code=spec['code']
    require(isinstance(spec['run_id'],str) and spec['run_id'] and isinstance(rows,list) and rows,'Explicit run and nonempty rows required')
    require(len({r['row_id'] for r in rows})==len(rows),'Duplicate row ID')
    spec_path=PurePosixPath(spec['spec_path'])
    require(not spec_path.is_absolute() and '..' not in spec_path.parts and '\\' not in spec['spec_path'] and spec_path.suffix=='.json','Relative spec_path required')
    require(probe['algorithm']==PROBE_CONFIG and probe['channel']==CHANNEL and probe['candidate']=='R_CONDITIONAL_seq'
        and probe['controls']==['R0'] and probe['query_access'] is False and probe['reuse_support_cache'] is True,
        'Frozen conditional structure/access mismatch')
    require(probe['interpretation']=='support_joint_pilot_no_direct_promotion','Unsupported result interpretation')
    permissions=spec['permissions']
    require(permissions['query_use']=='none; query IQ/labels/truth/scores never read'
        and all(permissions.get(k) is False for k in ('source_samples','source_per_record_features','summary_inputs','old_target_scores_for_adaptation','cross_run_result_tuning'))
        and permissions['adapted_state_reuse']=='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse','Input permissions mismatch')
    require(execution['cpu_lanes']==2 and execution['blas_threads_per_lane']==2 and isinstance(execution['launch_owner'],str)
        and execution['launch_owner'],'Explicit owner / CPU2 BLAS2 contract required')
    require(set(probe['cohorts'])=={r['cohort'] for r in rows},'Unused or missing explicit cohort')
    release=PurePosixPath(code['cwd']);root=PurePosixPath(execution['remote_run_root'])
    require(release.is_absolute() and root.is_absolute() and release!=root and release not in root.parents and root not in release.parents,'Release/run overlap')
    config_paths=set()
    for co in probe['cohorts'].values():
        matrix=co['matrix'];validate_selection(co['selection'])
        require(set(matrix)=={'receivers','scenarios','ks','new_counts','support_seeds'},'Producer matrix schema mismatch')
        for values in matrix.values():require(isinstance(values,list) and values and len(values)==len(set(values)),'Invalid producer axis')
        require(matrix['ks']==KS and matrix['new_counts']==NEW_COUNTS and set(matrix['scenarios'])==set(SCENARIOS),'Producer fixed axes mismatch')
        require(co['selection']['support_seed'] in matrix['support_seeds'] and all(rx in matrix['receivers'] and sc in matrix['scenarios']
            for rx,sc in co['selection']['receiver_scenes']),'Selection escaped producer')
        require(co['expected_split_count']==len(list(itertools.product(*matrix.values()))),'Producer split count mismatch')
        require(isinstance(co['capsule_id'],str) and co['capsule_id'].startswith('residual-noeq-'),'Residual capsule identity required')
        cfg=PurePosixPath(co['config_path'])
        require(not cfg.is_absolute() and '..' not in cfg.parts and '\\' not in co['config_path'] and cfg.suffix=='.json'
            and co['evaluation_config']==str(release/cfg),'Evaluator config escaped release')
        require(str(cfg) not in config_paths and cfg!=spec_path,'Overlapping evaluator/spec paths')
        config_paths.add(str(cfg))
    identities=set();checkpoint_by_model={}
    for row in rows:
        name=row['row_id'];require(isinstance(name,str) and name and '/' not in name and '\\' not in name and name not in ('.','..'),'Invalid row ID')
        co=probe['cohorts'][row['cohort']]
        require(PurePosixPath(row['output_root'])==root/name,'Row output escaped run')
        require(row['initial_step_size']==.125 and row['lr'] is None and row['method']==METHOD,'Optimizer metadata mismatch')
        seeds=row['seeds'];require(set(seeds)=={'model','split','data','augmentation','support','evaluation'}
            and all(v is None or type(v) is int for v in seeds.values()) and type(seeds['model']) is int
            and seeds['support']==co['selection']['support_seed'],'Six explicit seed roles required')
        digest=row['expected_checkpoint_sha256'];require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Checkpoint identity missing')
        require(checkpoint_by_model.setdefault(seeds['model'],digest)==digest,'Same model seed checkpoint identity mismatch')
        pair=(row['cohort'],seeds['model']);require(pair not in identities,'Duplicate model/cohort row');identities.add(pair)
        for source in (row['support_features'],co['capsule']):
            path=PurePosixPath(source);require(path.is_absolute() and path!=root and path not in root.parents and root not in path.parents,'Source/output overlap')
    require(spec['probe']['budget']==budget_for_spec(spec),'Explicit-row structural budget mismatch')
    return spec


def command(spec,row):
    co=spec['probe']['cohorts'][row['cohort']]
    return [sys.executable,'-u',str(Path(spec['code']['cwd'])/'tools/evaluate_d92_conditional_joint_probe.py'),
        '--support-features',row['support_features'],'--capsule',co['capsule'],'--output',str(Path(row['output_root'])/'probe'),
        '--config',co['evaluation_config'],'--expected-capsule-id',co['capsule_id'],'--expected-checkpoint-sha256',row['expected_checkpoint_sha256'],
        '--expected-model-seed',str(row['seeds']['model']),'--run-id',spec['run_id'],'--row-id',row['row_id']]


def verify_marker(path,spec,row):
    from evaluate_d92_conditional_joint_probe import COUNTERS
    marker=read(path);co=spec['probe']['cohorts'][row['cohort']];budget=spec['probe']['budget']['per_row'][row['row_id']]
    expected=dict(status=STATUS,schema=SCHEMA,method=METHOD,run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],algorithm=PROBE_CONFIG,
        selection=co['selection'],producer_matrix=co['matrix'],query_rows_used=0,source_rows_used=0,truth_read=False)
    require(all(marker.get(k)==v for k,v in expected.items()),'Marker identity/access mismatch')
    for key in COUNTERS:require(type(marker.get(key)) is int and marker[key]>=0,'Invalid actual counter: '+key)
    for key,count in budget['exact'].items():require(marker[key]==count,'Incomplete physical coverage: '+key)
    for key,count in budget['maximum'].items():
        if key in marker:require(marker[key]<=count,'Fixed structural upper bound exceeded: '+key)
    require(marker['head_fit_count']==sum(marker[k] for k in ('baseline_head_fit_count','inner_head_fit_count','final_head_fit_count','prior_head_fit_count')),'Head total mismatch')
    require(marker['factorization_count']==sum(marker[k] for k in ('baseline_factorization_count','inner_factorization_count','final_factorization_count','prior_factorization_count')),'Factor total mismatch')
    require(marker['optimizer_steps']==marker['accepted_trial_count'] and marker['trial_count']==marker['trial_attempt_count']
        and marker['trial_count']==marker['accepted_trial_count']+marker['rejected_trial_count'],'Trial/update totals mismatch')
    require(marker['optimizer_steps']<=4*marker['trained_conditional_stage_count'],'Accepted update budget mismatch')
    require(marker['candidate_preparation_count']==marker['ajlr_preparation_count'] and marker['candidate_stage_count']==marker['ajlr_stage_count'],
        'Actual candidate/core stage counters differ')
    require(marker['inner_factorization_count']+marker['final_factorization_count']==
        marker['projection_factorization_count']+marker['residual_factorization_count']==marker['completed_factorization_count'],
        'Projection/residual student factor totals mismatch')
    require(marker['spectral_diagnostic_count']==3*marker['projection_factorization_count'],'Conditional three-spectrum work mismatch')
    for prefix in ('projection','residual'):
        require(marker[prefix+'_triangular_solve_count']==2*marker[prefix+'_factorization_count'],
            'Completed forward factor/solve mismatch: '+prefix)
    require(marker['projection_adjoint_factorization_count']==marker['residual_adjoint_factorization_count']==0,
        'Adjoint must reuse the completed forward factors')
    for prefix in ('projection','residual','projection_adjoint','residual_adjoint'):
        calls=marker[prefix+'_triangular_solve_count'];rhs=marker[prefix+'_triangular_rhs_count']
        elements=marker[prefix+'_triangular_rhs_element_count'];dense=marker[prefix+'_triangular_dense_work_unit_count']
        require(calls%2==0 and calls<=rhs<=max(401,27)*calls and rhs<=elements and elements<=dense
            and elements*elements<=rhs*dense,'Invalid actual conditional RHS accounting: '+prefix)
    # Conditional heads can execute two factorizations, not the old single-head inequality.
    for prefix in ('inner','final'):require(marker[prefix+'_factorization_count']<=2*marker[prefix+'_head_fit_count'],'Impossible student factor count')
    require(marker['baseline_factorization_count']<=marker['baseline_head_fit_count'] and marker['prior_factorization_count']<=marker['prior_head_fit_count'],'Impossible baseline/prior factor count')
    return marker


def run(spec,commit,launch_fn=launch):
    from evaluate_d92_conditional_joint_probe import COUNTERS
    validate_spec(spec)
    for co in spec['probe']['cohorts'].values():require(read(co['evaluation_config'])==dict(algorithm=PROBE_CONFIG,producer_matrix=co['matrix'],selection=co['selection']),'Config/spec mismatch')
    root=Path(spec['execution']['remote_run_root']);root.mkdir(parents=True,exist_ok=False)
    rows=spec['rows'];state={r['row_id']:dict(status='PENDING') for r in rows};lock=threading.Lock()
    write(root/'startup.json',dict(spec=spec,run_id=spec['run_id'],commit=commit,pid=os.getpid(),argv=sys.argv,started=time.time(),
        query_access=False,source_sample_access=False,checkpoint_loaded=False,gpu_use=False,actual_A=None,
        cpu_lanes=spec['execution']['cpu_lanes'],blas_threads_per_lane=spec['execution']['blas_threads_per_lane']))
    def update(row,status,**fields):
        with lock:
            state[row['row_id']]=dict(status=status,updated=time.time(),**fields);write(root/'state.json',state)
        print(json.dumps(dict(row_id=row['row_id'],status=status,**fields)),flush=True)
    def work(row):
        try:
            out=Path(row['output_root']);out.mkdir(exist_ok=False);update(row,'TRAINING_ON_SUPPORT')
            launch_fn(command(spec,row),out/'probe.log',Path(spec['code']['cwd']),'probe')
            marker=verify_marker(out/'probe/probe_complete.json',spec,row)
            update(row,STATUS,**{key:marker[key] for key in COUNTERS})
        except Exception as exc:update(row,'FAILED',error_type=type(exc).__name__,error=str(exc))
    with ThreadPoolExecutor(max_workers=spec['execution']['cpu_lanes']) as pool:list(pool.map(work,rows))
    success=all(value['status']==STATUS for value in state.values())
    write(root/'complete.json',dict(status=STATUS if success else 'FAILED',run_id=spec['run_id'],commit=commit,
        model_rows=len(rows),completed_rows=sum(v['status']==STATUS for v in state.values()),
        **{key:sum(v.get(key,0) for v in state.values()) for key in COUNTERS},finished=time.time(),query_access=False,source_sample_access=False))
    if not success:raise RuntimeError('Conditional support lane failed; preserve healthy lanes and all failures; no retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(read(a.spec),a.commit)
