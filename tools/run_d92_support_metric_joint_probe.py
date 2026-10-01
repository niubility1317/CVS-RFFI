"""Explicit-row SupportMetric support supervisor; root owns the sole launch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi.d92_support_metric_joint_local_ridge import FROZEN_CONFIG
from run_d92_margin_joint_probe import validate_selection, selected_tasks, KS, NEW_COUNTS, CHANNEL, SCENARIOS

SCHEMA = 'd92_support_metric_joint_support_probe_v1'
METHOD = FROZEN_CONFIG['method']
STATUS = 'SUPPORT_METRIC_JOINT_PROBE_COMPLETE'
STARTED = 'SUPPORT_METRIC_JOINT_SUPERVISOR_STARTED'
PROBE_CONFIG = deepcopy(FROZEN_CONFIG)
CPU_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1')
RESOURCE_KEYS = {'max_newton_iterations','max_line_search_trials','max_factor_buffer_bytes',
                 'max_integer_bits','max_fraction_operations','max_secular_iterations'}
MODEL_SEEDS = {2026092701,2026092702}
COHORTS = {'rx3','rx1'}


def require(condition, message):
    if not condition: raise ValueError(message)


def oid(value):
    return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    # State snapshots are the only intentionally replaceable run metadata.
    path=Path(path)
    with path.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def exclusive_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def validate_resources(resources):
    require(isinstance(resources,dict) and set(resources)==RESOURCE_KEYS
        and all(type(v) is int and v>0 for v in resources.values()), 'Explicit positive integer SupportMetric resources required')
    require(resources['max_secular_iterations']<=128,'Fixed bounded secular execution ceiling exceeded')
    return resources


def evaluator_config(spec,cohort):
    co=spec['probe']['cohorts'][cohort]
    return dict(algorithm=PROBE_CONFIG,producer_matrix=co['matrix'],selection=co['selection'],
                support_metric_resources=spec['probe']['support_metric_resources'])


def selection_budget(selection,resources):
    """Exact physical stage counts; adapter ceilings are not measured Newton costs."""
    validate_selection(selection);validate_resources(resources)
    exact=dict(episodes=len(selection['splits']),k1_episodes=0,oof_episodes=0,proxy_anchor_count=0,
        sequence_paths=0,baseline_head_fit_count=0,candidate_preparation_count=0,candidate_stage_count=0,
        final_candidate_head_fit_count=0,row_basis_construction_count=1,basis_calls=1,
        basis_binding_calls=0,basis_binding_physical_gram_evaluation_count=0)
    maximum=dict(optimizer_steps=0,trial_count=0,accepted_trial_count=0,rejected_trial_count=0,
        ggn_step_count=0,ggn_parameter_direction_count=0,support_metric_step_calls=0,
        basis_fraction_operations_attempted=resources['max_fraction_operations'],
        support_metric_step_fisher_construction_attempts=0,
        support_metric_step_secular_iteration_count=0,
        support_metric_step_secular_evaluation_count=0,
        support_metric_step_factorization_attempts=0)
    for split in selection['splits']:
        k=split['k'];stages=1+int(split['new_count']>0)
        if k==1: exact['k1_episodes']+=1;train_ks=[1]
        else:
            exact['oof_episodes']+=1;exact['proxy_anchor_count']+=k;folds=min(k,3)
            train_ks=[k-len(range(fold,k,folds)) for fold in range(folds)]+[1]*k
        for train_k in train_ks:
            exact['sequence_paths']+=1
            for key in ('baseline_head_fit_count','candidate_preparation_count','candidate_stage_count','final_candidate_head_fit_count'):
                exact[key]+=stages
            exact['basis_binding_calls']+=stages
            exact['basis_binding_physical_gram_evaluation_count']+=stages
            informative=stages if train_k>1 else 0
            maximum['optimizer_steps']+=informative
            maximum['accepted_trial_count']+=informative
            maximum['ggn_step_count']+=informative
            maximum['support_metric_step_calls']+=informative
            maximum['support_metric_step_fisher_construction_attempts']+=informative
            maximum['support_metric_step_secular_iteration_count']+=resources['max_secular_iterations']*informative
            maximum['support_metric_step_secular_evaluation_count']+=(resources['max_secular_iterations']+2)*informative
            maximum['support_metric_step_factorization_attempts']+=(resources['max_secular_iterations']+3)*informative
            maximum['ggn_parameter_direction_count']+=5*informative
            maximum['trial_count']+=12*informative
            maximum['rejected_trial_count']+=12*informative
    return dict(exact=exact,maximum=maximum)


def budget_for_spec(spec):
    per={r['row_id']:selection_budget(spec['probe']['cohorts'][r['cohort']]['selection'],
        spec['probe']['support_metric_resources']) for r in spec['rows']}
    total={kind:{key:sum(v[kind][key] for v in per.values()) for key in next(iter(per.values()))[kind]}
           for kind in ('exact','maximum')}
    return dict(per_row=per,total=total)


def validate_spec(spec):
    rows=spec['rows'];probe=spec['probe'];execution=spec['execution'];code=spec['code']
    require(spec['schema']==SCHEMA and spec['group_id']=='d92-support-metric-joint-support','New independent schema/group required')
    validate_resources(probe['support_metric_resources'])
    require(isinstance(spec['run_id'],str) and spec['run_id'] and isinstance(rows,list) and rows,'Explicit run/rows required')
    require(len({r['row_id'] for r in rows})==len(rows),'Duplicate row ID')
    require(len(rows)==4 and set(probe['cohorts'])==COHORTS,'Fixed two-model by two-cohort four-row matrix required')
    sp=PurePosixPath(spec['spec_path'])
    require(not sp.is_absolute() and '..' not in sp.parts and '\\' not in spec['spec_path'] and sp.suffix=='.json','Relative spec path required')
    require(oid(code['commit']) and code.get('preparation_parent_commit')==code['commit'],'Explicit preparation source parent required')
    require(probe['algorithm']==PROBE_CONFIG and probe['channel']==CHANNEL and probe['candidate']=='R_SUPPORT_METRIC_seq'
        and probe['controls']==['R0'] and probe['query_access'] is False and probe['reuse_support_cache'] is True
        and probe['interpretation']=='support_joint_pilot_no_direct_promotion','Frozen method/access mismatch')
    permissions=spec['permissions']
    require(permissions['query_use']=='none; query IQ/labels/truth/scores never read'
        and all(permissions.get(k) is False for k in ('source_samples','source_per_record_features','old_target_scores_for_adaptation','cross_run_result_tuning'))
        and permissions['adapted_state_reuse']=='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse','Input permission mismatch')
    require(permissions.get('summary_inputs') is True and permissions.get('prototype_use')=='frozen_center_feature_dictionary_only; no_teacher_targets_or_extra_support','Fixed reference geometry permission mismatch')
    require(execution['cpu_lanes']==2 and execution['blas_threads_per_lane']==2 and execution['launch_owner']=='root','Root CPU2 BLAS2 contract required')
    require(set(probe['cohorts'])=={r['cohort'] for r in rows},'Unused/missing cohort')
    release=PurePosixPath(code['cwd']);root=PurePosixPath(execution['remote_run_root'])
    require(release.is_absolute() and root.is_absolute() and release!=root and release not in root.parents and root not in release.parents,'Release/run overlap')
    for name,co in probe['cohorts'].items():
        matrix=co['matrix'];validate_selection(co['selection'])
        require(set(matrix)=={'receivers','scenarios','ks','new_counts','support_seeds'},'Producer axes mismatch')
        for vals in matrix.values():require(isinstance(vals,list) and vals and len(vals)==len(set(vals)),'Invalid producer axis')
        require(matrix['ks']==KS and matrix['new_counts']==NEW_COUNTS and set(matrix['scenarios'])==set(SCENARIOS),'Fixed producer axes mismatch')
        require(co['expected_split_count']==math.prod(len(v) for v in matrix.values()),'Producer count mismatch')
        require(co['selection']['support_seed'] in matrix['support_seeds'] and all(rx in matrix['receivers'] and sc in matrix['scenarios']
            for rx,sc in co['selection']['receiver_scenes']),'Selection escaped producer')
        require(co['selected_split_count']==len(co['selection']['splits'])==40 and co['evaluator_config']==evaluator_config(spec,name),'Fixed forty-parent embedded literal evaluator config mismatch')
        require(co['capsule_id'].startswith('residual-noeq-'),'Residual capsule required')
    pairs=set();models={}
    for row in rows:
        name=row['row_id'];require(isinstance(name,str) and name and '/' not in name and '\\' not in name and name not in ('.','..'),'Invalid row ID')
        co=probe['cohorts'][row['cohort']]
        require(PurePosixPath(row['output_root'])==root/name,'Row output escaped run')
        require(row['method']==METHOD and row['initial_step_size']==PROBE_CONFIG['initial_step'] and row['lr'] is None,'Frozen optimizer metadata mismatch')
        seeds=row['seeds'];require(set(seeds)=={'model','split','data','augmentation','support','evaluation'}
            and all(v is None or type(v) is int for v in seeds.values()) and type(seeds['model']) is int
            and seeds['support']==co['selection']['support_seed'],'Six explicit seed roles required')
        digest=row['expected_checkpoint_sha256'];require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Checkpoint SHA missing')
        ground=row.get('ground_summary')
        require(isinstance(ground,str) and bool(ground) and type(row.get('ground_summary_already_deployed')) is bool,'Explicit existing ground component/deployment state required')
        packet=row.get('ground_packet')
        require(packet is None or isinstance(packet,str) and bool(packet),'Invalid optional Ground packet path')
        require(models.setdefault(seeds['model'],(digest,packet,ground,row['ground_summary_already_deployed']))==(digest,packet,ground,row['ground_summary_already_deployed']),'Same model checkpoint/ground packet differs')
        pair=(row['cohort'],seeds['model']);require(pair not in pairs,'Duplicate model/cohort row');pairs.add(pair)
        for source in (row['support_features'],co['capsule'],ground)+(() if packet is None else (packet,)):
            p=PurePosixPath(source);require(p.is_absolute() and p!=root and p not in root.parents and root not in p.parents,'Source/output overlap')
    require(pairs=={(co,seed) for co in COHORTS for seed in MODEL_SEEDS},'Fixed model/cohort crossproduct differs')
    require(probe['budget']==budget_for_spec(spec) and probe['budget']['total']['exact']['episodes']==160,'Declared 160-parent structural budget mismatch')
    return spec


def command(spec,row,commit):
    require(oid(commit),'Explicit actual runtime commit required')
    co=spec['probe']['cohorts'][row['cohort']];out=PurePosixPath(row['output_root'])
    argv=[sys.executable,'-u',str(PurePosixPath(spec['code']['cwd'])/'tools/evaluate_d92_support_metric_joint_probe.py'),
        '--support-features',row['support_features'],'--capsule',co['capsule'],'--output',str(out/'probe'),
        '--config',str(out/'resolved_config.json'),'--expected-capsule-id',co['capsule_id'],
        '--expected-checkpoint-sha256',row['expected_checkpoint_sha256'],'--expected-model-seed',str(row['seeds']['model']),
        '--run-id',spec['run_id'],'--row-id',row['row_id'],'--release-commit',commit]
    if row.get('ground_packet'):argv+=['--ground-packet',row['ground_packet']]
    argv+=['--ground-summary',row['ground_summary'],'--ground-summary-already-deployed','true' if row['ground_summary_already_deployed'] else 'false']
    return argv


def launch(argv,log,cwd,started):
    with Path(log).open('x',encoding='utf-8') as stream:
        p=subprocess.Popen(argv,cwd=cwd,env=dict(os.environ,**CPU_ENV),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        started(p.pid);code=p.wait()
    return dict(pid=p.pid,returncode=code)


def verify_marker(path,spec,row,*,commit,expected_pid):
    from evaluate_d92_support_metric_joint_probe import COUNTERS,PEAK_COUNTERS,WORK_SUM_KEYS,WORK_MAX_KEYS
    marker=read(path);out=Path(path).parent;startup=read(out/'startup.json')
    co=spec['probe']['cohorts'][row['cohort']];budget=spec['probe']['budget']['per_row'][row['row_id']]
    expected=dict(schema=SCHEMA,method=METHOD,run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],release_commit=commit,
        query_rows_used=0,source_rows_used=0,truth_read=False,support_metric_resources=spec['probe']['support_metric_resources'])
    require(type(expected_pid) is int and expected_pid>0 and all(marker.get(k)==v and startup.get(k)==v for k,v in expected.items()),'Marker/startup identity mismatch')
    require(startup['pid']==expected_pid and marker['pid']==expected_pid and startup['config']==evaluator_config(spec,row['cohort']),'Actual row PID/resolved config mismatch')
    require(marker['status']==STATUS and marker['workload_complete'] is True and marker['algorithm']==PROBE_CONFIG
        and marker['selection']==co['selection'] and marker['producer_matrix']==co['matrix'],'Incomplete declared method/selection')
    require(startup.get('ground_packet')==row.get('ground_packet') and marker.get('ground_packet')==row.get('ground_packet'),'Ground packet identity differs')
    require(startup.get('ground_summary')==marker.get('ground_summary')==row['ground_summary'] and startup.get('ground_summary_already_deployed')==marker.get('ground_summary_already_deployed')==row['ground_summary_already_deployed'],'Fixed reference geometry/deployment identity differs')
    plan=read(Path(row['support_features'])/'support_splits.json')
    require(plan['capsule_id']==co['capsule_id'] and plan['checkpoint_sha256']==row['expected_checkpoint_sha256'],'Physical support plan binding differs')
    lookup={v['split_id']:v for v in plan['splits']}
    require(len(lookup)==len(plan['splits']),'Duplicate source physical split')
    physical={v['split_id']:lookup[v['split_id']]['support_ids'] for v in co['selection']['splits']}
    require(marker.get('selected_support_physical_ids')==startup.get('selected_support_physical_ids')==physical,'Actual selected support physical coverage differs')
    require(all(startup['blas_environment'].get(k)=='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
        and startup['cuda_visible_devices']=='','CPU/BLAS execution differs')
    for key in COUNTERS:
        value=marker.get(key)
        require((type(value) in (int,float) and math.isfinite(value) if key.endswith('_seconds') else type(value) is int)
            and value>=0,'Invalid actual counter: '+key)
    for key,count in budget['exact'].items():require(marker[key]==count,'Incomplete structural coverage: '+key)
    for key,count in budget['maximum'].items():require(marker[key]<=count,'Fixed adapter ceiling exceeded: '+key)
    require(marker['optimizer_steps']==marker['accepted_trial_count'] and
        marker['trial_count']==marker['accepted_trial_count']+marker['rejected_trial_count'],'Actual acceptance ledger differs')
    rank=marker.get('row_basis_rank')
    require(type(rank) is int and 0<=rank<=5,'Explicit stored-dictionary physical rank required')
    require(marker['peak_effective_adapter_rank']==rank,'Actual row physical rank peak differs')
    expected_steps=budget['maximum']['support_metric_step_calls'] if rank>0 else 0
    require(marker['ggn_step_count']==marker['support_metric_step_calls']==expected_steps and
        marker['ggn_parameter_direction_count']==rank*marker['support_metric_step_calls'],
        'Actual physical G/F step and direction counts differ')
    require(isinstance(marker.get('actual_work'),dict) and marker.get('work_aggregation')=='SUM/MAX',
        'Measured core work and SUM/MAX scope required')
    work_keys=set(WORK_SUM_KEYS)|set(WORK_MAX_KEYS)
    parts=[marker.get(name) for name in ('row_basis_actual_work','preparation_actual_work','stage_actual_work','score_actual_work')]
    require(set(marker['actual_work'])==work_keys and all(isinstance(part,dict) and set(part)==work_keys for part in parts),
        'Complete row-basis/preparation/stage/score work fields required')
    require(marker.get('actual_work_aggregation')=={key:'MAX' if key in WORK_MAX_KEYS else 'SUM' for key in work_keys},
        'Actual work SUM/MAX declarations differ')
    for part in parts:
        for key,value in part.items():
            require((type(value) in (int,float) and math.isfinite(value) if key.endswith('_seconds') else type(value) is int)
                and value>=0,'Invalid actual phase work: '+key)
    for key in work_keys:
        expected_work=max(part[key] for part in parts) if key in WORK_MAX_KEYS else sum(part[key] for part in parts)
        actual=marker['actual_work'][key]
        if key.endswith('_seconds'):
            tolerance=128*math.ulp(max(1.,abs(expected_work),abs(actual)))
            require(abs(actual-expected_work)<=tolerance and abs(marker[key]-actual)<=tolerance,
                'Measured work phase sum differs: '+key)
        else:require(actual==expected_work==marker[key],'Measured work phase aggregation differs: '+key)
    for key in PEAK_COUNTERS:
        value=marker.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid measured peak: '+key)
        if key.endswith('peak_factor_buffer_bytes'):
            require(value<=spec['probe']['support_metric_resources']['max_factor_buffer_bytes'],'Owned factor buffer exceeds limit; not RSS')
    resources=spec['probe']['support_metric_resources']
    for key in ('basis_max_observed_integer_bits','basis_max_intermediate_bit_bound'):
        require(marker[key]<=resources['max_integer_bits'],'Owned integer execution guard exceeded')
    row_work=marker['row_basis_actual_work'];other_work=parts[1:]
    require(row_work['basis_calls']==1 and all(part['basis_calls']==0 for part in other_work),
        'Stored-dictionary basis must be constructed once at row owner only')
    require(row_work['basis_binding_calls']==row_work['basis_binding_physical_gram_evaluation_count']==0 and
        marker['preparation_actual_work']['basis_binding_calls']==budget['exact']['candidate_preparation_count'] and
        marker['preparation_actual_work']['basis_binding_physical_gram_evaluation_count']==budget['exact']['candidate_preparation_count'] and
        all(part['basis_binding_calls']==part['basis_binding_physical_gram_evaluation_count']==0 for part in parts[2:]),
        'Actual path basis binding/Gram preparation charges differ')
    for part in other_work:
        require(all(part[key]==0 for key in work_keys if key.startswith('basis_') and not key.startswith('basis_binding_')),
            'Prebuilt basis construction work was recharged outside row owner')
    require(marker['support_metric_step_fisher_construction_attempts']==marker['support_metric_step_fisher_constructions_completed']==marker['support_metric_step_calls'],
        'One actual prediction Fisher per completed metric step required')
    require(marker['support_metric_step_spectral_check_attempts']==marker['support_metric_step_spectral_checks_completed']==2*marker['support_metric_step_calls'],
        'Actual physical Gram/GGN PSD diagnostics differ')
    require(marker['support_metric_step_factorization_attempts']==marker['support_metric_step_factorizations_completed'] and
        marker['support_metric_step_triangular_calls']==marker['support_metric_step_triangular_calls_completed'],
        'Completed metric-step linear attempts are incomplete')
    # Read support trace metadata only. Numerical arrays and scores are not used by the supervisor.
    records=[json.loads(line) for line in (out/'compact.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    selected={s['split_id']:s for s in co['selection']['splits']}
    require(len(records)==len(selected) and len({r['split_id'] for r in records})==len(selected),'Duplicate/missing physical parent marker')
    for r in records:
        s=selected.get(r['split_id']);require(s is not None and all(r[k]==s[k] for k in ('receiver','scenario','k','support_seed','new_count'))
            and r['query_rows_used']==r['source_rows_used']==0,'Physical parent coverage differs')
        require(r.get('row_basis_ref')==marker.get('row_basis_ref') and r.get('row_basis_rank')==rank,
            'Parent path changed its row-owned physical basis binding')
    manifest=read(out/'state_manifest.json');require(manifest['status']=='COMPLETE' and manifest['schema']=='d92_support_metric_joint_state_archive_v1'
        and manifest['method']==METHOD and manifest['file_count']==marker['state_archive_file_count'],'State archive incomplete')
    ref=marker.get('row_basis_ref')
    require(isinstance(ref,dict) and ref in manifest.get('files',[]) and ref.get('failed_numeric_state') is False,
        'Row-owned basis numeric state is not in the complete ordinary archive')
    context=dict(run_id=spec['run_id'],row_id=row['row_id'],split_id=None,
        scope='row_frozen_ground_geometry',fold=None,trial=None,parent_k=None,train_k=None,
        state='ROW_SUPPORT_METRIC_BASIS')
    require(ref.get('key')=='basis' and isinstance(ref.get('namespace'),str) and
        json.loads(ref['namespace'])==context,'Row basis archive owner namespace differs')
    files=manifest.get('files')
    require(isinstance(files,list) and len(files)==manifest['file_count'] and
        len({item.get('path') for item in files})==len(files),'Ordinary state file inventory differs')
    for item in files:
        relative=item.get('path');p=PurePosixPath(relative) if isinstance(relative,str) else None
        require(p is not None and not p.is_absolute() and '..' not in p.parts and '\\' not in relative,
            'State archive file escaped row output')
        state_path=out/relative
        require(state_path.is_file() and type(item.get('file_bytes')) is int and
            state_path.stat().st_size==item['file_bytes'],'Actual ordinary state file missing or truncated')
    certificate=marker.get('basis_certificate')
    require(isinstance(certificate,dict) and certificate.get('path')=='basis_certificate.json' and
        certificate.get('scope')=='ORDINARY_METHOD_ARTIFACT_NOT_AUTHORIZATION','Ordinary basis certificate descriptor differs')
    payload=(out/'basis_certificate.json').read_bytes();text=payload.decode('utf-8',errors='strict')
    require(type(certificate.get('file_bytes')) is int and type(certificate.get('utf8_bytes')) is int and
        certificate['file_bytes']==certificate['utf8_bytes']==len(payload) and len(text.encode('utf-8'))==len(payload),
        'Actual basis certificate byte count differs')
    exact=json.loads(text)
    require(exact.get('schema')=='d92_support_metric_basis_v1' and exact.get('shape')==[160,5] and
        exact.get('rank')==rank and exact.get('input_scope')=='EXACT_STORED_BINARY64_NOT_PRE_ROUNDING_PROTOTYPES',
        'Stored-dictionary basis certificate metadata differs')
    basis_audit=marker.get('row_basis_audit')
    require(isinstance(basis_audit,dict) and basis_audit.get('actual_work')==row_work and
        basis_audit.get('schema')=='d92_support_metric_row_basis_v1' and
        basis_audit.get('method')==METHOD and basis_audit.get('status')=='COMPLETE' and
        basis_audit.get('row_basis_ref')==ref and basis_audit.get('row_basis_rank')==rank and
        basis_audit.get('basis_certificate')==certificate and basis_audit.get('context')==context and
        basis_audit.get('row_basis_construction_count')==1 and
        basis_audit.get('input_role')=='FROZEN_GROUND_Q_ONLY_NO_SUPPORT_TEACHER',
        'Row basis construction ledger descriptor differs')
    require(marker.get('row_basis')=='row_basis.json' and read(out/'row_basis.json')==basis_audit,
        'Actual row basis owner artifact differs')
    construction=basis_audit.get('construction_audit')
    require(isinstance(construction,dict) and construction.get('status')=='COMPLETE' and
        construction.get('exact_rank')==rank and all(construction.get(key)==resources[key]
            for key in ('max_integer_bits','max_fraction_operations')),
        'Actual basis construction resource/rank audit differs')
    return marker


def run(spec,commit,launch_fn=launch,preflight_fn=None):
    from evaluate_d92_support_metric_joint_probe import COUNTERS,PEAK_COUNTERS
    from preflight_d92_support_metric_joint_probe import inspect_metadata
    validate_spec(spec);require(oid(commit),'Actual runtime commit missing')
    root=Path(spec['execution']['remote_run_root']);require(not root.exists(),'Exclusive run already exists; reconcile')
    # All row input permissions/metadata are resolved before any row fit. Failed rows stay failed.
    checks=(preflight_fn or inspect_metadata)(spec,require_exclusive=False)
    root.mkdir(parents=True,exist_ok=False);rows=spec['rows'];state={r['row_id']:dict(status='PENDING') for r in rows};lock=threading.Lock()
    startup=dict(status=STARTED,schema=SCHEMA,method=METHOD,run_id=spec['run_id'],group_id=spec['group_id'],runtime_commit=commit,
        code_commit=spec['code']['commit'],resolved_spec=spec,pid=os.getpid(),argv=sys.argv,python=sys.executable,
        environment=CPU_ENV,input_preflight=checks,query_access=False,truth_read=False,source_sample_access=False,scorer_invoked=False,
        started=time.time(),cpu_lanes=2,blas_threads_per_lane=2)
    exclusive_json(root/'startup.json',startup)
    def update(row,status,**fields):
        with lock:state[row['row_id']]=dict(status=status,updated=time.time(),**fields);write(root/'state.json',state)
        print(json.dumps(dict(row_id=row['row_id'],status=status,**fields),allow_nan=False),flush=True)
    byrow={v['row_id']:v for v in checks['cache_bindings']}
    require(set(byrow)==set(state),'Metadata preflight row declarations differ')
    def work(row):
        argv=command(spec,row,commit);pid=None
        try:
            require(byrow[row['row_id']]['binding']=='VERIFIED','Input metadata failed: '+str(byrow[row['row_id']]))
            out=Path(row['output_root']);out.mkdir(exist_ok=False)
            exclusive_json(out/'resolved_config.json',evaluator_config(spec,row['cohort']))
            require(read(out/'resolved_config.json')==evaluator_config(spec,row['cohort']),'Resolved config readback differs')
            def started(actual_pid):
                nonlocal pid
                require(type(actual_pid) is int and actual_pid>0,'Actual row PID missing');pid=actual_pid
                exclusive_json(out/'row_startup.json',dict(run_id=spec['run_id'],row_id=row['row_id'],runtime_commit=commit,
                    pid=pid,argv=argv,cwd=spec['code']['cwd'],environment=CPU_ENV,model_seed=row['seeds']['model'],
                    checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id=spec['probe']['cohorts'][row['cohort']]['capsule_id']))
                update(row,'TRAINING_ON_SUPPORT',pid=pid,actual_command=argv,environment=CPU_ENV)
            result=launch_fn(argv,out/'probe.log',Path(spec['code']['cwd']),started)
            marker=verify_marker(out/'probe/probe_complete.json',spec,row,commit=commit,expected_pid=pid)
            require(result.get('pid')==pid and result.get('returncode')==0,'Row process ended unsuccessfully despite marker')
            update(row,STATUS,pid=pid,actual_command=argv,release_commit=commit,
                actual_work=marker['actual_work'],work_aggregation='SUM/MAX',
                **{k:marker[k] for k in COUNTERS})
        except Exception as exc:
            update(row,'FAILED',pid=pid,actual_command=argv,error_type=type(exc).__name__,error=str(exc),automatic_retry=False,
                partial_workload=None,partial_workload_status='SEE_PRESERVED_ROW_FAILURE_AND_STATE_ARCHIVE',
                failure_artifact=str(Path(row['output_root'])/'probe/probe_failed.json'))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,rows))
    success=all(v['status']==STATUS for v in state.values());completed=[v for v in state.values() if v['status']==STATUS]
    marker=dict(status=STATUS if success else 'FAILED',schema=SCHEMA,method=METHOD,run_id=spec['run_id'],group_id=spec['group_id'],
        runtime_commit=commit,code_commit=spec['code']['commit'],pid=os.getpid(),rows=state,model_rows=len(rows),completed_rows=len(completed),
        aggregate_scope='completed_rows_only; partial_failed_work_in_row_artifacts',workload_complete=success,
        actual_work_by_row={name:v['actual_work'] for name,v in state.items() if v['status']==STATUS},
        query_access=False,truth_read=False,source_sample_access=False,scorer_invoked=False,automatic_retry=False,finished=time.time(),
        **{k:max((v[k] for v in completed),default=0) if k in PEAK_COUNTERS else sum(v[k] for v in completed) for k in COUNTERS})
    exclusive_json(root/'complete.json',marker)
    if not success:raise RuntimeError('SupportMetric support row failed; healthy rows retained; no retry')
    return marker


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(read(a.spec),a.commit)
