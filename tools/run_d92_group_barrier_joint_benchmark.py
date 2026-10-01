"""Root-owned two-lane prediction supervisor; independent scoring is a later call."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import itertools
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'),str(ROOT/'tools')]
SCHEMA = 'd92_group_barrier_joint_query_benchmark_v1'
STARTED = 'GROUP_BARRIER_QUERY_SUPERVISOR_STARTED'
STATUS = 'GROUP_BARRIER_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
FAILED = 'GROUP_BARRIER_QUERY_BENCHMARK_PREDICTIONS_FAILED'
ROW_FIELDS = {'row_id','cohort','expected_model_seed','expected_checkpoint_sha256',
              'row_root','branch_features','ground_packet','output_root'}
CPU_ENV = {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
           'NUMEXPR_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'}
MODEL_SEEDS={2026092701,2026092702}
RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160)
MATRIX_BASE=dict(scenarios=['practical_high','practical_mid','practical_low_urban'],ks=[1,5,10,20],
                 new_counts=[0,2,5,10,20],support_seeds=[2026092711,2026092712,2026092713,2026092714,2026092715])
RECEIVERS=dict(rx3=['19-1','8-14','8-7'],rx1=['20-19'])
DECISION_SCHEMA='d92_group_barrier_single_query_decision_v1'
DECISION_POLICY='WITHIN_GROUP_RAW_ARGMAX_THEN_GATE_LOGPROB_GAP_LEXICAL_TIE'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key: '+key)
        value[key] = item
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_object)


def write(path, value, *, replace=False):
    with Path(path).open('w' if replace else 'x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def oid(value, length=40):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None


def _path(value):
    return PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)


def absolute(value):
    return isinstance(value,str) and bool(value) and '\\' not in value and '\n' not in value \
        and '\r' not in value and '..' not in _path(value).parts and _path(value).is_absolute()


def _separate(a, b):
    a,b = _path(a),_path(b)
    return a != b and a not in b.parents and b not in a.parents


def validate_spec(spec):
    require(isinstance(spec,dict) and set(spec)=={'schema','run_id','group_id','spec_path','code','execution','benchmark','rows'},
            'Explicit benchmark spec schema required')
    require(spec['schema']==SCHEMA and all(isinstance(spec[k],str) and spec[k] for k in ('run_id','group_id')), 'Run/group schema mismatch')
    relative = PurePosixPath(spec['spec_path'])
    require(not relative.is_absolute() and ':' not in str(relative) and '\\' not in spec['spec_path']
            and '..' not in relative.parts and relative.suffix=='.json', 'Relative new spec path required')
    code,execution,benchmark = spec['code'],spec['execution'],spec['benchmark']
    require(set(code)=={'cwd','environment','commit'} and absolute(code['cwd']) and absolute(code['environment'])
            and oid(code['commit']), 'Explicit release/interpreter/preparation commit required')
    require(set(execution)=={'remote_run_root','launch_owner','cpu_lanes','blas_threads'}
            and execution['launch_owner']=='root' and type(execution['cpu_lanes']) is int and execution['cpu_lanes']==2
            and type(execution['blas_threads']) is int and execution['blas_threads']==2
            and absolute(execution['remote_run_root']), 'Root-owned CPU2 / BLAS2 contract required')
    require(_separate(code['cwd'],execution['remote_run_root']), 'Release/run paths overlap')
    require(set(benchmark)=={'config','cohorts'}, 'Benchmark config/cohorts required')
    from cvsrffi.d92_group_barrier_joint_local_ridge import FROZEN_CONFIG
    config = benchmark['config']
    require(set(config)=={'algorithm','group_barrier_resources'} and config['algorithm']==FROZEN_CONFIG, 'Frozen method config mismatch')
    limits = config['group_barrier_resources']
    require(set(limits)=={'max_newton_iterations','max_line_search_trials','max_factor_buffer_bytes'} and
            all(type(v) is int and v>0 for v in limits.values()), 'Explicit positive integer GroupBarrier limits required')
    require(limits==RESOURCES,'Frozen GroupBarrier technical limits differ')
    cohorts = benchmark['cohorts']
    require(set(cohorts)=={'rx3','rx1'}, 'Both declared cohorts required')
    for name,co in cohorts.items():
        require(set(co)=={'capsule','truth','expected_capsule_id','matrix','expected_split_count'} and absolute(co['capsule']) and absolute(co['truth'])
                and isinstance(co['expected_capsule_id'],str) and co['expected_capsule_id'], 'Explicit cohort identities required')
        require(co['matrix']==dict(MATRIX_BASE,receivers=RECEIVERS[name]) and co['expected_split_count']==(900 if name=='rx3' else 300),
                'Fixed full receiver/scenario/K/new/support-seed matrix differs')
        for key in ('capsule','truth'):
            require(_separate(co[key],execution['remote_run_root']), 'Input/output paths overlap')
    rows = spec['rows']
    require(isinstance(rows,list) and len(rows)==4, 'Fixed four model/cohort prediction rows required')
    seen,pairs,models = set(),set(),{}
    for row in rows:
        require(set(row)==ROW_FIELDS, 'Explicit row/source fields required')
        name = row['row_id']
        require(isinstance(name,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name) is not None and name not in seen, 'Invalid/duplicate row ID')
        seen.add(name)
        require(row['cohort'] in cohorts and type(row['expected_model_seed']) is int and row['expected_model_seed']>=0
                and oid(row['expected_checkpoint_sha256'],64), 'Model/cohort/checkpoint identity required')
        pair = (row['cohort'],row['expected_model_seed'])
        require(pair not in pairs, 'Duplicate model/cohort row'); pairs.add(pair)
        require(_path(row['output_root'])==_path(execution['remote_run_root'])/name, 'Row output escaped exclusive run')
        for key in ('row_root','branch_features','ground_packet'):
            require(absolute(row[key]) and _separate(row[key],execution['remote_run_root']), 'Source/output overlap: '+key)
        model = (row['expected_checkpoint_sha256'],row['ground_packet'])
        require(models.setdefault(row['expected_model_seed'],model)==model, 'Same model seed changed source checkpoint/packet')
    require(pairs=={(co,seed) for co in cohorts for seed in MODEL_SEEDS}, 'Fixed model/cohort matrix is incomplete')
    require(len({value[0] for value in models.values()})==len(models), 'Different model seeds require distinct checkpoint identities')
    return spec


def predict_arguments(spec, row, commit):
    co = spec['benchmark']['cohorts'][row['cohort']]
    return dict(run_id=spec['run_id'],row_id=row['row_id'],release_commit=commit,row_root=row['row_root'],
                capsule=co['capsule'],output=str(PurePosixPath(row['output_root'])/'predictions'),
                config=deepcopy(spec['benchmark']['config']),expected_capsule_id=co['expected_capsule_id'],
                expected_checkpoint_sha256=row['expected_checkpoint_sha256'],branch_features=row['branch_features'],
                ground_packet=row['ground_packet'])


def command(spec, row, commit, config_path, *, preflight=False):
    args = predict_arguments(spec,row,commit)
    args['config'] = str(PurePosixPath(str(config_path).replace('\\','/')))
    argv = [spec['code']['environment'],'-u',str(PurePosixPath(spec['code']['cwd'])/'tools/evaluate_d92_group_barrier_joint_benchmark.py')]
    for key,value in args.items():
        argv.extend(['--'+key.replace('_','-'),str(value)])
    if preflight:
        argv.append('--preflight-only')
    return argv


def preflight_process(argv, log, cwd, environment):
    result = subprocess.run(argv,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8')
    with Path(log).open('x',encoding='utf-8') as stream:
        stream.write(result.stdout); stream.write(result.stderr)
    if result.returncode:
        raise RuntimeError('Row metadata preflight failed; see preserved log')
    return json.loads(result.stdout,object_pairs_hook=_object)


def launch(argv, log, cwd, environment, started):
    with Path(log).open('x',encoding='utf-8') as stream:
        process = subprocess.Popen(argv,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,
                                   stdout=stream,stderr=subprocess.STDOUT)
        started(process.pid)
        return dict(pid=process.pid,returncode=process.wait())


def verify_marker(directory, spec, row, commit, preflight):
    """Verify frozen output identity/coverage; never fit, solve or connect truth."""
    directory = Path(directory)
    marker,startup = read(directory/'predictions_complete.json'),read(directory/'startup.json')
    validate_preflight(preflight,spec,row,commit)
    fields = ('schema','method','run_id','row_id','release_commit','capsule_id','checkpoint_sha256','model_seed',
              'algorithm','group_barrier_resources','run_binding','source_identity','ground_packet_identity',
              'ordered_ground_classes','old_classes','split_count','query_record_count','query_fit_access',
              'source_fit_access','truth_read','checkpoint_loaded','encoder_called','cross_split_adapted_state_reuse',
              'technical_query_chunk_size','A_tie_policy','B_C_tie_policy','fit_scope','query_inference_scope')
    fields+=('C_decision_policy','C_decision_certificate_schema')
    for value in (startup,marker):
        require(all(value.get(k)==preflight[k] for k in fields), 'Predictor startup/complete identity mismatch')
    require(startup.get('config')==spec['benchmark']['config'] and marker.get('status')=='COMPLETE', 'Incomplete predictor output')
    require(type(startup.get('pid')) is int and startup['pid']>0 and marker.get('pid')==startup['pid'],'Predictor actual PID evidence differs')
    require(startup.get('cuda_visible_devices')=='' and all(startup.get('blas_environment',{}).get(k)=='2' for k in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'Actual CPU2/BLAS2 environment differs')
    splits = {item['split_id']:item for item in preflight['splits']}
    completed = marker.get('splits',[])
    require(marker.get('completed_split_count')==len(splits) and len(completed)==len(splits)
            and {item['split_id'] for item in completed}==set(splits), 'Incomplete physical split coverage')
    expected = set(); registries = {}
    stages = 0
    for item in completed:
        original = splits[item['split_id']]
        require(all(item.get(k)==v for k,v in original.items()), 'Completed split physical metadata mismatch')
        reuse = not original['new_support_ids']; stages += 1+int(not reuse)
        require(item.get('c_reuses_b') is reuse and item.get('c_inherited_from_b_state_ref')==item.get('b_state_ref')
                and item.get('support_only_fit') is True and item.get('query_fit_access') is False
                and item.get('source_fit_access') is False, 'Actual B/C inheritance differs')
        if reuse:
            require(item.get('c_state_ref')==item.get('b_state_ref'), 'new0 did not reuse actual B')
        for key,state_name in (('b_state_ref','B_MARGIN'),('c_state_ref','B_MARGIN' if reuse else 'C_GROUP_BARRIER_seq')):
            ref = item.get(key,{})
            namespace = json.loads(ref.get('namespace','{}'),object_pairs_hook=_object)
            require(all(namespace.get(k)==v for k,v in dict(run_id=spec['run_id'],row_id=row['row_id'],
                    split_id=item['split_id'],scope='query_benchmark_support_training',state=state_name).items()),
                    'Final stage reference namespace differs')
            relative = PurePosixPath(ref.get('path',''))
            require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
                    and (directory/relative).is_file(), 'Missing actual final state archive')
        require(item.get('B_classes')==original['old_classes'] and item.get('C_classes')==original['registered_classes'],
                'Actual stage registry differs')
        for pid in original['query_ids']:
            key = (item['split_id'],pid); require(key not in expected,'Duplicate physical query coordinate')
            expected.add(key); registries[key] = original
    require(marker.get('actual_stage_count')==stages, 'Stage coverage differs')
    require(set(marker.get('streams',{}))=={'A','B','C'}, 'Three fixed prediction streams required')
    all_records = {}
    for name in ('A','B','C'):
        info = marker['streams'][name]; path = directory/('predictions_'+name+'.jsonl')
        require(info.get('path')==path.name and info.get('record_count')==len(expected)
                and info.get('file_bytes')==path.stat().st_size, 'Stream file metadata differs')
        records = {}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line,object_pairs_hook=_object)
                require(set(record)=={'split_id','query_id','classes','scores','prediction'}, 'Prediction record schema differs')
                key = (record['split_id'],record['query_id'])
                require(key in expected and key not in records, 'Unexpected/duplicate query physical ID')
                classes = preflight['ordered_ground_classes'] if name=='A' else registries[key]['old_classes' if name=='B' else 'registered_classes']
                values = record['scores']
                require(record['classes']==classes and isinstance(values,list) and len(values)==len(classes)
                        and all(type(v) in (int,float) and math.isfinite(v) for v in values), 'Incomplete all-column finite scores')
                if name!='C':require(record['prediction']==classes[max(range(len(values)),key=values.__getitem__)], 'Prediction was not fixed by declared tie policy')
                else:require(record['prediction'] in classes,'Structured C predicted class unregistered')
                records[key] = record
        require(set(records)==expected, 'Missing query physical IDs in stream')
        all_records[name] = records
    alias = marker.get('compatibility_predictions',{}); path = directory/'predictions.jsonl'
    require(alias==dict(path=path.name,alias_of='C',file_bytes=path.stat().st_size)
            and path.read_bytes()==(directory/'predictions_C.jsonl').read_bytes(), 'Compatibility output is not the fixed C stream')
    require(marker.get('state_manifest')=='state_manifest.json' and read(directory/'state_manifest.json').get('status')=='COMPLETE',
            'Training state archive is incomplete')
    info=marker.get('query_score_work',{});workpath=directory/'query_score_work.jsonl'
    require(info==dict(path=workpath.name,record_count=len(expected),file_bytes=workpath.stat().st_size),'Query decision/work file metadata differs')
    work={}
    with workpath.open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line,object_pairs_hook=_object);key=(item['split_id'],item['query_id'])
            require(key in expected and key not in work,'Ambiguous query decision coordinate')
            decision=item['C_structural_decision'];reuse=not registries[key]['new_support_ids']
            require(decision.get('schema')==DECISION_SCHEMA and decision.get('new0_reuses_actual_B') is reuse
                and decision.get('prediction')==item['C_structural_prediction']==all_records['C'][key]['prediction'],'Public structured decision binding differs')
            require(item['C_reused_B_scores'] is reuse and item['C_public_predict_calls']==int(not reuse),'Actual public inference call/reuse differs')
            if reuse:
                require(decision['policy']=='EXACT_ACTUAL_B_REUSE' and all_records['C'][key]==all_records['B'][key],'new0 fixed C is not exact B record')
            else:
                bc=all_records['B'][key];old=decision['old_classes'];new=decision['new_classes']
                require(decision['policy']==DECISION_POLICY and old==bc['classes'] and decision['old_raw_scores']==bc['scores']
                    and sorted(old+new)==all_records['C'][key]['classes'] and not set(old).intersection(new),'Decision all-column/frozen B binding differs')
                require(all(len(decision[k])==len(old if k.startswith('old') else new) and
                    all(type(v) in (int,float) and math.isfinite(v) for v in decision[k]) for k in
                    ('old_raw_scores','new_raw_scores','old_log_probabilities','new_log_probabilities')),'Incomplete finite structural arrays')
                bo=max(range(len(old)),key=decision['old_raw_scores'].__getitem__);bn=max(range(len(new)),key=decision['new_raw_scores'].__getitem__)
                gap=decision['gate_logit']+decision['old_log_probabilities'][bo]-decision['new_log_probabilities'][bn]
                require(math.isfinite(gap) and gap==decision['group_gap'] and decision['old_winner']==old[bo] and decision['new_winner']==new[bn],
                    'Recorded raw maxima/group gap differ')
                predicted=old[bo] if gap>0 else new[bn] if gap<0 else min(old[bo],new[bn])
                require(decision['prediction']==predicted,'Recorded structural all-class rule differs')
            work[key]=item
    require(set(work)==expected,'Missing public structural decision coordinates')
    from evaluate_d92_group_barrier_joint_benchmark import AUDIT_COUNTERS,PEAKS,factorization_attempts
    counters=marker['resources']['actual_training_counters']
    for key in AUDIT_COUNTERS+PEAKS:
        value=counters.get(key)
        require((type(value) in (int,float) and math.isfinite(value) if key.endswith('_seconds') else type(value) is int) and value>=0,
            'Invalid actual workload '+key)
    require(marker['resources']['actual_factorization_attempts']==factorization_attempts(counters),'Actual forward/adjoint factor total differs')
    return marker


def validate_preflight(value, spec, row, commit):
    args = predict_arguments(spec,row,commit)
    expected = dict(schema='d92_group_barrier_joint_query_predictions_v1',method='D92-GroupBarrierJointLocalRidge-v1',
        run_id=spec['run_id'],row_id=row['row_id'],release_commit=commit,capsule_id=args['expected_capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],
        algorithm=args['config']['algorithm'],group_barrier_resources=args['config']['group_barrier_resources'],
        query_fit_access=False,source_fit_access=False,truth_read=False,checkpoint_loaded=False,encoder_called=False,
        cross_split_adapted_state_reuse=False,technical_query_chunk_size=1)
    expected.update(C_decision_policy=DECISION_POLICY,C_decision_certificate_schema=DECISION_SCHEMA)
    require(value.get('status')=='GROUP_BARRIER_QUERY_PREFLIGHT_COMPLETE' and all(value.get(k)==v for k,v in expected.items()),
            'Actual preflight source/method/access identity differs')
    paths = dict(prediction_output_root=args['output'],**{k:args[k] for k in ('row_root','capsule','branch_features','ground_packet')})
    require(set(value.get('run_binding',{}))==set(paths) and all(_path(value['run_binding'][k])==_path(v) for k,v in paths.items()),
            'Actual preflight source paths differ')
    for key in ('source_identity','ground_packet_identity'):
        require(value.get(key,{}).get('checkpoint_sha256')==row['expected_checkpoint_sha256']
                and value[key].get('model_seed')==row['expected_model_seed'], 'Actual packet/cache model differs')
    splits = value.get('splits',[])
    require(isinstance(splits,list) and splits and value.get('split_count')==len(splits)
            and len({v['split_id'] for v in splits})==len(splits), 'Explicit complete split metadata required')
    total = 0
    for item in splits:
        support,old,new,query = (item[k] for k in ('support_ids','old_support_ids','new_support_ids','query_ids'))
        require(all(isinstance(v,list) and len(v)==len(set(v)) and all(isinstance(p,str) and p for p in v)
                    for v in (support,old,new,query)), 'Physical ID metadata is ambiguous')
        require(set(old).isdisjoint(new) and set(support)==set(old)|set(new) and set(support).isdisjoint(query)
                and item['query_count']==len(query), 'Support/query identity metadata differs')
        total += len(query)
    require(value.get('query_record_count')==total, 'Expected physical query coverage differs')
    co=spec['benchmark']['cohorts'][row['cohort']];matrix=co['matrix']
    cells=[tuple(v[k] for k in ('receiver','scenario','k','new_count','support_seed')) for v in splits]
    require(len(cells)==co['expected_split_count'] and len(set(cells))==len(cells)
        and set(cells)==set(itertools.product(*(matrix[k] for k in ('receivers','scenarios','ks','new_counts','support_seeds')))),
        'Actual complete physical matrix differs from all declared axes')
    return value


def row_metadata(spec, row, commit):
    args = predict_arguments(spec,row,commit)
    return dict(status='PENDING',release_commit=commit,expected_model_seed=row['expected_model_seed'],
                expected_checkpoint_sha256=row['expected_checkpoint_sha256'],expected_capsule_id=args['expected_capsule_id'],
                source_paths={k:args[k] for k in ('row_root','capsule','branch_features','ground_packet')},
                output_root=row['output_root'],prediction_output=args['output'],
                predictions_complete_path=str(PurePosixPath(args['output'])/'predictions_complete.json'))


def run(spec, commit, *, preflight_fn=preflight_process, launch_fn=launch, marker_fn=verify_marker):
    validate_spec(spec); require(oid(commit), 'Actual publisher runtime commit required')
    root = Path(spec['execution']['remote_run_root']); root.mkdir(parents=True,exist_ok=False)
    config_path = root/'resolved_algorithm.json'; write(config_path,spec['benchmark']['config'])
    environment = dict(os.environ,**CPU_ENV)
    states = {row['row_id']:row_metadata(spec,row,commit) for row in spec['rows']}
    lock = threading.Lock()
    startup = dict(schema=SCHEMA,status=STARTED,run_id=spec['run_id'],group_id=spec['group_id'],runtime_commit=commit,
        code_commit=spec['code']['commit'],resolved_spec=deepcopy(spec),pid=os.getpid(),argv=sys.argv,python=sys.executable,
        environment=CPU_ENV,rows=deepcopy(states),started=time.time(),truth_read=False,scorer_invoked=False,
        checkpoint_loaded=False,encoder_loaded=False,probe_state_inherited=False,automatic_retry=False)
    write(root/'startup.json',startup); write(root/'state.json',states)
    def update(row, **fields):
        with lock:
            states[row['row_id']].update(fields,updated=time.time())
            write(root/'state.json',states,replace=True)
            with (root/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(run_id=spec['run_id'],row_id=row['row_id'],**fields),allow_nan=False)+'\n')
            print(json.dumps(dict(row_id=row['row_id'],**fields),allow_nan=False),flush=True)
    # All rows receive their preflight decision before any prediction is invoked.
    physical_cohorts = {}
    for row in spec['rows']:
        out = Path(row['output_root'])
        row_config=out/'resolved_config.json'
        argv = command(spec,row,commit,row_config,preflight=True)
        try:
            out.mkdir(exist_ok=False)
            write(row_config,spec['benchmark']['config'])
            value = preflight_fn(argv,out/'preflight.log',Path(spec['code']['cwd']),environment)
            require(value.get('status')=='GROUP_BARRIER_QUERY_PREFLIGHT_COMPLETE', 'Preflight did not verify inputs')
            validate_preflight(value,spec,row,commit)
            physical = value['splits']
            require(physical_cohorts.setdefault(row['cohort'],physical)==physical, 'Same cohort model rows changed physical split metadata')
            write(out/'preflight.json',value)
            update(row,status='PREFLIGHT_COMPLETE',preflight=value,preflight_argv=argv,environment=CPU_ENV)
        except Exception as exc:
            update(row,status='FAILED',phase='PREFLIGHT',error_type=type(exc).__name__,error=str(exc),preflight_argv=argv,environment=CPU_ENV)
    def work(row):
        if states[row['row_id']]['status']=='FAILED':
            return
        out = Path(row['output_root']); argv = command(spec,row,commit,out/'resolved_config.json')
        try:
            update(row,status='PREDICTING',argv=argv,environment=CPU_ENV,prediction_log=str(out/'prediction.log'))
            result = launch_fn(argv,out/'prediction.log',Path(spec['code']['cwd']),environment,
                               lambda pid:update(row,pid=pid))
            update(row,status='VERIFYING_FIXED_OUTPUT',pid=result['pid'],process_returncode=result['returncode'])
            marker = marker_fn(out/'predictions',spec,row,commit,states[row['row_id']]['preflight'])
            require(marker.get('pid')==result['pid'] and type(result['pid']) is int and result['pid']>0,'Observed row process PID does not match complete evidence')
            require(result['returncode']==0, 'Predictor exit failed; preserve completion evidence and reconcile')
            update(row,status='COMPLETE',pid=result['pid'],process_returncode=result['returncode'],marker=marker)
        except Exception as exc:
            update(row,status='FAILED',phase='PREDICTION',error_type=type(exc).__name__,error=str(exc),argv=argv,
                   preserved_prediction_output=str(out/'predictions'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work,spec['rows']))
    success = all(value['status']=='COMPLETE' for value in states.values())
    complete = dict(schema=SCHEMA,status=STATUS if success else FAILED,run_id=spec['run_id'],group_id=spec['group_id'],
        runtime_commit=commit,code_commit=spec['code']['commit'],resolved_spec=deepcopy(spec),rows=states,
        row_count=len(states),completed_row_count=sum(v['status']=='COMPLETE' for v in states.values()),
        all_predictions_fixed=success,truth_read=False,scorer_invoked=False,automatic_retry=False,finished=time.time())
    write(root/'complete.json',complete)
    if not success:
        raise RuntimeError('One or more prediction rows failed; healthy rows and all artifacts preserved; no retry')
    return complete


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True); parser.add_argument('--commit',required=True)
    args = parser.parse_args(); run(read(args.spec),args.commit)
