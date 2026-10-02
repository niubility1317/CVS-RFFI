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
SCHEMA = 'd92_support_metric_joint_query_benchmark_v1'
STARTED = 'SUPPORT_METRIC_QUERY_BENCHMARK_STARTED'
STATUS = 'SUPPORT_METRIC_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
FAILED = 'SUPPORT_METRIC_QUERY_BENCHMARK_PREDICTIONS_FAILED'
ROW_FIELDS = {'row_id','cohort','expected_model_seed','expected_checkpoint_sha256',
              'row_root','branch_features','ground_packet','ground_summary',
              'ground_summary_already_deployed','output_root'}
CPU_ENV = {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
           'NUMEXPR_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'}
MODEL_SEEDS={2026092701,2026092702}
RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160,
    max_integer_bits=65536,max_fraction_operations=65536,max_secular_iterations=128)
MATRIX_BASE=dict(scenarios=['practical_high','practical_mid','practical_low_urban'],ks=[1,5,10,20],
                 new_counts=[0,2,5,10,20],support_seeds=[2026092711,2026092712,2026092713,2026092714,2026092715])
RECEIVERS=dict(rx3=['19-1','8-14','8-7'],rx1=['20-19'])
DECISION_SCHEMA='d92_support_metric_single_query_decision_v1'
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
    from cvsrffi.d92_support_metric_joint_local_ridge import FROZEN_CONFIG
    config = benchmark['config']
    require(set(config)=={'algorithm','support_metric_resources'} and config['algorithm']==FROZEN_CONFIG, 'Frozen method config mismatch')
    limits = config['support_metric_resources']
    require(set(limits)==set(RESOURCES) and
            all(type(v) is int and v>0 for v in limits.values()), 'Explicit positive integer SupportMetric limits required')
    require(limits['max_secular_iterations']<=128,'Bounded secular execution ceiling exceeded')
    require(limits==RESOURCES,'Frozen SupportMetric technical limits differ')
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
        require(type(row['ground_summary_already_deployed']) is bool, 'Explicit ground-summary deployment flag required')
        for key in ('row_root','branch_features','ground_packet','ground_summary'):
            require(absolute(row[key]) and _separate(row[key],execution['remote_run_root']), 'Source/output overlap: '+key)
        model = (row['expected_checkpoint_sha256'],row['ground_packet'],row['ground_summary'],row['ground_summary_already_deployed'])
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
                ground_packet=row['ground_packet'],ground_summary=row['ground_summary'],
                ground_summary_already_deployed=row['ground_summary_already_deployed'])


def command(spec, row, commit, config_path, *, preflight=False):
    args = predict_arguments(spec,row,commit)
    args['config'] = str(PurePosixPath(str(config_path).replace('\\','/')))
    argv = [spec['code']['environment'],'-u',str(PurePosixPath(spec['code']['cwd'])/'tools/evaluate_d92_support_metric_joint_benchmark.py')]
    for key,value in args.items():
        argv.extend(['--'+key.replace('_','-'),str(value).lower() if type(value) is bool else str(value)])
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
    """Use the standalone zero-truth validator; no fitting module is imported."""
    from score_d92_support_metric_joint_benchmark import validate_row_output
    return validate_row_output(directory,spec,row,commit,preflight)


def validate_preflight(value, spec, row, commit):
    args = predict_arguments(spec,row,commit)
    expected = dict(schema='d92_support_metric_joint_query_predictions_v1',method='D92-ProtoFrameSupportMetric-GGN1-LocalRidge',
        run_id=spec['run_id'],row_id=row['row_id'],release_commit=commit,capsule_id=args['expected_capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],
        algorithm=args['config']['algorithm'],support_metric_resources=args['config']['support_metric_resources'],
        query_fit_access=False,source_fit_access=False,truth_read=False,checkpoint_loaded=False,encoder_called=False,
        cross_split_adapted_state_reuse=False,technical_query_chunk_size=1,
        ground_summary_already_deployed=row['ground_summary_already_deployed'])
    expected.update(C_decision_policy=DECISION_POLICY,C_decision_certificate_schema=DECISION_SCHEMA)
    require(value.get('status')=='SUPPORT_METRIC_QUERY_PREFLIGHT_COMPLETE' and all(value.get(k)==v for k,v in expected.items()),
            'Actual preflight source/method/access identity differs')
    paths = dict(prediction_output_root=args['output'],**{k:args[k] for k in ('row_root','capsule','branch_features','ground_packet','ground_summary')})
    require(set(value.get('run_binding',{}))==set(paths) and all(_path(value['run_binding'][k])==_path(v) for k,v in paths.items()),
            'Actual preflight source paths differ')
    for key in ('source_identity','ground_packet_identity','ground_geometry_identity'):
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
                source_paths={k:args[k] for k in ('row_root','capsule','branch_features','ground_packet','ground_summary')},
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
            require(value.get('status')=='SUPPORT_METRIC_QUERY_PREFLIGHT_COMPLETE', 'Preflight did not verify inputs')
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
        declared_episode_count=sum(spec['benchmark']['cohorts'][r['cohort']]['expected_split_count'] for r in spec['rows']),
        completed_episode_count=sum(v['preflight']['split_count'] for v in states.values() if v['status']=='COMPLETE'),
        all_predictions_fixed=success,truth_read=False,scorer_invoked=False,automatic_retry=False,finished=time.time())
    write(root/'complete.json',complete)
    if not success:
        raise RuntimeError('One or more prediction rows failed; healthy rows and all artifacts preserved; no retry')
    return complete


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True); parser.add_argument('--commit',required=True)
    args = parser.parse_args(); run(read(args.spec),args.commit)
