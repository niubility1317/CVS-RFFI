"""Synthetic supervisor/physical-output contracts; no checkpoint or IQ access."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from cvsrffi.d92_margin_joint_local_ridge import FROZEN_CONFIG
import run_d92_margin_joint_benchmark as run

COMMIT = 'b'*40


def spec(tmp_path):
    root = (tmp_path/'new-run').as_posix(); source = (tmp_path/'source').as_posix()
    cohorts = {co:dict(capsule=source+'/'+co,truth=source+'/'+co+'/truth.json',expected_capsule_id='capsule-'+co)
               for co in ('rx3','rx1')}
    rows = [dict(row_id=co+'-m'+str(seed),cohort=co,expected_model_seed=seed,
        expected_checkpoint_sha256=str(seed+1)*64,row_root=source+'/model'+str(seed),
        branch_features=source+'/'+co+'-features'+str(seed),ground_packet=source+'/packet'+str(seed),
        output_root=root+'/'+co+'-m'+str(seed)) for co in cohorts for seed in (0,1)]
    return dict(schema=run.SCHEMA,run_id='new',group_id='matched',spec_path='configs/new.json',
        code=dict(cwd=(tmp_path/'release').as_posix(),environment=Path('/synthetic/python').as_posix(),commit='a'*40),
        execution=dict(remote_run_root=root,launch_owner='root',cpu_lanes=2,blas_threads=2),
        benchmark=dict(config=dict(algorithm=deepcopy(FROZEN_CONFIG),qp_resources=dict(max_transitions=20,max_factor_buffer_bytes=1000000)),cohorts=cohorts),rows=rows)


def preflight(s, row, *, new=True):
    args = run.predict_arguments(s,row,COMMIT); old = ['old'+str(i) for i in range(6)]
    extra = ['new0','new1'] if new else []
    names = sorted(old+extra); prefix = row['cohort']
    oldids = [prefix+'-o'+str(i) for i in range(6)]; newids = [prefix+'-n'+str(i) for i in range(len(extra))]
    item = dict(split_id=prefix+'-split',receiver=prefix,scenario='leo_clear_weak',k=1,new_count=len(extra),support_seed=0,
        old_classes=old,registered_classes=names,declared_registered_classes=names,
        support_ids=sorted(oldids+newids),old_support_ids=oldids,new_support_ids=newids,
        query_ids=[prefix+'-q0',prefix+'-q1'],query_count=2)
    return dict(status='MARGIN_QUERY_PREFLIGHT_COMPLETE',schema='d92_margin_joint_query_predictions_v1',method='D92-MarginJointLocalRidge-v1',
        run_id=s['run_id'],row_id=row['row_id'],release_commit=COMMIT,capsule_id=args['expected_capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],
        algorithm=deepcopy(args['config']['algorithm']),qp_resources=deepcopy(args['config']['qp_resources']),
        run_binding=dict(prediction_output_root=args['output'],**{k:args[k] for k in ('row_root','capsule','branch_features','ground_packet')}),
        source_identity=dict(checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed']),
        ground_packet_identity=dict(checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed']),
        ordered_ground_classes=list(reversed(old)),old_classes=old,splits=[item],split_count=1,query_record_count=2,
        query_fit_access=False,source_fit_access=False,truth_read=False,checkpoint_loaded=False,encoder_called=False,
        cross_split_adapted_state_reuse=False,technical_query_chunk_size=1,A_tie_policy='first_original_native_head_column',B_C_tie_policy='physical_class_id_ascending',
        fit_scope='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY',query_inference_scope='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS')


def fixed_output(s,row,p):
    out = Path(row['output_root'])/'predictions'; out.mkdir(parents=True)
    run.write(out/'startup.json',dict(p,config=s['benchmark']['config']))
    item = deepcopy(p['splits'][0]); reuse = not item['new_support_ids']
    for key,state in (('b_state_ref','B_MARGIN'),('c_state_ref','B_MARGIN' if reuse else 'C_MARGIN_seq')):
        file = out/('state_arrays/'+state+'.npz'); file.parent.mkdir(exist_ok=True)
        if not file.exists():file.write_bytes(b'synthetic structural reference, not a numerical state')
        item[key] = dict(path=file.relative_to(out).as_posix(),namespace=json.dumps(dict(run_id=s['run_id'],row_id=row['row_id'],
            split_id=item['split_id'],scope='query_benchmark_support_training',state=state)))
    item.update(c_reuses_b=reuse,c_inherited_from_b_state_ref=item['b_state_ref'],B_classes=item['old_classes'],
                C_classes=item['registered_classes'],support_only_fit=True,query_fit_access=False,source_fit_access=False)
    streams = {}
    for name in ('A','B','C'):
        classes = p['ordered_ground_classes'] if name=='A' else item['old_classes' if name=='B' else 'registered_classes']
        records = [dict(split_id=item['split_id'],query_id=pid,classes=classes,scores=[0.]*len(classes),prediction=classes[0]) for pid in item['query_ids']]
        file = out/('predictions_'+name+'.jsonl'); file.write_text(''.join(json.dumps(v)+'\n' for v in records),encoding='utf-8')
        streams[name] = dict(path=file.name,record_count=2,file_bytes=file.stat().st_size)
    (out/'predictions.jsonl').write_bytes((out/'predictions_C.jsonl').read_bytes())
    run.write(out/'state_manifest.json',dict(status='COMPLETE'))
    marker = dict(p,status='COMPLETE',splits=[item],completed_split_count=1,actual_stage_count=1 if reuse else 2,
        streams=streams,compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),state_manifest='state_manifest.json')
    run.write(out/'predictions_complete.json',marker)
    return out


@pytest.mark.parametrize('change', ['bool_limit','method','lane','owner','missing_row','overlap','seed_packet'])
def test_reject_changed_contract(tmp_path,change):
    s = spec(tmp_path)
    if change=='bool_limit':s['benchmark']['config']['qp_resources']['max_transitions']=True
    elif change=='method':s['benchmark']['config']['algorithm']['max_iterations']=5
    elif change=='lane':s['execution']['cpu_lanes']=3
    elif change=='owner':s['execution']['launch_owner']='another'
    elif change=='missing_row':s['rows'].pop()
    elif change=='overlap':s['rows'][0]['ground_packet']=s['execution']['remote_run_root']
    else:s['rows'][2]['ground_packet']+='/changed'
    with pytest.raises(ValueError):run.validate_spec(s)


def test_command_actual_commit_and_truth_last(tmp_path):
    s = spec(tmp_path); run.validate_spec(s)
    argv = run.command(s,s['rows'][0],COMMIT,tmp_path/'resolved.json')
    assert argv[argv.index('--release-commit')+1]==COMMIT
    assert COMMIT!=s['code']['commit']
    assert not any('truth' in v for v in argv)
    assert '--checkpoint' not in argv and '--encoder' not in argv
    assert '--preflight-only' in run.command(s,s['rows'][0],COMMIT,tmp_path/'resolved.json',preflight=True)


@pytest.mark.parametrize('new', [True,False])
def test_complete_physical_streams_and_new0(tmp_path,new):
    s = spec(tmp_path); row = s['rows'][0]; p = preflight(s,row,new=new)
    out = fixed_output(s,row,p)
    assert run.verify_marker(out,s,row,COMMIT,p)['status']=='COMPLETE'


@pytest.mark.parametrize('change', ['alias','id','release','inherit','namespace','column','duplicate','prediction','archive'])
def test_reject_tampered_fixed_output(tmp_path,change):
    s=spec(tmp_path);row=s['rows'][0];p=preflight(s,row);out=fixed_output(s,row,p)
    marker=run.read(out/'predictions_complete.json')
    if change=='alias':(out/'predictions.jsonl').write_text('{}\n',encoding='utf-8');marker['compatibility_predictions']['file_bytes']=3
    elif change=='release':marker['release_commit']='c'*40
    elif change=='inherit':marker['splits'][0]['c_inherited_from_b_state_ref']={'path':'another'}
    elif change=='namespace':marker['splits'][0]['b_state_ref']['namespace']='{}'
    elif change=='archive':(out/marker['splits'][0]['b_state_ref']['path']).unlink()
    else:
        file=out/'predictions_A.jsonl';records=[json.loads(v) for v in file.read_text(encoding='utf-8').splitlines()]
        if change=='id':records[0]['query_id']='extra-query'
        elif change=='column':records[0]['classes'].reverse()
        elif change=='prediction':records[0]['prediction']='invented'
        else:records.append(records[0])
        file.write_text(''.join(json.dumps(v)+'\n' for v in records),encoding='utf-8');marker['streams']['A']['file_bytes']=file.stat().st_size
    run.write(out/'predictions_complete.json',marker,replace=True)
    with pytest.raises((ValueError,FileNotFoundError)):run.verify_marker(out,s,row,COMMIT,p)


def test_all_preflight_first_failed_row_keeps_healthy(tmp_path):
    s=spec(tmp_path);book=[];lookup={r['row_id']:r for r in s['rows']}
    def pre(argv,log,cwd,env):
        name=argv[argv.index('--row-id')+1];book.append(('pre',name));Path(log).write_text('synthetic',encoding='utf-8')
        return preflight(s,lookup[name])
    def launch(argv,log,cwd,env,started):
        name=argv[argv.index('--row-id')+1]
        assert len([v for v in book if v[0]=='pre'])==4
        assert all(env[k]==v for k,v in run.CPU_ENV.items());started(100)
        Path(log).write_text('synthetic process',encoding='utf-8');book.append(('launch',name))
        if name==s['rows'][0]['row_id']:raise RuntimeError('synthetic owned failure')
        fixed_output(s,lookup[name],preflight(s,lookup[name]));return dict(pid=100,returncode=0)
    with pytest.raises(RuntimeError):run.run(s,COMMIT,preflight_fn=pre,launch_fn=launch)
    done=run.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert done['status']==run.FAILED and done['completed_row_count']==3
    assert done['truth_read'] is False and done['scorer_invoked'] is False and done['all_predictions_fixed'] is False
    assert sum(v[0]=='launch' for v in book)==4
    assert done['runtime_commit']==COMMIT and done['code_commit']=='a'*40
    with pytest.raises(FileExistsError):run.run(s,COMMIT,preflight_fn=pre,launch_fn=launch)


def test_zero_exit_without_marker_is_failure(tmp_path):
    s=spec(tmp_path); lookup={r['row_id']:r for r in s['rows']}
    def pre(argv,*args):return preflight(s,lookup[argv[argv.index('--row-id')+1]])
    def launch(argv,log,cwd,env,started):started(101);return dict(pid=101,returncode=0)
    with pytest.raises(RuntimeError):run.run(s,COMMIT,preflight_fn=pre,launch_fn=launch)
    done=run.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert done['completed_row_count']==0 and all(v['status']=='FAILED' for v in done['rows'].values())


def test_preflight_model_mismatch_before_any_inference(tmp_path):
    s=spec(tmp_path);lookup={r['row_id']:r for r in s['rows']};calls=[]
    def pre(argv,*args):
        value=preflight(s,lookup[argv[argv.index('--row-id')+1]]);value['model_seed']=99;return value
    def launch(*args):calls.append(args);raise AssertionError('must not infer')
    with pytest.raises(RuntimeError):run.run(s,COMMIT,preflight_fn=pre,launch_fn=launch)
    assert not calls
