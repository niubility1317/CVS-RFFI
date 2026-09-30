"""Synthetic identity/transport tests; no real features, model or query access."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import prepare_d92_registration_diagnostic as prep
import run_d92_registration_diagnostic as run
import preflight_d92_registration_diagnostic as preflight
import publish_d92_registration_diagnostic as publish


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);run.write(path,value)


@pytest.fixture
def source():
    # Configuration metadata only; all support identities below are synthetic.
    parent=run.read(ROOT/prep.PARENT_SPEC)
    manifests={}
    for name,co in parent['probe']['cohorts'].items():
        matrix=co['matrix'];matrix['receivers']=['synthetic-'+name]
        matrix['support_seeds']=[11,12] if name=='rx3' else [11,13]
        co['expected_split_count']=120;co['capsule_id']='residual-noeq-synthetic-'+name
        splits=[]
        for receiver,scene,k,n,seed in itertools.product(matrix['receivers'],matrix['scenarios'],matrix['ks'],matrix['new_counts'],matrix['support_seeds']):
            classes=['old'+str(i) for i in range(6)]+['new'+str(i) for i in range(n)]
            labels=[c for c in range(len(classes)) for _ in range(k)]
            ids=[f'{receiver}:{scene}:{k}:{seed}:{c}:{j}' for c in classes for j in range(k)]
            splits.append(dict(split_id=f'{receiver}:{scene}:{k}:{n}:{seed}',receiver=receiver,scenario=scene,k=k,
                support_seed=seed,registered_classes=classes,support_ids=ids,support_indices=list(range(len(ids))),support_labels=labels))
        manifests[name]=dict(schema='d92_branch_support_features_v1',capsule_id=co['capsule_id'],
            checkpoint_sha256=next(r['expected_checkpoint_sha256'] for r in parent['rows'] if r['cohort']==name),splits=splits)
    for row in parent['rows']:
        co=parent['probe']['cohorts'][row['cohort']]
        row['data_overrides']=dict(capsule=co['capsule'],capsule_id=co['capsule_id'],expected_split_count=120)
    return parent,manifests


def prepared(source):
    return prep.documents(*source,commit='a'*40)


def local_spec(source,tmp_path):
    docs=prepared(source);spec=docs[prep.SPEC]
    spec['code']['cwd']=(tmp_path/'release').as_posix()
    spec['execution']['remote_run_root']=(tmp_path/'output').as_posix()
    for name,co in spec['probe']['cohorts'].items():
        co['evaluation_config']=(tmp_path/'release'/run.CONFIG_NAMES[name]).as_posix()
        path=Path(co['evaluation_config']);path.parent.mkdir(parents=True,exist_ok=True)
        write(path,docs[run.CONFIG_NAMES[name]])
    for row in spec['rows']:row['output_root']=(tmp_path/'output'/row['row_id']).as_posix()
    return spec


def marker(spec,row):
    co=spec['probe']['cohorts'][row['cohort']]
    return dict(status=run.STATUS,capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],
        model_seed=row['seeds']['model'],algorithm=deepcopy(run.DIAGNOSTIC_CONFIG),selection=co['selection'],producer_matrix=co['matrix'],
        episodes=40,k1_episodes=10,oof_episodes=30,proxy_anchor_count=350,sequence_paths=440,
        head_fit_count=792,factorization_count=792,optimizer_steps=0,query_rows_used=0,source_rows_used=0)


def test_deterministic_complete_pilot_and_preserved_lineage(source):
    before=deepcopy(source);a=prepared(source)
    assert source==before
    parent,manifests=deepcopy(source)
    parent['rows'].reverse()
    for value in manifests.values():value['splits'].reverse()
    b=prepared((parent,manifests));s=a[prep.SPEC]
    assert s['probe']['selected_model_seeds']==sorted({r['seeds']['model'] for r in source[0]['rows']})[:2]
    assert len(s['rows'])==4 and s['probe']['expected_parents']==160
    assert s['execution']['cpu_lanes']==s['execution']['blas_threads_per_lane']==2
    for name,co in s['probe']['cohorts'].items():
        assert co['selection']==b[prep.SPEC]['probe']['cohorts'][name]['selection']
        assert co['matrix']==source[0]['probe']['cohorts'][name]['matrix']
        assert co['expected_split_count']==120 and co['selected_split_count']==40
        assert co['selection']['support_seed']==11
        assert [x[1] for x in co['selection']['receiver_scenes']]==['practical_high','practical_low_urban']
        assert len(co['selection']['splits'])==40
        assert all(set(x)==run.IDENTITY_KEYS for x in co['selection']['splits'])
    for row in s['rows']:
        old=next(r for r in source[0]['rows'] if r['row_id']==row['row_id'])
        assert row['support_features']==old['support_features']
        assert row['expected_checkpoint_sha256']==old['expected_checkpoint_sha256']
        assert row['data_overrides']==old['data_overrides']
        assert row['seeds']['support']==11


@pytest.mark.parametrize('mutation',[
    lambda m:m['rx3']['splits'].pop(),
    lambda m:m['rx3']['splits'].__setitem__(0,deepcopy(m['rx3']['splits'][1])),
    lambda m:m['rx3'].__setitem__('capsule_id','wrong'),
    lambda m:m['rx3'].__setitem__('checkpoint_sha256','0'*64),
    lambda m:m['rx3']['splits'][0].__setitem__('query_scores',[1]),
    lambda m:m['rx3']['splits'][0]['support_labels'].__setitem__(0,True),
    lambda m:m['rx3']['splits'][0]['support_ids'].__setitem__(0,'changed-old'),
    lambda m:m['rx3']['splits'][0]['support_indices'].__setitem__(0,-1),
])
def test_invalid_identity_metadata_rejected(source,mutation):
    mutation(source[1])
    with pytest.raises(ValueError):prepared(source)


@pytest.mark.parametrize('mutation',[
    lambda s:s['execution'].__setitem__('cpu_lanes',4),
    lambda s:s['probe']['channel'].__setitem__('route','ideal'),
    lambda s:s['probe'].__setitem__('selected_model_seeds',s['probe']['available_model_seeds'][1:3]),
    lambda s:s['probe']['cohorts']['rx3']['selection'].__setitem__('support_seed',12),
    lambda s:s['probe']['cohorts']['rx3']['selection']['splits'].pop(),
    lambda s:s['probe']['cohorts']['rx3'].__setitem__('expected_split_count',40),
    lambda s:s['rows'][0]['seeds'].__setitem__('support',None),
    lambda s:s['rows'][0].__setitem__('output_root',s['execution']['remote_run_root']+'/../escape'),
    lambda s:s['permissions'].__setitem__('query_use','allowed'),
    lambda s:s['probe'].__setitem__('expected_head_fits',3520),
])
def test_runner_rejects_contract_drift(source,mutation):
    spec=prepared(source)[prep.SPEC];mutation(spec)
    with pytest.raises((ValueError,KeyError)):run.validate_spec(spec)


def test_two_lane_run_and_nonoverwrite(source,tmp_path):
    spec=local_spec(source,tmp_path);lock=threading.Lock();barrier=threading.Barrier(2);active=0;maximum=0;calls=[]
    def fake(argv,log,cwd,stage):
        nonlocal active,maximum
        row=next(r for r in spec['rows'] if Path(r['output_root'])==log.parent)
        assert stage=='probe' and '--device' not in argv and '--expected-model-seed' in argv
        with lock:active+=1;maximum=max(maximum,active);calls.append(row['row_id'])
        barrier.wait(timeout=10)
        write(log.parent/'probe/probe_complete.json',marker(spec,row))
        with lock:active-=1
    run.run(spec,'b'*40,launch_fn=fake)
    assert maximum==2 and len(calls)==4
    done=run.read(Path(spec['execution']['remote_run_root'])/'complete.json')
    assert done['status']==run.STATUS and done['head_fit_count']==3168 and done['sequence_paths']==1760
    assert done['optimizer_steps']==0 and done['episodes']==160
    with pytest.raises(FileExistsError):run.run(spec,'b'*40,launch_fn=fake)


def test_failed_lane_preserved_and_no_retry(source,tmp_path):
    spec=local_spec(source,tmp_path);calls=[]
    def fake(argv,log,cwd,stage):
        row=next(r for r in spec['rows'] if Path(r['output_root'])==log.parent);calls.append(row['row_id'])
        if row is spec['rows'][0]:raise RuntimeError('synthetic technical failure')
        write(log.parent/'probe/probe_complete.json',marker(spec,row))
    with pytest.raises(RuntimeError,match='healthy lanes retained'):run.run(spec,'b'*40,launch_fn=fake)
    state=run.read(Path(spec['execution']['remote_run_root'])/'state.json')
    assert len(calls)==4 and len(set(calls))==4
    assert sum(r['status']==run.STATUS for r in state.values())==3
    assert state[spec['rows'][0]['row_id']]['error']=='synthetic technical failure'


@pytest.mark.parametrize('field,value',[('episodes',39),('head_fit_count',880),('optimizer_steps',1),('factorization_count',793),('factorization_count',True),('model_seed',0),('source_rows_used',1)])
def test_completion_binding(source,tmp_path,field,value):
    spec=prepared(source)[prep.SPEC];row=spec['rows'][0];m=marker(spec,row);m[field]=value
    path=tmp_path/'marker.json';write(path,m)
    with pytest.raises(ValueError):run.verify_marker(path,spec,row)


def test_config_mismatch_rejected_before_output_creation(source,tmp_path):
    spec=local_spec(source,tmp_path);p=Path(spec['probe']['cohorts']['rx3']['evaluation_config'])
    value=run.read(p);value['selection']['splits'].pop();write(p,value)
    with pytest.raises(ValueError,match='config/spec'):run.run(spec,'b'*40,launch_fn=lambda *a:None)
    assert not Path(spec['execution']['remote_run_root']).exists()


def test_private_cpu_publisher_and_standalone_preflight():
    import publish_d92_branch_support_probe as original
    assert 'nvidia-smi' in original.REMOTE and 'nvidia-smi' not in publish.transport.REMOTE
    assert original.validate_spec is not publish.transport.validate_spec
    assert "max_workers=2" in (ROOT/'tools/run_d92_registration_diagnostic.py').read_text(encoding='utf-8')
    for token in ("OMP_NUM_THREADS='2'","MKL_NUM_THREADS='2'","OPENBLAS_NUM_THREADS='2'","CUDA_VISIBLE_DEVICES=''"):
        assert token in publish.transport.REMOTE
    assert 'code' not in publish.transport.PATHS
    assert not any('support_residual' in path for path in publish.transport.PATHS)
    compile(preflight.REMOTE.replace('SPEC','{}'),'preflight','exec')
    compile(publish.transport.REMOTE.replace('CONFIG','{}'),'publisher','exec')
    assert '.npz\').read' not in preflight.REMOTE and 'np.load' not in preflight.REMOTE


@pytest.mark.parametrize('corrupt',[None,'channel','selection','checkpoint'])
def test_remote_preflight_metadata_only(source,tmp_path,monkeypatch,capsys,corrupt):
    spec=prepared(source)[prep.SPEC]
    spec['code']['cwd']=(tmp_path/'release').as_posix()
    spec['execution']['remote_run_root']=(tmp_path/'run').as_posix()
    for name,co in spec['probe']['cohorts'].items():
        co['capsule']=(tmp_path/('capsule-'+name)).as_posix()
        m=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=co['capsule_id'],
            split_count=120,channel=deepcopy(run.CHANNEL),scenarios=run.SCENARIOS)
        if corrupt=='channel':m['channel']['equalization_enabled']=True
        write(Path(co['capsule'])/'manifest.json',m)
    for row in spec['rows']:
        co=spec['probe']['cohorts'][row['cohort']];cache=tmp_path/row['row_id'];row['support_features']=str(cache)
        write(cache/'features_complete.json',dict(status='BRANCH_SUPPORT_FEATURES_COMPLETE',capsule_id=co['capsule_id'],
            checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],query_rows_read=0))
        plan=deepcopy(source[1][row['cohort']]);plan['checkpoint_sha256']=row['expected_checkpoint_sha256']
        if corrupt=='checkpoint':plan['checkpoint_sha256']='0'*64
        if corrupt=='selection':plan['splits'][0]['receiver']='wrong'
        write(cache/'support_splits.json',plan)
        for name in ('checkpoint_provenance.json','startup.json','support_branch_features.npz'):(cache/name).touch()
    original=Path.read_text;reads=[]
    def read_metadata(path,*args,**kwargs):
        assert path.name in {'manifest.json','features_complete.json','support_splits.json'}
        reads.append(path.name);return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_metadata)
    monkeypatch.setattr(os,'getloadavg',lambda:(0,0,0),raising=False)
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=4*1024**3))
    script=compile(preflight.REMOTE.replace('SPEC',repr(spec)),'preflight','exec')
    if corrupt:
        with pytest.raises(ValueError):exec(script,{})
    else:
        exec(script,{})
        result=json.loads(capsys.readouterr().out)
        assert result['status']=='VERIFIED' and len(result['cache_bindings'])==4
        assert result['gpu_use'] is False and result['cpu_lanes']==2
        assert set(reads)=={'manifest.json','features_complete.json','support_splits.json'}


def test_release_import_closure_help_without_encoder(tmp_path):
    # Future generated per-cohort JSONs are runtime inputs, not import dependencies.
    for name in publish.transport.PATHS:
        path=ROOT/name
        if name.startswith('configs/d92_registration_diagnostic_'):continue
        assert path.is_file(),name
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for tool in ('run_d92_registration_diagnostic.py','evaluate_d92_registration_diagnostic.py','summarize_d92_registration_diagnostic.py'):
        result=subprocess.run([sys.executable,'-s',str(tmp_path/'tools'/tool),'--help'],cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert '--' in result.stdout
