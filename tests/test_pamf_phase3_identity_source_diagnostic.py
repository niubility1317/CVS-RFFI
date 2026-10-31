"""Source-only native identity diagnostics use one cached-P1 pass per model."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.identity_source_diagnostic')


def fixture(root,mp):
    from test_pamf_phase3_star_signal_open_model import return_aux_source
    from experiments.pamf_phase3_open_world import star_signal_open_model as signal,joint_identity_model as joint
    from test_pamf_phase3_star_class_density_run import ToyDURA
    m=api();torch.manual_seed(2026092701);native,_=return_aux_source();reader=native.id_backbone.encoder.eval().requires_grad_(False)
    parent=signal.SignalOpen(signal.HeadlessSignal(reader)).eval()
    with torch.no_grad():parent.projection.network[-1].bias.add_(.01)
    parent.requires_grad_(False);source=joint.JointIdentity(parent,reader).eval()
    with torch.no_grad():
        source.signal.encoder.core.id_backbone.cls_head.weight.copy_(source.signal.encoder.core.id_backbone.cls_head.weight.roll(1,0))
        source.projection.network[-1].bias.add_(.005)
    source.requires_grad_(False)
    F0=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
    G=dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940);r03=dict(run_id='r03',source_only=True,parent_F0_identity=F0,dura_identity=G)
    dura=ToyDURA().eval().requires_grad_(False);b=dict(reader=reader,reader_identity=F0,model=parent,model_identity=r03,dura=(dura,lambda x:(dura(x),)),dura_identity=G)
    identity=dict(run_id=m.TRAIN_RUN_ID,code_commit=m.TRAIN_CODE,arm='SOURCE_JOINT',updates=444,identity_head_updates=444,signal_parameters=335681,
        source_only=True,target_U_used=False,target_U_reads=0,identity_head_jointly_trained=True,identity_head_replaced=False,Sinc_frozen=True,
        parent_signal_identity=r03,parent_F0_identity=F0,dura_identity=G)
    pc=dict(IQ_root=str(root/'cache'),native_config=dict(source_output='original-F0',source_contract='source-contract'),output_root=str(root/'r03'))
    ref=dict(train_config=pc);old=dict(reference_config=ref);tc=dict(run_id=m.TRAIN_RUN_ID,code_commit=m.TRAIN_CODE,reference_config=dict(reference_config=old),output_root=str(root/'joint'),steps=444,lr=.0001)
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=tc,output_root=str(root/'diagnostic'),batch_size=128)
    mp.setattr(m.training,'validate_config',lambda a:a);mp.setattr(m,'_backend',lambda *a:b);loads=[]
    def load(config,arm,device):
        assert arm=='SOURCE_JOINT';loads.append(arm);return source,identity
    mp.setattr(m.training,'load_trained_model',load);mp.setattr(m,'COUNTS',dict(L=360,OE=90))
    rng=np.random.default_rng(51)
    for role in ('L','OE'):
        groups=[(y,r,d) for y in range(6) for r in (1,3,4,6,8) for d in (1,2,3)] if role=='L' else [(0,r,d) for r in (1,3,4,6,8) for d in (1,2,3)]
        groups=groups*(4 if role=='L' else 6);n=len(groups);folder=root/'cache'/role;folder.mkdir(parents=True)
        meta=dict(rx_i=np.asarray([x[1] for x in groups],np.int64),day_i=np.asarray([x[2] for x in groups],np.int64))
        meta['domain']=np.searchsorted([1,3,4,6,8],meta['rx_i'])*3+meta['day_i']-1
        if role=='L':meta['y']=np.asarray([x[0] for x in groups],np.int64)
        manifest=dict(schema=m.IQ.ROLE_SCHEMA,status='STAR_REPRESENTATION_ROLE_COMPLETE',role=role,count=n,iq_shape=[n,2,256],source_only=True,restored_input_path='P1',parent_F0_identity=F0,dura_identity=G)
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');np.save(folder/'ids.npy',np.asarray([role+'-'+str(i) for i in range(n)]));np.savez(folder/'metadata.npz',**meta)
        np.save(folder/'restored.npy',rng.normal(size=(n,2,256)).astype(np.float32))
    return m,c,b,source,identity,loads


def test_config_permissions_source_checkpoint_and_no_source_before_smoke(tmp_path,monkeypatch,subtests):
    m,c,b,source,identity,loads=fixture(tmp_path,monkeypatch)
    for change in (dict(truth='private'),dict(Q_public_root='query'),dict(batch_size=True),dict(train_config=dict(c['train_config'],code_commit='b'*40))):
        with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    identity['target_U_reads']=576
    monkeypatch.setattr(m,'_load_role',lambda *a:pytest.fail('Bad SOURCE checkpoint must fail before IQ'))
    with pytest.raises(ValueError):m.diagnose(c,'cpu')
    identity['target_U_reads']=0
    monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke failed')))
    with pytest.raises(ValueError,match='smoke failed'):m.diagnose(c,'cpu')
    assert not Path(c['output_root']).exists() and set(loads)=={'SOURCE_JOINT'}


def test_actual_native_same_pass_initial_W_and_frozen_readonly_source_only(tmp_path,monkeypatch):
    m,c,b,source,identity,loads=fixture(tmp_path,monkeypatch);open_role=m._load_role;opened=[];np_load=np.load
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('Diagnostic must not create an optimizer'))
    monkeypatch.setattr(m.training,'train',lambda *a,**kw:pytest.fail('Diagnostic must not train'))
    def only_P1(path,*a,**kw):
        p=Path(path)
        assert p.name not in ('received.npy','teacher_logits.npy','baseline_closed.npy') and not any(x in p.parts for x in ('V','novel','U','targets','truth'))
        return np_load(path,*a,**kw)
    monkeypatch.setattr(np,'load',only_P1)
    def role(root,name,backend):
        assert Path(c['output_root'],'smoke.json').exists();opened.append(name);a=open_role(root,name,backend);assert not a['restored'].flags.writeable;return a
    monkeypatch.setattr(m,'_load_role',role)
    # Count the real native core; same-pass classifying must not encode twice.
    counters={};encode=m._encode
    def tracked(reader,init,trained,values,device):
        refs=(reader,init.signal.encoder,trained.signal.encoder);before=[r.core.id_backbone.calls for r in refs]
        result=encode(reader,init,trained,values,device)
        assert [r.core.id_backbone.calls-a for r,a in zip(refs,before)]==[1,1,1]
        counters['batches']=counters.get('batches',0)+1;return result
    monkeypatch.setattr(m,'_encode',tracked)
    before={k:v.detach().clone() for k,v in source.state_dict().items()};done=m.diagnose(c,'cpu')
    assert opened==['L','OE'] and loads==['SOURCE_JOINT'] and counters['batches']==4
    assert done['status']=='IDENTITY_SOURCE_DIAGNOSTIC_COMPLETE' and done['source_only']
    assert done['INIT_W_exact_F0'] and done['INIT_matches_r03_except_native_W'] and done['model_parameters_buffers_unchanged']
    assert done['real_F_forward_rows']==450 and done['real_signal_forward_rows']==900 and done['G_forward_rows']==0
    assert done['target_reads']==done['validation_reads']==done['private_truth_reads']==done['fit_updates']==done['optimizer_updates']==0
    assert m._read(Path(c['output_root'])/'complete.json')==done
    summary=m._read(Path(c['output_root'])/'summary.json');assert summary['L']['by_scene'] is None and summary['L']['scene_status']=='metadata_missing'
    assert set(summary['L']['models'])=={'F0','INIT','SOURCE'} and 'accuracy' not in summary['OE']
    assert summary['W']['INIT_vs_F0']['exact'] and not summary['W']['SOURCE_vs_F0']['exact']
    for k,v in source.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with np_load(Path(c['output_root'])/'OE_rows.npz',allow_pickle=False) as z:
        assert 'y' not in z.files and z['logits'].shape==(90,3,6) and z['closed'].shape==(90,3) and not any('IQ' in k for k in z.files)
    with pytest.raises(FileExistsError):m.diagnose(c,'cpu')


def test_role_metadata_disallows_OE_TX_and_preserves_failure(tmp_path,monkeypatch,subtests):
    m,c,b,*_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):m._load_role(tmp_path/'cache','V',b)
    path=tmp_path/'cache/OE/metadata.npz';a=dict(np.load(path));a['y']=np.zeros(90,np.int64);np.savez(path,**a)
    with pytest.raises(ValueError):m.diagnose(c,'cpu')
    assert not Path(c['output_root'],'complete.json').exists()
