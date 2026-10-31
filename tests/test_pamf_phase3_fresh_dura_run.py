"""Frozen fresh-PCD raw paths preserve all original DURA controls."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.fresh_dura_run')


def test_raw_identity_and_density_use_the_actual_input_class(monkeypatch):
    m=api()
    class Reader(torch.nn.Module):
        def features(self,x):return x[:,0,0,None].repeat(1,160)
        def classify_features(self,f):return torch.stack((f[:,0],-f[:,0],*([torch.zeros_like(f[:,0])]*4)),1)
    class Signal(torch.nn.Module):
        def forward(self,x):return torch.nn.functional.normalize(torch.cat((x[:,0,:80],x[:,1,:80]),1),dim=1)
    b=dict(reader=Reader(),dura=(torch.nn.Identity(),lambda x:(-x,)));models={a:Signal() for a in m.MODEL_NAMES}
    x=np.ones((3,2,256),np.float32);raw=m._current(b,models,x,'cpu',False);p1=m._current(b,models,x,'cpu',True)
    np.testing.assert_array_equal(raw['closed'],0);np.testing.assert_array_equal(p1['closed'],1)
    np.testing.assert_array_equal(raw['baseline_closed'],raw['closed'])
    seen=[]
    def scores(state,z,closed):seen.append(closed.copy());return np.column_stack((closed+state,closed+state+1)).astype(np.float64)
    monkeypatch.setattr(m.fresh,'_raw',scores)
    a=m._raw(dict(E0=10,PCD=20),raw['signal_features'],raw['closed'])
    np.testing.assert_array_equal(a,np.tile([10.,11.,20.,21.],(3,1)))
    assert len(seen)==2 and all(np.array_equal(v,raw['closed']) for v in seen)


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as inherited
    with inherited(root,mp) as (old,r,_,b,events,identity):
        m=api();r['code_commit']=m.REFERENCE_CODE;r['train_config']['code_commit']=m.REFERENCE_CODE;identity['code_commit']=m.REFERENCE_CODE
        parent=old.run(r,'cpu');pcd,_=old._load_model(r,'cpu')
        mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',old.Q_COUNT)
        mp.setattr(m.received,'ROLE_COUNTS',dict(old.COUNTS));mp.setattr(m.received.IQ,'COUNTS',dict(old.COUNTS))
        source=Path(r['source_control_config']['reference_config']['train_config']['IQ_root'])
        for role in ('V','novel'):
            p1=np.load(source/role/'restored.npy',allow_pickle=False)
            np.save(source/role/'received.npy',(-.8*p1+.07).astype(np.float32))
        c=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=deepcopy(r),Q_conditions=deepcopy(r['Q_conditions']),output_root=str(root/'fresh_dura'),batch_size=128)
        mp.setattr(m,'_backend',lambda *a:b);mp.setattr(m,'_load_model',lambda *a:(pcd,identity))
        public=old._public;calls=[];role_loader=m._source_role
        def role(config,name):
            assert Path(c['output_root'],'smoke.json').is_file() and Path(c['output_root'],'thresholds.npz').is_file()
            calls.append(name);return role_loader(config,name)
        def query(config,name):
            assert Path(c['output_root'],'source/source_raw/novel_predictions.npz').is_file()
            calls.append(name);return public(r,name)
        mp.setattr(m,'_source_role',role);mp.setattr(m,'_public',query)
        mp.setattr(old,'_training_metadata',lambda *a:pytest.fail('No new U IDs or U cache access'))
        yield m,c,parent,b,identity,calls


def test_actual_raw_producer_preserves_dura_and_never_reads_u_or_fits(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,parent,b,identity,calls):
        for change in (dict(truth='PRIVATE'),dict(batch_size=True),dict(output_root=c['reference_config']['output_root']),
                       dict(Q_conditions=dict(c['Q_conditions'],star_DURA=dict(c['Q_conditions']['star_DURA'],capsule_id='wrong')))):
            with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
        backend=m._backend
        monkeypatch.setattr(m,'_backend',lambda *a:dict(backend(*a),reader_identity={'wrong':'F0'}))
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert calls==[] and not Path(c['output_root']).exists()
        monkeypatch.setattr(m,'_backend',backend)
        for name in ('fit_known','fit_background'):monkeypatch.setattr(m.density,name,lambda *a:pytest.fail('No new density fit'))
        monkeypatch.setattr(m.fresh.risk,'calibrate_known',lambda *a:pytest.fail('No new thresholds'))
        load=np.load;opened=[]
        def guarded(path,*a,**kw):
            opened.append(str(path));assert Path(path).name not in ('U_features.npz','restored.npy','teacher_logits.npy','baseline_closed.npy')
            return load(path,*a,**kw)
        monkeypatch.setattr(np,'load',guarded)
        g_rows=[];h=b['dura'][0].register_forward_pre_hook(lambda module,args:g_rows.append(len(args[0])))
        try:done=m.run(c,'cpu')
        finally:h.remove()
        n=m.COUNTS['V']+m.COUNTS['novel']+2*m.Q_COUNT
        assert done['real_F_forward_rows']==n and done['real_signal_forward_rows']==2*n and done['G_forward_rows']==0
        assert sum(g_rows)==2 and done['synthetic_F_forward_rows']==4 and done['synthetic_signal_forward_rows']==8
        assert done['all_six_paths_complete'] and done['all_DURA_controls_reused_exactly'] and done['U_ID_metadata_reads']==done['target_U_reads']==0
        assert calls==['V','novel','clean_DURA','star_DURA'] and done['source_L_reads']==done['source_OE_reads']==0
        out=Path(c['output_root']);_,states,tau,cal=m._reference(c)
        assert tau.shape==(4,6) and tau.nbytes==192 and m._read(out/'calibration/complete.json')==cal
        for role in ('known','novel'):
            old=m._select(m.fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role))
            dura=m._source_rows(out,'source_DURA',role);raw=m._source_rows(out,'source_raw',role)
            for key in old:np.testing.assert_array_equal(dura[key],old[key])
            for key in ('ids','rx_i','day_i'):np.testing.assert_array_equal(raw[key],dura[key])
            if 'baseline_closed' in raw:np.testing.assert_array_equal(raw['baseline_closed'],raw['closed'])
            np.testing.assert_array_equal(raw['scores'],m._raw(states,raw['signal_features'],raw['closed'])-tau[:,raw['closed']].T)
        for condition in ('clean_DURA','star_DURA'):
            old=m._select(m.fresh._target_rows(c['reference_config']['output_root'],condition));dura=m._target_rows(out,condition)
            for key in old:np.testing.assert_array_equal(dura[key],old[key])
            raw=m._target_rows(out,condition.replace('_DURA','_raw'));np.testing.assert_array_equal(raw['ids'],dura['ids'])
            np.testing.assert_array_equal(raw['baseline_closed'],raw['closed'])
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_partial_second_query_preserves_outputs_without_complete(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        public=m._public
        def interrupted(config,name):
            if name=='star_DURA':raise ValueError('synthetic second query failure')
            return public(config,name)
        monkeypatch.setattr(m,'_public',interrupted)
        with pytest.raises(ValueError,match='second query'):m.run(c,'cpu')
        assert Path(c['output_root'],'targets/clean_raw/predictions.npz').exists()
        assert not Path(c['output_root'],'complete.json').exists()
