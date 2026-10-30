"""Fixed P1 checkpoints/stats/thresholds, six synthetic paired input paths."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_dura_best_run')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')


class ToyDURA(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.7));self.padding=nn.Parameter(torch.zeros(266950));self.outputs=[]
    def forward(self,x):
        value=x.flip(-1)*self.scale+.03;self.outputs.append(value.detach().clone());return value


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_raw_open_eval import fixture as parent_fixture
    from test_pamf_phase3_star_signal_open_model import return_aux_source
    with parent_fixture(root,mp) as (raw,rc,counts,_,_,values):
        m=api();mp.setattr(m,'COUNTS',dict(counts));mp.setattr(m,'Q_COUNT',8)
        tc=rc['train_config'];tc['code_commit']=m.REFERENCE_CODE;tc['native_config']['code_commit']=m.REFERENCE_CODE;rc['code_commit']=m.REFERENCE_CODE
        cp=Path(tc['output_root'])/'final.pt';payload=torch.load(cp,weights_only=False)
        payload['config']=deepcopy(tc);payload['identity']['code_commit']=m.REFERENCE_CODE;payload['teacher_provenance']['code_commit']=m.REFERENCE_CODE;torch.save(payload,cp)
        done=raw._train()._read(cp.parent/'complete.json');done.update(config=deepcopy(tc),code_commit=m.REFERENCE_CODE,identity=payload['identity'],teacher_provenance=payload['teacher_provenance']);write(cp.parent/'complete.json',done)
        pm=raw.evaluate(rc,'cpu');p1=deepcopy(tc['control_train_config'])
        p1payload=torch.load(Path(p1['output_root'])/'final.pt',weights_only=False)
        def native(config,device):
            model,_=return_aux_source();model._phase3_parent_identity=deepcopy(p1payload['original_parent_identity']);model._phase3_source_handle=deepcopy(p1payload['source_data_handle']);return model.to(device)
        mp.setattr(m.signal_train.native,'make_trainable_model',native)
        cp.unlink();events=[];backend={};seen=dict(F0=[],signal=[])
        def load(c,device):
            assert c['train_config']==p1;events.append('checkpoint')
            model,identity=m.signal_train.load_trained_model(p1,Path(p1['output_root'])/'final.pt',device)
            original,_=return_aux_source();reader=original.id_backbone.encoder.eval().requires_grad_(False);g=ToyDURA().eval().requires_grad_(False)
            reader.core.id_backbone.register_forward_pre_hook(lambda mod,args:seen['F0'].append(args[0].detach().clone()))
            model.register_forward_pre_hook(lambda mod,args:seen['signal'].append(args[0].detach().clone()))
            backend.update(model=model,model_identity=identity,reader=reader,reader_identity=identity['parent_F0_identity'],dura=(g,lambda x:(g(x),)),dura_identity=identity['dura_identity']);return backend
        mp.setattr(m,'_backend',load)
        rng=np.random.default_rng(31);public={};conditions={}
        for name,family in (('clean_DURA','clean_cross_receiver'),('star_DURA','practical_cross_receiver')):
            folder=root/name/'evaluation/public';conditions[name]=dict(public_root=str(folder),capsule_id=name+'cap',split_id=name+'split',expected_count=8,view_family=family)
            public[name]=(dict(capsule_id=name+'cap',split_id=name+'split',count=8,registered_classes=m.signal_train.native.CLASSES),
                np.asarray([name+'-'+str(i) for i in range(8)]),rng.normal(size=(8,2,256)).astype(np.float32))
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=p1,reference_prediction_root=rc['output_root'],source_IQ_root=p1['IQ_root'],output_root=str(root/'dura_best'),
            batch_size=128,Q_conditions=conditions,dura_checkpoint='frozen-R69.pt',dura_release=str(root/'G_release'))
        rows=m._rows;role=m._source_role
        def source_rows(path,name):
            assert (Path(c['output_root'])/'models/complete.json').is_file() and (Path(c['output_root'])/'thresholds.npz').is_file()
            events.append('parent_'+name);return rows(path,name)
        def source_role(config,name):
            assert events[0]=='checkpoint' and len(backend['dura'][0].outputs)==1
            events.append(name);return role(config,name)
        def opened(config,name):
            assert all((Path(c['output_root'])/'source'/condition/'novel_predictions.npz').is_file() for condition in m.SOURCE_CONDITIONS)
            events.append(name);return public[name]
        mp.setattr(m,'_rows',source_rows);mp.setattr(m,'_source_role',source_role);mp.setattr(m,'_public',opened)
        yield m,c,pm,backend,public,events,seen,values


def test_actual_P1_checkpoint_all_six_paths_exact_frozen_controls_and_same_input_baselines(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen,values):
        monkeypatch.setattr(m.reference._train(),'load_trained_model',lambda *a:pytest.fail('Never load P0 checkpoint'))
        for name in ('fit_known','fit_background'):monkeypatch.setattr(m.density,name,lambda *a:pytest.fail('No new statistics fit'))
        for module in (m.reference.risk,m.signal_eval.risk):monkeypatch.setattr(module,'calibrate_known',lambda *a:pytest.fail('No threshold fit'))
        load=np.load;as_tensor=torch.as_tensor
        def safe(path,*a,**kw):
            assert Path(path).name not in ('restored.npy','teacher_logits.npy','baseline_closed.npy');return load(path,*a,**kw)
        def bridge(value,*a,**kw):
            if isinstance(value,np.ndarray):pytest.fail('No NumPy ABI bridge')
            return as_tensor(value,*a,**kw)
        monkeypatch.setattr(np,'load',safe);monkeypatch.setattr(torch,'as_tensor',bridge);monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI bridge'))
        result=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','parent_known','V','parent_novel','novel','clean_DURA','star_DURA']
        assert result['model_identity']==pm['signal_reference_identity']['model_identity'] and result['model_identity']!=pm['model_identity']
        assert result['all_six_paths_complete'] and result['all_DURA_main_conditions_complete']
        assert result['real_signal_forward_rows']==result['real_F_forward_rows']==270+32
        assert result['G_forward_rows']==16 and result['source_DURA_rows_reused']==270
        assert result['synthetic_signal_forward_rows']==result['synthetic_F_forward_rows']==result['G_synthetic_forward_rows']==2
        with load(out/'thresholds.npz') as z:tau=z['conditional']
        with load(Path(c['reference_prediction_root'])/'thresholds.npz') as z:np.testing.assert_array_equal(tau,z['conditional'][:2])
        assert m._read(out/'calibration/complete.json')==m._read(Path(c['reference_prediction_root'])/'calibration/complete.json')
        for role in ('known','novel'):
            old=m._rows(c['reference_prediction_root'],role);copied=m._source_rows(out,'source_DURA',role);raw=m._source_rows(out,'source_raw',role)
            for k,v in old.items():np.testing.assert_array_equal(copied[k],v)
            for k in ('ids','rx_i','day_i'):np.testing.assert_array_equal(raw[k],copied[k])
            assert ('baseline_closed' in raw)==(role=='known')
            if role=='known':np.testing.assert_array_equal(raw['baseline_closed'],raw['closed'])
            np.testing.assert_array_equal(raw['scores'],raw['raw_scores']-tau[:,raw['closed']].T)
        offset=1
        for role in ('V','novel'):
            n=len(values[role]);chunks=(n+127)//128
            torch.testing.assert_close(torch.cat(seen['F0'][offset:offset+chunks]),torch.tensor(values[role].tolist()),atol=0,rtol=0);offset+=chunks
        for key in m.QUERY_CONDITIONS:
            raw=m._target_rows(out,key.replace('_DURA','_raw'));restored=m._target_rows(out,key)
            for a in (raw,restored):
                np.testing.assert_array_equal(a['ids'],public[key][1]);np.testing.assert_array_equal(a['baseline_closed'],a['closed'])
                np.testing.assert_array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)
            torch.testing.assert_close(seen['F0'][offset],torch.tensor(public[key][2].tolist()),atol=0,rtol=0)
            torch.testing.assert_close(seen['F0'][offset+1],b['dura'][0].outputs[1+(offset-4)//2],atol=0,rtol=0);offset+=2
        assert len(b['dura'][0].outputs)==3
        for f0,signal in zip(seen['F0'],seen['signal']):torch.testing.assert_close(f0,signal,atol=0,rtol=0)
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_config_and_wrong_P0_identity_fail_before_new_values(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen,values):
        for delta in (dict(truth='private'),dict(dura_checkpoint=None),dict(source_IQ_root='different'),dict(train_config=pm['config']['train_config']),dict(output_root=c['source_IQ_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        backend=m._backend
        def wrong(*a):
            result=backend(*a);result['model_identity']=pm['model_identity'];return result
        monkeypatch.setattr(m,'_backend',wrong)
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root'],'complete.json').is_file()


def test_partial_second_query_preserves_first_paths_without_complete(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen,values):
        opened=m._public
        def failed(config,name):
            if name=='star_DURA':events.append('star_failure');raise ValueError('synthetic public error')
            return opened(config,name)
        monkeypatch.setattr(m,'_public',failed)
        with pytest.raises(ValueError,match='public error'):m.run(c,'cpu')
        assert events[-2:]==['clean_DURA','star_failure']
        assert Path(c['output_root'],'targets/clean_raw/predictions.npz').is_file() and Path(c['output_root'],'targets/clean_DURA/predictions.npz').is_file()
        assert not Path(c['output_root'],'complete.json').is_file()
