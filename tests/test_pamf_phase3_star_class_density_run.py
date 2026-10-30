"""Three DURA tests with a real synthetic r03 signal checkpoint, no real IQ."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_class_density_run')
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
    with parent_fixture(root,mp) as (raw,rc,counts,_,_,_):
        m=api();mp.setattr(m,'COUNTS',dict(counts));mp.setattr(m,'Q_COUNT',8)
        # Give the synthetic completed raw parent its fixed code identity.
        tc=rc['train_config'];tc['code_commit']=m.REFERENCE_CODE;tc['native_config']['code_commit']=m.REFERENCE_CODE;rc['code_commit']=m.REFERENCE_CODE
        cp=Path(tc['output_root'])/'final.pt';payload=torch.load(cp,weights_only=False)
        payload['config']=deepcopy(tc);payload['identity']['code_commit']=m.REFERENCE_CODE;payload['teacher_provenance']['code_commit']=m.REFERENCE_CODE;torch.save(payload,cp)
        done=raw._train()._read(cp.parent/'complete.json');done.update(config=deepcopy(tc),code_commit=m.REFERENCE_CODE,identity=payload['identity'],teacher_provenance=payload['teacher_provenance']);write(cp.parent/'complete.json',done)
        pm=raw.evaluate(rc,'cpu');p1=deepcopy(tc['control_train_config'])
        p1payload=torch.load(Path(p1['output_root'])/'final.pt',weights_only=False)
        def native(config,device):
            model,_=return_aux_source();model._phase3_parent_identity=deepcopy(p1payload['original_parent_identity']);model._phase3_source_handle=deepcopy(p1payload['source_data_handle'])
            return model.to(device)
        mp.setattr(m.signal_train.native,'make_trainable_model',native)
        # A P0 checkpoint is deliberately unavailable to this producer.
        cp.unlink();events=[];backend={};seen=dict(F0=[],signal=[])
        def load(c,device):
            assert c['train_config']==p1;events.append('checkpoint')
            model,identity=m.signal_train.load_trained_model(p1,Path(p1['output_root'])/'final.pt',device)
            original,_=return_aux_source();reader=original.id_backbone.encoder.eval().requires_grad_(False);g=ToyDURA().eval().requires_grad_(False)
            # reader.features directly calls core.features, then the backbone.
            reader.core.id_backbone.register_forward_pre_hook(lambda mod,args:seen['F0'].append(args[0].detach().clone()))
            model.register_forward_pre_hook(lambda mod,args:seen['signal'].append(args[0].detach().clone()))
            backend.update(model=model,model_identity=identity,reader=reader,reader_identity=identity['parent_F0_identity'],dura=(g,lambda x:(g(x),)),dura_identity=identity['dura_identity'])
            return backend
        mp.setattr(m,'_backend',load)
        rng=np.random.default_rng(31);public={};conditions={}
        for name,family in (('clean_DURA','clean_cross_receiver'),('star_DURA','practical_cross_receiver')):
            folder=root/name/'evaluation/public';conditions[name]=dict(public_root=str(folder),capsule_id=name+'cap',split_id=name+'split',expected_count=8,view_family=family)
            public[name]=(dict(capsule_id=name+'cap',split_id=name+'split',count=8,registered_classes=m.signal_train.native.CLASSES),
                np.asarray([name+'-'+str(i) for i in range(8)]),rng.normal(size=(8,2,256)).astype(np.float32))
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=p1,reference_prediction_root=rc['output_root'],output_root=str(root/'class_density'),
            batch_size=128,Q_conditions=conditions,dura_checkpoint='frozen-R69.pt',dura_release=str(root/'G_release'))
        rows=m._rows
        def source_rows(path,role):
            if role=='novel':assert (Path(c['output_root'])/'calibration/complete.json').is_file()
            events.append(role);return rows(path,role)
        def opened(config,name):
            assert all((Path(c['output_root'])/p).exists() for p in ('calibration/complete.json','thresholds.npz','known_predictions.npz','novel_predictions.npz'))
            events.append(name);return public[name]
        mp.setattr(m,'_rows',source_rows);mp.setattr(m,'_public',opened)
        yield m,c,pm,backend,public,events,seen


def test_actual_r03_checkpoint_all_three_DURA_conditions_shared_tau_and_exact_source_controls(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen):
        monkeypatch.setattr(m.reference._train(),'load_trained_model',lambda *a:pytest.fail('Never load P0 checkpoint'))
        monkeypatch.setattr(m.core.density,'fit_known',lambda *a:pytest.fail('No source fitting'))
        monkeypatch.setattr(m.core.density,'fit_background',lambda *a:pytest.fail('No source fitting'))
        calibrate=m.risk.calibrate_known
        def source_failure(*a):
            state,report=calibrate(*a);report['constraints_passed']=False;return state,report
        monkeypatch.setattr(m.risk,'calibrate_known',source_failure)
        result=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','known','novel','clean_DURA','star_DURA']
        assert result['model_identity']==pm['signal_reference_identity']['model_identity'] and result['model_identity']!=pm['model_identity']
        assert result['all_three_conditions_complete'] and result['all_conditions_through_DURA']
        assert result['real_signal_forward_rows']==result['real_F_forward_rows']==result['G_forward_rows']==16
        assert result['synthetic_signal_forward_rows']==result['synthetic_F_forward_rows']==result['G_synthetic_forward_rows']==2
        assert len(b['dura'][0].outputs)==3 and len(seen['signal'])==len(seen['F0'])==3
        for restored,f0,signal in zip(b['dura'][0].outputs,seen['F0'],seen['signal']):
            torch.testing.assert_close(f0,restored,atol=0,rtol=0);torch.testing.assert_close(signal,restored,atol=0,rtol=0)
        with np.load(out/'thresholds.npz') as z:tau=z['conditional']
        for name in m.CONDITIONS:
            a=m._target_rows(out,name);np.testing.assert_array_equal(a['ids'],public[name][1]);np.testing.assert_array_equal(a['baseline_closed'],a['closed'])
            np.testing.assert_array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)
            assert 'y' not in a and 'scenario' not in a
        for role in ('known','novel'):
            with np.load(out/(role+'_predictions.npz')) as z:a={k:z[k] for k in z.files}
            with np.load(Path(c['reference_prediction_root'])/(role+'_predictions.npz')) as z:parent={k:z[k] for k in z.files}
            for key,value in parent.items():np.testing.assert_array_equal(a[key][:,:2] if key in ('scores','raw_scores') else a[key],value[:,:2] if key in ('scores','raw_scores') else value)
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_config_rejects_truth_missing_DURA_and_P0_theta_before_public(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen):
        for delta in (dict(truth='private'),dict(dura_checkpoint=None),dict(train_config=pm['config']['train_config']),dict(Q_conditions=dict(c['Q_conditions'],clean_DURA=dict(c['Q_conditions']['clean_DURA'],view_family='practical_cross_receiver')))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        backend=m._backend
        def wrong(*a):
            result=backend(*a);result['model_identity']=pm['model_identity'];return result
        monkeypatch.setattr(m,'_backend',wrong)
        with pytest.raises(ValueError):m.run(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root'],'complete.json').exists()


def test_source_scientific_failure_still_predicts_queries_and_partial_query_failure_never_completes(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,public,events,seen):
        calibrate=m.risk.calibrate_known;opened=m._public
        def failed_source(*a):
            state,report=calibrate(*a);report['constraints_passed']=False;return state,report
        def fail_star(config,name):
            if name=='star_DURA':events.append('star_failure');raise ValueError('synthetic public error')
            return opened(config,name)
        monkeypatch.setattr(m.risk,'calibrate_known',failed_source);monkeypatch.setattr(m,'_public',fail_star)
        with pytest.raises(ValueError,match='public error'):m.run(c,'cpu')
        assert events[-2:]==['clean_DURA','star_failure'] and Path(c['output_root'],'targets/clean_DURA/predictions.npz').exists()
        assert not Path(c['output_root'],'complete.json').exists()
