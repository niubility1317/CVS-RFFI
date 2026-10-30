from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_ssl_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_signal_ssl_train import fixture as train_fixture
    with train_fixture(root,mp) as (t,tc,pm,b,U,events,created):
        t.train(tc,'cpu');events.clear()
        training_roles={role:t.base.load_role(tc['source_train_config']['IQ_root'],role) for role in ('L','OE')}
        m=api();mp.setattr(m,'COUNTS',dict(m.best.COUNTS));mp.setattr(m,'Q_COUNT',8)
        mp.setattr(m.source_eval,'COUNTS',dict(m.COUNTS));mp.setattr(m.source_eval.IQ,'COUNTS',dict(m.COUNTS))
        # Historical raw-parent fixtures had no restored V/NV files. Complete
        # the synthetic P1 cache using the same toy G, preserving original IDs.
        for role in ('V','novel'):
            folder=Path(tc['source_train_config']['IQ_root'])/role;raw=np.load(folder/'received.npy')
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));manifest['code_commit']='a'*40
            (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with torch.no_grad():restored=b['dura'][1](torch.tensor(raw.tolist(),dtype=torch.float32))[0]
            np.save(folder/'restored.npy',np.asarray(restored.tolist(),np.float32))
            if role=='V':
                a=m.best._source_rows(tc['reference_prediction_root'],'source_DURA','known')
                np.save(folder/'baseline_closed.npy',np.stack((a['closed'],a['closed']),axis=1))
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=deepcopy(tc),output_root=str(root/'ssl_predict'),batch_size=128,Q_conditions=deepcopy(pm['config']['Q_conditions']))
        events.clear();rows=m._iq_role;opened=m._public;loaded=m._load_model;inputs={'SOURCE':[],'TARGET_SSL':[],'F0':[],'DA0':[]}
        def load(config,arm,device):
            model,identity=loaded(config,arm,device);events.append('load_'+arm)
            model.register_forward_pre_hook(lambda mod,args:inputs[arm].append(args[0].detach().clone()))
            return model,identity
        def source(config,role,parent):
            assert (Path(c['output_root'])/'smoke.json').is_file()
            if role=='V':assert (Path(c['output_root'])/'models/complete.json').is_file()
            if role=='novel':assert (Path(c['output_root'])/'calibration/complete.json').is_file()
            events.append(role)
            return training_roles[role] if role in ('L','OE') else rows(config,role,parent)
        def query(config,name):
            assert (Path(c['output_root'])/'source/source_DURA/novel_predictions.npz').is_file()
            events.append(name);return m.best._public(pm['config'],name)
        backend=m._backend
        def load_backend(config,device):
            result=backend(config,device)
            result['model'].register_forward_pre_hook(lambda mod,args:inputs['DA0'].append(args[0].detach().clone()))
            result['reader'].core.id_backbone.register_forward_pre_hook(lambda mod,args:inputs['F0'].append(args[0].detach().clone()))
            return result
        mp.setattr(m,'_backend',load_backend);mp.setattr(m,'_load_model',load);mp.setattr(m,'_iq_role',source);mp.setattr(m,'_public',query)
        yield m,c,pm,b,events,inputs,U


def test_two_strict_students_source_fit_beforeV_allthree_DURA_and_exact_source_controls(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,inputs,U):
        calibration=m.risk.calibrate_known
        def source_failure(*args):
            state,report=calibration(*args);report['constraints_passed']=False;return state,report
        monkeypatch.setattr(m.risk,'calibrate_known',source_failure)
        done=m.evaluate(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','load_SOURCE','load_TARGET_SSL','L','OE','V','novel','clean_DURA','clean_DURA','star_DURA','star_DURA']
        assert done['all_three_DURA_conditions_complete'] and done['classification_unchanged']
        assert done['source_density_models_fitted']==2 and done['new_threshold_sets_fitted']==4 and done['private_truth_reads']==0
        assert done['real_signal_forward_rows']==2*sum(m.COUNTS.values())+6*m.Q_COUNT
        assert done['G_forward_rows']==done['real_F_forward_rows']==2*m.Q_COUNT
        with np.load(out/'thresholds.npz') as z:tau=z['conditional']
        with np.load(Path(c['train_config']['reference_prediction_root'])/'thresholds.npz') as z:np.testing.assert_array_equal(tau[:2],z['conditional'])
        for role in ('known','novel'):
            a=m._source_rows(out,'source_DURA',role);old=m.best._source_rows(c['train_config']['reference_prediction_root'],'source_DURA',role)
            for key,v in old.items():
                actual=a[key][:,:2] if key in ('raw_scores','scores') else a[key][:,0] if key=='signal_features' else a[key]
                np.testing.assert_array_equal(actual,v)
        for condition in m.CONDITIONS[1:]:
            a=m._target_rows(out,condition);np.testing.assert_array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)
            assert a['signal_features'].shape==(8,3,160) and a['baseline_closed'].tolist()==a['closed'].tolist()
        for key in inputs:
            for actual,wanted in zip(inputs[key][-2:],b['dura'][0].outputs[-2:]):torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
        with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_incomplete_neural_training_never_opens_source_or_query(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,inputs,U):
        Path(c['train_config']['output_root'],'complete.json').unlink()
        with pytest.raises((ValueError,FileNotFoundError)):m.evaluate(c,'cpu')
        assert all(x not in events for x in ('L','OE','V','novel','clean_DURA','star_DURA'))
        assert not Path(c['output_root'],'complete.json').exists()


def test_invalid_scope_and_partial_query_failure_never_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events,inputs,U):
        for delta in (dict(truth='private'),dict(output_root=c['train_config']['output_root']),dict(batch_size=64),dict(Q_conditions={'star_DURA':c['Q_conditions']['star_DURA']})):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        opened=m._public
        def failed(config,name):
            if name=='star_DURA':raise ValueError('synthetic query failure')
            return opened(config,name)
        monkeypatch.setattr(m,'_public',failed)
        with pytest.raises(ValueError,match='synthetic query failure'):m.evaluate(c,'cpu')
        assert Path(c['output_root'],'targets/clean_DURA/predictions.npz').is_file()
        assert not Path(c['output_root'],'complete.json').exists()
