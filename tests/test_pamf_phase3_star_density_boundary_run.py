"""Two strict source students, fixed density and no-query test isolation."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_density_boundary_train import fixture as train_fixture
    from experiments.pamf_phase3_open_world import star_density_boundary_run as m
    with train_fixture(root,mp) as (t,tc,b,train_events,_):
        t.train(tc,'cpu');counts=dict(m.parent.COUNTS);mp.setattr(m,'COUNTS',counts);mp.setattr(m.best,'COUNTS',counts);mp.setattr(m,'Q_COUNT',m.parent.Q_COUNT)
        mp.setattr(m.source_eval,'COUNTS',counts);mp.setattr(m.source_eval.IQ,'COUNTS',counts)
        # Earlier synthetic raw-parent fixtures omit only the restored V/NV
        # files. Complete those using the same synthetic G and original IDs.
        for role in ('V','novel'):
            folder=Path(tc['reference_config']['train_config']['IQ_root'])/role;raw=np.load(folder/'received.npy')
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));manifest['code_commit']='a'*40;(folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with torch.no_grad():restored=b['dura'][1](torch.tensor(raw.tolist(),dtype=torch.float32))[0]
            np.save(folder/'restored.npy',np.asarray(restored.tolist(),np.float32))
            if role=='V':
                a=m.parent._rows(tc['reference_config']['output_root'],'known');np.save(folder/'baseline_closed.npy',np.stack((a['closed'],a['closed']),axis=1))
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],train_config=deepcopy(tc),output_root=str(root/'boundary_predict'),batch_size=128,Q_conditions=deepcopy(tc['reference_config']['Q_conditions']))
        events=[];source=m._iq_role;public=m._public;load=m._load_model;inputs={k:[] for k in m.ARMS};backend=m._backend
        def loaded(config,arm,device):
            model,identity=load(config,arm,device);events.append('load_'+arm);model.register_forward_pre_hook(lambda mod,args:inputs[arm].append(args[0].detach().clone()));return model,identity
        def role(config,name,identity):
            assert (Path(c['output_root'])/'smoke.json').is_file()
            assert (Path(c['output_root'])/'models/complete.json').is_file()
            if name=='novel':assert (Path(c['output_root'])/'calibration/complete.json').is_file()
            events.append(name);return source(config,name,identity)
        def query(config,name):
            assert (Path(c['output_root'])/'source/source_DURA/novel_predictions.npz').is_file();events.append(name);return public(config,name)
        def get_backend(config,device):
            result=backend(config,device);result['model'].register_forward_pre_hook(lambda mod,args:inputs['E0'].append(args[0].detach().clone()));return result
        mp.setattr(m,'_load_model',loaded);mp.setattr(m,'_iq_role',role);mp.setattr(m,'_public',query);mp.setattr(m,'_backend',get_backend)
        yield m,c,counts,events,inputs,b


def test_config_rejects_target_truth_or_mismatched_public_roles_before_neural_loading(monkeypatch,tmp_path,subtests):
    from experiments.pamf_phase3_open_world import star_density_boundary_run as m
    train=m._train();monkeypatch.setattr(train,'validate_config',lambda c:c)
    q=dict(clean_DURA=dict(public_root=str(tmp_path/'clean')),star_DURA=dict(public_root=str(tmp_path/'star')))
    tc=dict(run_id=m.RUN_ID,code_commit='a'*40,output_root=str(tmp_path/'train'),reference_config=dict(output_root=str(tmp_path/'ref'),train_config=dict(IQ_root=str(tmp_path/'IQ')),Q_conditions=q))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,train_config=tc,output_root=str(tmp_path/'new'),batch_size=128,Q_conditions=q)
    assert m.validate_config(c) is c
    for delta in (dict(U_truth='private'),dict(batch_size=64),dict(output_root=tc['output_root']),dict(Q_conditions={'star_DURA':q['star_DURA']})):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))


def test_existing_output_never_loads_checkpoints_or_repeats_predictions(monkeypatch,tmp_path):
    from experiments.pamf_phase3_open_world import star_density_boundary_run as m
    monkeypatch.setattr(m,'validate_config',lambda c:c);monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('No repeat'))
    with pytest.raises(FileExistsError):m.evaluate(dict(output_root=str(tmp_path)),'cpu')
