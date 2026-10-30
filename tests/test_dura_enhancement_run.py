from types import SimpleNamespace
import copy
import pytest
import torch
from experiments.pamf_dura_direct_train import enhanced as trial


def test_E30_routes_and_permissions(monkeypatch):
    base=trial.base
    # configure mutates only process-local module bindings.
    for key,value in vars(base).copy().items():monkeypatch.setattr(base,key,value)
    from experiments.pamf_dura_direct_train import evaluation as ev
    for key in ('score','load_model','configuration'):monkeypatch.setattr(ev,key,getattr(ev,key))
    trial.configure();config=base.configuration()
    assert (config['epochs'],config['updates'],config['L_exposures'],config['U_exposures'])==(30,7410,189000,1701000)
    assert config['training_F_access'] and config['training_TX_labels']==0 and not config['target_training']
    assert config['readouts']==['physical','scratch_joint'] and config['training_stages']==1
    assert 266951+sum(p.numel() for p in trial.ShapeEnhancer().parameters())==config['parameters']
    args=SimpleNamespace(root='root',commit='a'*40)
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))
    assert trial.learning_rate(0)==pytest.approx(.001/247)
    assert trial.learning_rate(7409)==pytest.approx(.00002)
    with pytest.raises(ValueError):trial.learning_rate(7410)


def test_E20_metadata_cannot_load_as_E30(monkeypatch,tmp_path):
    payload=dict(schema='dura_source_F_guided_v1',epoch=20,updates=4940)
    path=tmp_path/'e20.pt';torch.save(payload,path)
    with pytest.raises(ValueError,match='checkpoint metadata'):trial.load_model(path,'cpu',arm='scratch_joint',commit='a'*40)


def test_truth_stays_closed_if_either_readout_is_incomplete(monkeypatch):
    from experiments.pamf_dura_direct_train import evaluation as ev
    calls=[]
    monkeypatch.setattr(ev,'reference',lambda *a:(None,[1,2],None,None))
    def validate(*a,arm,**kw):
        calls.append(arm)
        if arm=='physical':raise ValueError('incomplete physical')
    monkeypatch.setattr(ev,'complete_arrays',validate)
    monkeypatch.setattr(ev,'score',lambda *a:calls.append('TRUTH_OPENED'))
    with pytest.raises(ValueError,match='incomplete'):
        trial.score(SimpleNamespace(commit='a'*40),{'output':__import__('pathlib').Path('none')},'target')
    assert calls==['scratch_joint','physical']


def test_real_module_entry_reaches_argument_parser():
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    native=root/'.codex_tmp/dura_rf_native47/experiments/adv3b02_xuc/code'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(native),str(root))))
    result=subprocess.run([sys.executable,'-m',trial.MODULE,'--help'],cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert '--commit' in result.stdout and '--root' in result.stdout
