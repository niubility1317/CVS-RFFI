import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.pamf_phase3_restoration import coupled_resume_support as support
from experiments.pamf_phase3_restoration import coupled_gradient_resume_trial as gradient
from experiments.pamf_phase3_restoration import coupled_feature_response_resume_trial as response


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_parent_requires_stopped_source_only_epoch_boundary(tmp_path):
    parent=tmp_path/'runs'/'parent';arm='arm';pid=999999999
    write(parent/'resource_stop.json',{'stage':'train','reason':'TOTAL_COMPUTE_PID_CAP_EXCEEDED','cap':3,
        'only_own_child_signaled':True,'other_owner_processes_signaled':False,'pid':pid})
    write(parent/('terminal-'+arm+'-train.json'),{'pid':pid,'exit_code':-15})
    write(parent/'launch.json',{'pid':pid})
    train=parent/'rows'/arm/'training'
    write(train/'smoke.json',{'status':'PASS','TX_labels_read':0,'V_reads':0,'OE_reads':0,'target_inputs':0})
    ckpt=train/'recovery'/'E15.pt';ckpt.parent.mkdir(parents=True);ckpt.write_bytes(b'test')
    old=SimpleNamespace(RUN='parent',MODULE='nonexistent_original_module')
    assert support.parent_training(tmp_path,old,arm,15)==train
    write(parent/('submitted-'+arm+'-cross_rx.json'),{'pid':pid})
    with pytest.raises(ValueError,match='prediction'):support.parent_training(tmp_path,old,arm,15)


@pytest.mark.parametrize('module,epoch,cap,gpu',[(gradient,15,3,5),(response,2,5,6)])
def test_resume_preserves_original_recipe_and_budget(module,epoch,cap,gpu,monkeypatch):
    old=module.PARENT_CONFIG
    cfg=module.configuration(module.ARM)
    assert cfg['epochs']==old['epochs']==30 and cfg['updates']==old['updates']==7410
    assert cfg['source_draws']==old['source_draws'] and cfg['objective']==old['objective']
    assert cfg['technical_recovery_epoch']==epoch and cfg['physical_gradient_coupling']==.05
    got={}
    def gpu_choice(*,allowed_gpus,maximum_processes):
        got.update(gpu=allowed_gpus,maximum=maximum_processes)
        return [{'GPU':gpu,'free_memory_MB':20000},{'GPU':gpu,'free_memory_MB':10000}]
    monkeypatch.setattr(module,'recovery_gpu_choice',gpu_choice)
    assert len(module.admitted_gpus())==1 and got['gpu']==(gpu,)
    assert got['maximum']==(3 if cap==3 else 5)


def test_technical_resume_passes_response_trace_and_coupling(monkeypatch,tmp_path):
    capture={}
    monkeypatch.setattr(response.resume,'parent_training',lambda *args:tmp_path)
    monkeypatch.setattr(response.training,'train',lambda *args,**kwargs:capture.update(kwargs))
    response.train(SimpleNamespace(root=tmp_path),tmp_path/'new')
    assert capture['training_trace'] and capture['physical_detail_coupled']
    assert capture['training_posterior_multiplier']==2 and capture['training_F0_multiplier']==3
    assert capture['technical_resume']['epoch']==2 and capture['technical_resume']['config']==response.PARENT_CONFIG
