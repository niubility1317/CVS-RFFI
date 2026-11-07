from types import SimpleNamespace

import pytest
import torch

from experiments.pamf_phase3_restoration import bounded_projected_resource_trial as resource


def test_admission_leaves_headroom(monkeypatch):
    observed={}
    def choice(**kwargs):observed.update(kwargs);return [{'GPU':2,'existing_compute_PIDs':1}]
    monkeypatch.setattr(resource,'recovery_gpu_choice',choice)
    assert resource.admitted_gpus()==[{'GPU':2,'existing_compute_PIDs':1}]
    assert observed=={'allowed_gpus':(2,3,4),'maximum_processes':2}


def test_gpu_pid_count_deduplicates_and_rejects_wrong_device():
    def ok(*args,**kwargs):return SimpleNamespace(returncode=0,stdout='GPU-x, 3\nGPU-x, 3\nGPU-x, 4\n',stderr='')
    assert resource.compute_pids(2,'GPU-x',ok)=={3,4}
    def wrong(*args,**kwargs):return SimpleNamespace(returncode=0,stdout='GPU-y, 3\n',stderr='')
    with pytest.raises(ValueError):resource.compute_pids(2,'GPU-x',wrong)


def test_replacement_preserves_scientific_initialization():
    previous=resource.method.factory(resource.ARM);replacement=resource.factory(resource.ARM)
    assert previous.state_dict().keys()==replacement.state_dict().keys()
    assert all(torch.equal(v,replacement.state_dict()[k]) for k,v in previous.state_dict().items())
    first=resource.method.configuration(resource.ARM);second=resource.configuration(resource.ARM)
    for key in ('model_seed','epochs','updates','batch','source_draws','objective','max_relative_correction'):
        assert first[key]==second[key]
