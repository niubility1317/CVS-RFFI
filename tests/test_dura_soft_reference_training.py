import copy
import pytest
from experiments.pamf_dura_direct_train import soft_reference as method


def qualified_sources():
    teacher=method.f_guided.teacher_reference()
    reference=dict(source_reference_geometry_eligible=True,beta=.25,F_teacher=teacher,source_records=63000,probe_records=512,F_unchanged=True,
        G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    calibration=dict(decision={'eligible':True},alpha=method.ALPHA,reference_beta=.25,reference_run=method.REFERENCE_RUN,
        raw_J_cache_run=method.parent.CACHE_RUN,F_teacher=teacher,F_unchanged=True,
        source_checkpoint='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01',source_commit='8cf8e22c9ffb54d271b22e621401de8e34c69068',
        new_G_inputs=0,new_F_clean_J_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    return reference,calibration


def test_unqualified_or_target_consuming_reference_cannot_enter_training():
    reference,calibration=qualified_sources();method.validate_reference_sources(reference,calibration)
    bad=copy.deepcopy(reference);bad['target_reads']=1
    with pytest.raises(ValueError):method.validate_reference_sources(bad,calibration)
    bad=copy.deepcopy(calibration);bad['decision']['eligible']=False
    with pytest.raises(ValueError):method.validate_reference_sources(reference,bad)


def test_different_f_or_reference_beta_is_not_a_compatible_source():
    reference,calibration=qualified_sources();bad=copy.deepcopy(calibration);bad['F_teacher']['seed']=2026092702
    with pytest.raises(ValueError):method.validate_reference_sources(reference,bad)
    bad=copy.deepcopy(reference);bad['beta']=.5
    with pytest.raises(ValueError):method.validate_reference_sources(bad,calibration)


def test_worker_config_retains_budget_and_dispatches_the_new_module():
    from types import SimpleNamespace
    from pathlib import Path
    method.configure();c=method.base.configuration()
    assert (c['epochs'],c['updates'],c['L_exposures'],c['U_exposures'])==(30,7410,189000,1701000)
    assert c['reference_beta']==.25 and c['fixed_source_weight']==method.ALPHA and c['source_reference_offset_numeric_bytes']==1260000
    assert c['matched_control']==method.parent.RUN and c['source_reference_cache_new_F_inputs']==0
    command=method.base.command(SimpleNamespace(root=Path('/source'),commit='code'),'worker','scratch_joint')
    assert command[3]==method.MODULE and command[-2:]==['--arm','scratch_joint']
