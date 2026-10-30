from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_efficiency import context_oracle as trial


def test_rank_projection_retains_constant_and_full_rank_complex_trace():
    constant=torch.full((2,256),2+3j,dtype=torch.complex64)
    torch.testing.assert_close(trial.low_rank_trace(constant),constant,atol=2e-5,rtol=2e-5)
    generator=torch.Generator().manual_seed(8)
    value=torch.complex(torch.randn(2,256,generator=generator,dtype=torch.float64),torch.randn(2,256,generator=generator,dtype=torch.float64))
    torch.testing.assert_close(trial.low_rank_trace(value,256),value,atol=1e-11,rtol=1e-11)
    with pytest.raises(ValueError):trial.low_rank_trace(value,0)


def test_interventions_replace_only_named_context_and_do_not_mutate_source_action():
    clean=torch.ones(2,2,256);coarse=clean*2;variance=torch.ones(2)*.1
    kernel=torch.zeros(2,64,dtype=torch.complex64);kernel[:,0]=1
    main=torch.ones(2,256,dtype=torch.complex64);image=torch.zeros_like(main)
    estimated=Action(kernel,main,image);actual=Action(kernel,main*1.1,image)
    values=trial.contexts(clean,(coarse,estimated,variance),actual)
    assert tuple(values)==trial.CASES
    assert values['known_action'][0] is coarse and values['known_action'][1] is actual
    assert values['known_coarse'][0] is clean and values['known_coarse'][1] is estimated
    assert all(v[2] is variance for v in values.values())
    torch.testing.assert_close(estimated.main,main,rtol=0,atol=0)
    assert values['rank32_action'][1].kernel is kernel


def source_summary():
    cases={case:dict(iq_nmse=.016 if case!='native' else .02,teacher_correct=504 if case!='native' else 498) for case in trial.CASES}
    groups={'all':dict(count=512,operator_rank32_nmse=.001,cases=cases)}
    for i,view in enumerate(trial.base.VIEWS):
        count=74 if i==0 else 73;scene_cases=copy.deepcopy(cases)
        for case in trial.CASES:scene_cases[case]['teacher_correct']=count-2+(1 if case!='native' and i<6 else 0)
        groups['scene:'+view]=dict(count=count,operator_rank32_nmse=.001,cases=scene_cases)
    clean=dict(all=dict(count=512,cases={case:dict(teacher_correct=512) for case in trial.CASES}))
    return {'clean':clean,'1':copy.deepcopy(groups),'20':copy.deepcopy(groups)}


def test_fixed_source_rules_require_both_views_and_physical_fit():
    summary=source_summary();decision=trial.decide(summary)
    assert decision['preferred_independent_study']=='rank32_action' and not decision['automatic_training']
    summary['20']['scene:practical_low_urban']['operator_rank32_nmse']=.02
    assert trial.decide(summary)['preferred_independent_study']=='source_action_curriculum'
    summary['1']['all']['cases']['known_action']['teacher_correct']=503
    assert trial.decide(summary)['preferred_independent_study'] is None
    summary=source_summary();summary['20']['scene:clean']['cases']['rank32_action']['teacher_correct']=71
    assert not trial.decide(summary)['rank32_action']['eligible']


def test_source_checkpoint_and_CPU_boundary_reject_wrong_contract(monkeypatch,tmp_path):
    done=dict(status='SOURCE_TRAINED',epoch=30,updates=7410,L_exposures=189000,U_exposures=1701000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,teacher_updates=0,
        learned_teacher_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,head_fit_updates=0,parameters=266951)
    trial.validate_source(done)
    for key,value in [('epoch',20),('training_F_access',True),('target_reads',1)]:
        with pytest.raises(ValueError):trial.validate_source(dict(done,**{key:value}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError):trial.run(SimpleNamespace(root=tmp_path,output=tmp_path/'runs'/trial.RUN))
