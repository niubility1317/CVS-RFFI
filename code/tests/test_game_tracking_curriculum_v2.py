from copy import deepcopy
import pytest
from cvsrffi.game_tracking import curriculum
from cvsrffi.game_tracking import step_context


def capability(step,identity=.95,margin=.3):
    return dict(schema='game_capability_v2',valid=True,step=step,encoder_version=step,
                identity_valid=True,identity=identity,margin=margin,collapsed=False,
                next_identity=identity,next_margin=margin,next_worst_tx=identity,
                current_identity=identity,consistency=.01,policy_level=0.)


def test_effective_continuous_policy_changes_at_each_increment():
    assert hasattr(step_context,'satellite_policy'), 'continuous policy is missing'
    previous=None
    for i in range(11):
        policy=step_context.satellite_policy(i/10)
        assert policy['probability']==pytest.approx(.3+.5*i/10)
        assert sum(policy['weights'])==pytest.approx(1.)
        assert policy['weights'][1]==pytest.approx(i/30)
        assert policy!=previous
        previous=policy
    assert step_context.satellite_policy(1.)['weights']==pytest.approx([1/3]*3)
    with pytest.raises(ValueError): step_context.satellite_policy(float('nan'))


def test_capability_clock_does_not_require_lag_or_cosine_and_resumes_exactly():
    assert hasattr(curriculum,'CapabilityCurriculumV2')
    c=curriculum.CapabilityCurriculumV2(curriculum.CurriculumConfigV2(cooldown_steps=250))
    for step in [0,250]:
        assert not c.update(capability(step),step=step,encoder_version=step)['changed']
    restored=curriculum.CapabilityCurriculumV2(c.config)
    restored.load_state_dict(deepcopy(c.state_dict()))
    event=c.update(capability(500),step=500,encoder_version=500)
    assert event['changed'] and event['level']==.1 and event['reset_optimistic']
    assert event==restored.update(capability(500),step=500,encoder_version=500)
    assert not c.update(capability(500),step=500,encoder_version=500)['changed']


def test_high_cosine_random_identity_or_invalid_next_view_cannot_upgrade():
    assert hasattr(curriculum,'CapabilityCurriculumV2')
    c=curriculum.CapabilityCurriculumV2()
    for step in [0,250,500,750]:
        m=capability(step,identity=1/6);m['consistency']=1.
        assert not c.update(m,step=step,encoder_version=step)['changed']
    m=capability(1000);m['next_identity']=None
    assert not c.update(m,step=1000,encoder_version=1000)['changed']


def test_old_schema_cannot_silently_drive_new_curriculum():
    assert hasattr(curriculum,'CapabilityCurriculumV2')
    c=curriculum.CapabilityCurriculumV2()
    m=capability(0);m.pop('schema')
    assert not c.update(m,step=0,encoder_version=0)['changed']
    with pytest.raises(ValueError): c.load_state_dict(dict(config={},level=.1))


def test_v2_hysteresis_band_retains_readiness_but_cannot_enter():
    c=curriculum.CapabilityCurriculumV2()
    def band(step):
        m=capability(step,identity=.6,margin=-.02)
        m.update(next_identity=.55,next_margin=.1,next_worst_tx=.5)
        return m
    assert not c.update(band(0),step=0,encoder_version=0)['ready']
    assert c.update(capability(250),step=250,encoder_version=250)['ready']
    middle=c.update(band(500),step=500,encoder_version=500)
    assert middle['ready'] and not middle['changed'] and c.streak==2
    assert not c.update(band(500),step=500,encoder_version=500)['changed']
    resumed=curriculum.CapabilityCurriculumV2(c.config);resumed.load_state_dict(deepcopy(c.state_dict()))
    event=c.update(band(750),step=750,encoder_version=750)
    assert event==resumed.update(band(750),step=750,encoder_version=750)
    assert event['changed'] and event['reset_optimistic'] and not event['ready']


@pytest.mark.parametrize('field,value',[('identity',.49),('margin',-.051),('next_identity',.44)])
def test_v2_hysteresis_exit_resets_all_confirmations(field,value):
    c=curriculum.CapabilityCurriculumV2()
    assert c.update(capability(0),step=0,encoder_version=0)['ready']
    m=capability(250);m[field]=value
    assert not c.update(m,step=250,encoder_version=250)['ready'] and c.streak==0
    m=capability(500,identity=.6);m['next_identity']=.55
    assert not c.update(m,step=500,encoder_version=500)['ready']


def test_v2_hysteresis_respects_cooldown_and_new_evidence():
    c=curriculum.CapabilityCurriculumV2(curriculum.CurriculumConfigV2(cooldown_steps=250,max_age_steps=500))
    for step in [0,1,2]: event=c.update(capability(step),step=step,encoder_version=step)
    assert event['changed']
    for step in [3,4,5]:
        m=capability(step);m['policy_level']=.1
        event=c.update(m,step=step,encoder_version=step)
    assert event['ready'] and not event['changed']
    # Fresh evidence after cooldown is required; repeated old observation cannot commit.
    assert not c.update(m,step=252,encoder_version=5)['changed']
    m=capability(253);m['policy_level']=.1
    assert c.update(m,step=253,encoder_version=253)['changed']


def test_same_identity_head_measures_current_and_next_view_without_fitting_monitor():
    import torch
    from cvsrffi.game_tracking import capability as module
    assert hasattr(module,'evaluate_capability_v2'), 'direct difficulty readout is missing'
    tx=torch.tensor([0,0,1,1]*4);rx=torch.tensor([0,1,0,1]*4)
    z=torch.eye(2)[tx]*3.
    result=module.evaluate_capability_v2(z,tx,rx,z,tx,rx,
          fit_groups=['f'+str(i) for i in range(16)],monitor_groups=['m'+str(i) for i in range(16)],
          monitor_views={'current':z,'next':z.flip(1)},policy_level=0.,
          config=module.CapabilityConfig(steps=25,lr=.15),independence_verified=True)
    assert result['schema']=='game_capability_v2' and result['identity']==1.
    assert result['current_identity']==1. and result['next_identity']==0.
    assert result['next_worst_tx']==0. and result['views']['next']['per_rx_tx']
    assert result['views']['clean']['per_rx_tx']
    assert result['views']['clean']['worst_tx'] == result['clean_worst_tx']


def test_invalid_current_view_returns_unavailable_consistency():
    import torch
    from cvsrffi.game_tracking.capability import evaluate_capability_v2,CapabilityConfig
    tx=torch.tensor([0,0,1,1]*4);rx=torch.tensor([0,1,0,1]*4)
    z=torch.eye(2)[tx]*3.
    result=evaluate_capability_v2(z,tx,rx,z,tx,rx,
        fit_groups=['f'+str(i) for i in range(16)],monitor_groups=['m'+str(i) for i in range(16)],
        monitor_views={'current':z[:2],'next':z},policy_level=0.,
        config=CapabilityConfig(steps=1),independence_verified=True)
    assert not result['valid'] and result['consistency'] is None
