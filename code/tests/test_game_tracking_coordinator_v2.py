from types import SimpleNamespace
import torch
from cvsrffi.game_tracking.budget import ComputeBudget, BudgetConfig


def test_independent_clocks_and_resume(monkeypatch):
    from cvsrffi.game_tracking import runtime_control as rc
    calls=[]
    def game(*a,**kw):
        calls.append(('game',kw['step']))
        return dict(schema='game_audit_v2',step=kw['step'],encoder_version=kw['version'],
                    lag={'status':'OPTIMIZATION_FAILURE'},elapsed_seconds=0.),{}
    def cap(*a,**kw):
        calls.append(('cap',kw['step']))
        return dict(schema='game_capability_v2',step=kw['step'],encoder_version=kw['version'],
                    valid=False,elapsed_seconds=0.)
    monkeypatch.setattr(rc,'game_audit_v2',game)
    monkeypatch.setattr(rc,'capability_audit_v2',cap)
    args=SimpleNamespace(game_audit_interval=4,game_capability_interval=2,
        game_source_calibration_steps=0,num_classes=6,game_max_extra_head=2)
    def make():
        return rc.V2Coordinator(torch.nn.Linear(1,1),None,args,None,None,
                 ComputeBudget(BudgetConfig(max_audit_seconds=100.)),{})
    coordinator=make()
    for step in range(5):coordinator.observe(None,step=step,version=step,epoch=1)
    assert calls==[('cap',0),('game',0),('cap',2),('cap',4),('game',4)]
    state=coordinator.state_dict()
    restored=make();restored.load_state_dict(state)
    assert restored.state_dict()==state
    assert restored.controller is None and restored.curriculum is None
    assert restored.next_game_audit_step==8 and restored.next_capability_step==6


def test_capability_calibration_does_not_need_reliable_lag():
    from cvsrffi.game_tracking.runtime_control import calibrate_capability_v2
    args=SimpleNamespace(num_classes=6,game_capability_interval=250)
    observations=[dict(valid=True,identity_valid=True,collapsed=False,identity=.8,
        margin=.2,next_identity=.7,next_margin=.1,next_worst_tx=.5,step=s)
        for s in (0,250,500)]
    curriculum=calibrate_capability_v2(observations,args)
    assert curriculum is not None
    assert curriculum.config.calibration_id.startswith('source_capability_')
