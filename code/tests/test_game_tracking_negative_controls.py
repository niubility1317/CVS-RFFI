"""Actual CORE90 CPU boundary replays; these are not E200 experiment runs."""
import torch
import pytest
from pathlib import Path

from scripts.accept_core90_v2_source import replay, first_difference, acceptance_args
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking import runtime


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('epochs,masks', [([1],None),([40,41],None),([79,80],None),
                                        ([130,131],None),([131,131],[False,True])])
def test_v2_audit_no_action_boundary_trajectory(epochs,masks):
    args = acceptance_args(synthetic=True)
    source = build_source(args)
    ordinary = replay(args,source,epochs,audited=False,mask_schedule=masks)
    observed = replay(args,source,epochs,audited=True,mask_schedule=masks)
    controlled = replay(args,source,epochs,audited=True,controlled=True,mask_schedule=masks)
    for name,branch in [('audit_only',observed),('no_action_controller',controlled)]:
        for i,(left,right) in enumerate(zip(ordinary,branch)):
            difference = first_difference(left['training'],right['training'])
            assert difference is None, f'{name}: first divergence at step={i}, E{epochs[i]}: {difference}'
            assert right['audit_schema'] == 'game_audit_v2'
            assert right['decision']['action'] == 'NORMAL'
    if masks:
        assert not ordinary[0]['training']['context']['base_mask'].any()
        assert ordinary[1]['training']['context']['base_mask'].all()
        assert not ordinary[0]['training']['context']['strong_mask'].any()
        assert ordinary[1]['training']['context']['strong_mask'].all()


def test_first_difference_reports_exact_state_path():
    left = {'optimizer':{'state':{0:{'step':torch.tensor(1.)}}}}
    right = {'optimizer':{'state':{0:{'step':torch.tensor(2.)}}}}
    out = first_difference(left,right)
    assert out['path'] == 'state.optimizer.state.0.step'
    assert out['max_abs_difference'] == 1.


def test_v2_runtime_checkpoint_trajectory_audit_and_disabled_actions(tmp_path,monkeypatch):
    """Exercise production dispatch, not only the fixed-context replay helper."""
    checkpoints={}
    original=runtime.checkpoint
    def capture(path,*args,**kwargs):
        original(path,*args,**kwargs)
        if Path(path).name=='latest_ssdg.pth':
            payload=torch.load(path,map_location='cpu',weights_only=False)
            checkpoints.setdefault(Path(path).parent.name,[]).append(payload)
    monkeypatch.setattr(runtime,'checkpoint',capture)
    for branch in ('ordinary','audit_only','no_action_controller'):
        args=acceptance_args(synthetic=True)
        args.output_dir=str(tmp_path/branch)
        args.epochs=2;args.game_max_steps_per_epoch=1
        args.game_no_audit=branch=='ordinary'
        args.game_control='both' if branch=='no_action_controller' else 'off'
        args.game_max_extra_head=0;args.game_correction_fraction=0.
        assert runtime.train(args)==0
    baseline=checkpoints['ordinary']
    for branch in ('audit_only','no_action_controller'):
        assert len(baseline)==len(checkpoints[branch])==2
        for left,right in zip(baseline,checkpoints[branch]):
            for key in ('model','ema','optimizer','scaler','prototype','solver','satellite_generator',
                        'epoch','step','encoder_version','source_info'):
                difference=first_difference(left[key],right[key],path=key)
                assert difference is None, f'{branch}, E{left["epoch"]}: {difference}'
            difference=first_difference(vars(left['rng']),vars(right['rng']),path='rng')
            assert difference is None,f'{branch}, E{left["epoch"]}: {difference}'
