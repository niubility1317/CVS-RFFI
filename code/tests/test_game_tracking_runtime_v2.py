import json
import inspect
import pytest
import torch

from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.step_context import prepare_context
from test_game_tracking_integration import setup
from test_game_tracking_resume import _training_states_equal


def test_v2_replay_records_requested_but_rejects_untrusted_state():
    from cvsrffi.game_tracking.runtime import replay_decision_v2
    row=dict(action='CORRECT',requested_head_steps=0)
    result=replay_decision_v2(row,None,{},step=0,version=0,budget=None)
    assert result['action']=='NORMAL' and result['requested_action']=='CORRECT'
    assert result['reason']=='replay_source_controller_uncalibrated'
    class Controller:
        def decide(self,*args,**kwargs): return dict(action='CATCHUP',catchup_steps=2,observation_id='obs',reason='trusted')
        def record_outcome(self,obs,**kwargs): self.outcome=(obs,kwargs)
    controller=Controller()
    result=replay_decision_v2(row,controller,{},step=0,version=0,budget=None)
    assert result['action']=='NORMAL' and result['reason']=='replay_action_not_supported_by_current_evidence'
    assert controller.outcome[1]['accepted'] is False


def test_v2_rejects_legacy_online_random_schedule():
    with pytest.raises(ValueError,match='frozen replay'):
        parse_args(['--output_dir','unused','--game_evidence_version','2','--game_control','random'])
    args=parse_args(['--output_dir','unused','--game_evidence_version','1','--game_control','random'])
    assert args.game_control=='random'


def test_v2_context_records_exact_mask_seed_and_replays_received_view():
    args, source, model, proto, device = setup()
    args.game_evidence_version = 2
    batch = next(iter(source.loader('train', 18, seed=42, shuffle=True)))
    generator = torch.Generator(device=device).manual_seed(7)
    ctx = prepare_context(batch, None, model, None, args, 1, 1, {'adv': .35}, generator, capability_level=.1)
    assert ctx.satellite_probability == .35
    assert ctx.satellite_selected_mask.dtype == torch.bool
    assert int(ctx.satellite_selected_mask.sum()) == ctx.satellite_mask_count
    record = dict(step=0, epoch=1, sample_ids=ctx.sample_ids, scenario=ctx.satellite_scenario,
                  selected_mask=ctx.satellite_selected_mask.cpu().tolist(), channel_seed=ctx.satellite_channel_seed)
    other = prepare_context(batch, None, model, None, args, 1, 1, {'adv': .35},
                            torch.Generator(device=device).manual_seed(777), exposure_record=record)
    assert torch.equal(ctx.satellite, other.satellite)
    assert torch.equal(ctx.satellite_selected_mask, other.satellite_selected_mask)
    record['sample_ids'] = list(reversed(record['sample_ids']))
    with pytest.raises(ValueError, match='sample'):
        prepare_context(batch, None, model, None, args, 1, 1, {'adv': .35}, generator, exposure_record=record)


def test_v2_no_audit_runtime_persists_new_implementation_and_exposure(tmp_path):
    torch.set_num_threads(2)
    args = parse_args(['--output_dir',str(tmp_path/'run'),'--game_synthetic','--device','cpu',
        '--epochs','1','--batch_size','18','--num_workers','0','--game_max_steps_per_epoch','1',
        '--game_skip_final_eval','--game_no_audit','--game_evidence_version','2',
        '--game_solver','head_lookahead','--game_b8_impl','head_grad_only'])
    assert runtime.train(args) == 0
    saved = torch.load(tmp_path/'run/final_ssdg.pth',map_location='cpu',weights_only=False)
    assert saved['schema'] == 'core90_game_epoch_boundary_v2'
    assert saved['solver']['b8_impl'] == 'head_grad_only'
    row = json.loads((tmp_path/'run/game_actions.jsonl').read_text())
    assert row['satellite_count'] == sum(row['selected_mask'])
    assert sum(row['actual_scenario_counts'].values()) == row['satellite_count']
    assert len(row['sample_ids']) == 18
    assert row['requested_action'] == row['action'] == 'NORMAL'


def _v2_args(output, *, no_audit=False, resume=None):
    values=['--output_dir',str(output),'--game_synthetic','--device','cpu',
        '--epochs','2','--batch_size','18','--num_workers','0','--game_max_steps_per_epoch','1',
        '--game_skip_final_eval','--game_evidence_version','2','--game_audit_interval','1',
        '--game_capability_interval','1','--game_probe_steps','2','--game_audit_samples_per_capture','2']
    if no_audit: values.append('--game_no_audit')
    if resume: values.extend(['--game_resume',str(resume)])
    return parse_args(values)


def test_v2_audit_only_and_epoch_resume_preserve_actual_training(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    continuous=tmp_path/'continuous'
    runtime.train(_v2_args(continuous))
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    a=load(continuous/'final_ssdg.pth')
    assert a['v2_coordinator']['next_game_audit_step'] == 2
    assert a['v2_coordinator']['next_capability_step'] == 2
    plain=tmp_path/'plain'
    runtime.train(_v2_args(plain,no_audit=True))
    _training_states_equal(a,load(plain/'final_ssdg.pth'))
    interrupted=tmp_path/'interrupted'
    original=runtime.checkpoint
    signature=inspect.signature(original)
    def stop(*args, **kwargs):
        original(*args, **kwargs)
        call=signature.bind(*args,**kwargs).arguments
        if call['epoch']==1: raise RuntimeError('intentional epoch boundary interruption')
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'checkpoint',stop)
        with pytest.raises(RuntimeError,match='intentional'): runtime.train(_v2_args(interrupted))
    runtime.train(_v2_args(interrupted,resume=interrupted/'latest_ssdg.pth'))
    b=load(interrupted/'final_ssdg.pth')
    _training_states_equal(a,b)
    assert a['v2_coordinator']['next_game_audit_step']==b['v2_coordinator']['next_game_audit_step']
    assert a['v2_coordinator']['next_capability_step']==b['v2_coordinator']['next_capability_step']


def test_runtime_consumes_frozen_actual_exposure_schedule(tmp_path, monkeypatch):
    from cvsrffi.game_tracking.data import build_source
    from scripts.build_core90_game_exposure_replay import channel_config_signature
    torch.set_num_threads(2)
    args=parse_args(['--output_dir',str(tmp_path/'run'),'--game_synthetic','--device','cpu',
        '--epochs','1','--batch_size','18','--num_workers','0','--game_skip_final_eval','--game_no_audit',
        '--game_evidence_version','2','--game_data_order_seed','777','--seed','2'])
    source=build_source(args)
    source.train=torch.utils.data.Subset(source.train,list(range(18)))
    batch=next(iter(source.loader('train',18,seed=778,shuffle=True,drop_last=True)))
    row=dict(step=0,epoch=1,sample_ids=list(batch[3]['sample_id']),scenario='leo_rain_weak',
             selected_mask=[i%2==0 for i in range(18)],channel_seed=917)
    schedule=dict(schema='core90_exposure_schedule_v2',source_only=True,target_evaluated=False,
        donor_seed=1,recipient_seed=2,recipient_data_order_seed=777,horizon=1,records=[row],
        data_contract=runtime.plain(source.info),channel_config=channel_config_signature(vars(args)))
    path=tmp_path/'exposure.json';path.write_text(json.dumps(schedule),encoding='utf-8')
    args.game_exposure_replay=str(path)
    monkeypatch.setattr(runtime,'build_source',lambda _:source)
    runtime.train(args)
    actual=json.loads((tmp_path/'run/game_actions.jsonl').read_text())
    assert all(actual[key]==row[key] for key in ('sample_ids','scenario','selected_mask','channel_seed'))
    assert actual['actual_scenario_counts']['leo_rain_weak']==9


def test_objective_stage_boundary_expires_prior_game_evidence(tmp_path):
    torch.set_num_threads(2)
    args=_v2_args(tmp_path/'boundary')
    args.game_audit_interval=args.game_capability_interval=250
    args.sat_cons_start_epoch=2  # synthetic boundary check, not an E80 training claim
    runtime.train(args)
    saved=torch.load(tmp_path/'boundary/final_ssdg.pth',map_location='cpu',weights_only=False)
    state=saved['v2_coordinator']
    assert state['last_metrics']['lag']['status']=='UNAVAILABLE'
    assert 'objective_or_stage_changed' in state['last_metrics']['lag']['reason_codes']
    assert state['next_game_audit_step']==250
    assert state['capability_metrics']['step']==0
