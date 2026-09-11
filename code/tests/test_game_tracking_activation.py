"""Controlled source signals exercise actual runtime actions, not performance."""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.controller import GameController,ControllerConfig
from cvsrffi.game_tracking.curriculum import CapabilityCurriculum,CurriculumConfig


def test_actual_runtime_catchup_correction_and_curriculum(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    args=parse_args(['--output_dir',str(tmp_path/'run'),'--game_synthetic','--epochs','1',
                     '--game_max_steps_per_epoch','3','--batch_size','12','--num_workers','0',
                     '--device','cpu','--game_audit_interval','1','--game_source_calibration_steps','0',
                     '--game_control','both','--game_curriculum','capability','--game_correction_fraction','1',
                     '--game_skip_final_eval','--game_evidence_version','1'])
    def controlled_audit(model,source,indexes,auditor,args,step,version,*extra):
        return dict(valid=True,step=step,encoder_version=version,G_lag=.4 if step==0 else .01,
                    S_rx=.5,identity=.9 if step<2 else .1,identity_valid=True,margin=.5,consistency=.95,
                    direction_imbalance=.9,gradient_valid=True,elapsed_seconds=.001),{}
    def calibrated(observations,args):
        controller=GameController(ControllerConfig(confirmation_windows=1,cooldown_steps=0,catchup_steps=1,
                         audit_interval=1,sparse_audit_interval=2,max_version_lag=1))
        curriculum=CapabilityCurriculum(CurriculumConfig(confirmation_windows=1,cooldown_steps=0,max_version_lag=1))
        return controller,curriculum
    monkeypatch.setattr(runtime,'audit',controlled_audit)
    monkeypatch.setattr(runtime,'calibrate',calibrated)
    runtime.train(args)
    rows=[json.loads(line) for line in (tmp_path/'run/game_actions.jsonl').read_text().splitlines()]
    assert rows[0]['action']=='CATCHUP' and rows[0]['committed_head_steps']==1
    assert rows[1]['action']=='CORRECT' and rows[1]['field_evaluations']==2 and rows[1]['accepted']
    assert rows[2]['action']=='HOLD_CURRICULUM' and rows[2]['field_evaluations']==1
    events=[json.loads(line) for line in (tmp_path/'run/curriculum_events.jsonl').read_text().splitlines()]
    assert events[1]['changed'] and not events[0]['changed'] and not events[2]['changed']
    checkpoint=torch.load(tmp_path/'run/final_ssdg.pth',weights_only=False)
    deployment=torch.load(tmp_path/'run/deployment.pth',weights_only=False)
    assert all(torch.equal(value,deployment['model'][key]) for key,value in checkpoint['model'].items())
    source=runtime.build_source(args)
    model=runtime.build_model(args,len(source.domains),torch.device('cpu')).eval()
    x=source.train[0][0][None]
    with torch.no_grad():
        model.load_state_dict(checkpoint['model']); original=model(x,return_aux=True)['tx_logits']
        model.load_state_dict(deployment['model']); exported=model(x,return_aux=True)['tx_logits']
    assert torch.equal(original,exported)


def test_v2_typed_evidence_commits_real_actions_once(tmp_path,monkeypatch):
    from cvsrffi.game_tracking import runtime_control as rc
    from cvsrffi.game_tracking.controller import GameControllerV2
    from cvsrffi.game_tracking.curriculum import CapabilityCurriculumV2,CurriculumConfigV2
    torch.set_num_threads(2)
    args=parse_args(['--output_dir',str(tmp_path/'v2'),'--game_synthetic','--epochs','1',
        '--game_max_steps_per_epoch','3','--batch_size','12','--num_workers','0','--device','cpu',
        '--game_audit_interval','1','--game_capability_interval','1','--game_evidence_version','2',
        '--game_control','both','--game_curriculum','capability','--game_correction_fraction','1',
        '--game_skip_final_eval','--game_telemetry_interval','0'])
    original=rc.V2Coordinator
    class Controlled(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.controller=GameControllerV2(ControllerConfig(confirmation_windows=1,cooldown_steps=0,
                catchup_steps=1,audit_interval=1,sparse_audit_interval=1,max_version_lag=1))
            self.curriculum=CapabilityCurriculumV2(CurriculumConfigV2(confirmation_windows=1,cooldown_steps=0))
    def cap(*a,step,version,policy_level,**kw):
        return dict(schema='game_capability_v2',step=step,encoder_version=version,
            valid=True,identity_valid=True,collapsed=False,identity=.95 if step<2 else .1,
            margin=.5,next_identity=.95,next_margin=.5,next_worst_tx=.9,
            policy_level=policy_level,elapsed_seconds=.001)
    def game(*a,step,version,**kw):
        return dict(schema='game_audit_v2',observation_id=f'fixture:{step}',step=step,
            encoder_version=version,data_valid=True,coverage_valid=True,elapsed_seconds=.001,
            lag=dict(status='RELIABLE_HIGH_GAP' if step==0 else 'RELIABLE_LOW_GAP',
                     quality_pass=True,control_ready=True,gap_normalized=.4 if step==0 else .01,readability=.5),
            gradient=dict(valid=True,representative=True,scope='full_current_training_objective',direction_imbalance=.9)),{}
    monkeypatch.setattr(rc,'V2Coordinator',Controlled)
    monkeypatch.setattr(rc,'game_audit_v2',game);monkeypatch.setattr(rc,'capability_audit_v2',cap)
    runtime.train(args)
    rows=[json.loads(line) for line in (tmp_path/'v2/game_actions.jsonl').read_text().splitlines()]
    assert [r['action'] for r in rows]==['CATCHUP','CORRECT','NORMAL']
    assert [r['committed_head_steps'] for r in rows]==[1,0,0]
    assert [r['field_evaluations'] for r in rows]==[1,2,1]
    saved=torch.load(tmp_path/'v2/final_ssdg.pth',weights_only=False)
    control=saved['v2_coordinator']['controller']
    assert control['accepted_events']==2 and len(control['consumed_observations'])==2
    assert not control['requested_observations']
