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
                     '--game_skip_final_eval'])
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
