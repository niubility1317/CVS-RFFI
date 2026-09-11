"""Read-only CPU reproducer: HIGH_GAP without convergence becomes CORRECT."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.game_tracking.runtime_control import calibrate_game_v2
from cvsrffi.game_tracking.audit_evidence import make_evidence_v2
from cvsrffi.game_tracking.source_audit import fit_empirical_lag,ProbeConfigV2


def run():
    caps=[dict(schema='game_capability_v2',valid=True,identity_valid=True,collapsed=False,
               identity=.9,margin=.3,next_identity=.9,next_margin=.2,next_worst_tx=.8)
          for _ in range(3)]
    history=[make_evidence_v2(observation_id=str(i),step=i,encoder_version=i,
        data_valid=True,coverage_valid=True,lag=dict(status='RELIABLE_HIGH_GAP',
        quality_pass=True,control_ready=True,gap_normalized=1.)) for i in range(3)]
    args=SimpleNamespace(num_classes=6,game_max_extra_head=3,game_audit_interval=250)
    controller=calibrate_game_v2(history,caps,args)
    torch.set_num_threads(1)
    head=torch.nn.Linear(1,2).train()
    with torch.no_grad(): head.weight.zero_();head.bias.zero_()
    actual=fit_empirical_lag(head,torch.tensor([[-3.],[3.]]),torch.tensor([0,1]),sample_weights=None,
        objective_scope='current_training_head_objective',config=ProbeConfigV2(objective_scale=.35),
        fixed_head_state=dict(cpu_rng=torch.get_rng_state(),module_training={'':True}))
    evidence=make_evidence_v2(observation_id='new1',step=1000,encoder_version=1000,
        data_valid=True,coverage_valid=True,lag=actual.metrics,
        gradient=dict(valid=True,representative=True,scope='full_current_training_objective',
                      direction_imbalance=.8),capability=dict(caps[0],step=1000,encoder_version=1000))
    first=controller.decide(evidence,step=1000,encoder_version=1000)
    evidence.update(observation_id='new2',step=1250,encoder_version=1250)
    evidence['capability'].update(step=1250,encoder_version=1250)
    second=controller.decide(evidence,step=1250,encoder_version=1250)
    return dict(lag_enter=controller.config.lag_enter,lag_exit=controller.config.lag_exit,
                evidence_lag=evidence['lag'],first=first,second=second,
                defect_reproduced=second['action']=='CORRECT')


if __name__=='__main__':
    result=run()
    output=Path(__file__).with_name('repro_high_gap_correction_actual_probe.json')
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result))
