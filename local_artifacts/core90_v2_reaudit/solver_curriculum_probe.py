from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/tests'))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.curriculum import CapabilityCurriculumV2
from cvsrffi.game_tracking.runtime_control import V2Coordinator
from cvsrffi.game_tracking.step_context import satellite_policy,satellite_stage,Core90Objective
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.schedule import build_stage_state,build_aug_base_cfg,make_augmentor,configure_augmentor_for_epoch,configure_mixstyle_for_epoch
from test_game_tracking_b8_fastpath import core_fixture

torch.set_num_threads(2)
a=parse_args(['--output_dir','unused','--game_synthetic','--device','cpu','--game_curriculum','capability'])
report={}
c=CapabilityCurriculumV2();events=[]
for step in (0,250,500):
    good=step==0
    m=dict(schema='game_capability_v2',valid=True,identity_valid=True,collapsed=False,
           step=step,encoder_version=step,policy_level=0.,identity=.95 if good else .6,
           margin=.3 if good else -.02,next_identity=.95 if good else .55,
           next_margin=.1,next_worst_tx=.5)
    events.append(dict(input=m,event=c.update(m,step=step,encoder_version=step),streak=c.streak))
report['one_enter_two_middle_band']=events

model,args,proto,ctx,_=core_fixture(131)
base=build_aug_base_cfg(a);aug=make_augmentor(base)
stages=[]
for left,right in ((40,41),(90,91)):
    w1=_loss_weights(a,build_stage_state(left,a));w2=_loss_weights(a,build_stage_state(right,a))
    aug1=configure_augmentor_for_epoch(aug,base,left,a)
    aug2=configure_augmentor_for_epoch(aug,base,right,a)
    mix1=configure_mixstyle_for_epoch(model,a,left);mix2=configure_mixstyle_for_epoch(model,a,right)
    stages.append(dict(epochs=[left,right],weights_equal=w1==w2,augmentation_equal=aug1==aug2,
        augmentation=[aug1,aug2],mixstyle_equal=mix1==mix2,mixstyle=[mix1,mix2],
        actual_capability_policy=satellite_policy(0.),runtime_stage_changed=right in (41,91,a.label_epochs+1,a.sat_cons_start_epoch)))
report['unchanged_stage_resets']=stages

class Budget:
    def can_afford(self,*a,**k):return True,[]
    def commit(self,*a,**k):pass
a.game_curriculum='fixed'
coord=V2Coordinator(None,None,a,None,None,Budget(),{})
coord.next_game_audit_step=99999
with patch('cvsrffi.game_tracking.runtime_control.capability_audit_v2',return_value=dict(valid=False,elapsed_seconds=0.)) as mock:
    coord.observe(None,step=0,version=0,epoch=131)
    report['fixed_current_view_mismatch']=dict(training_fixed_policy=satellite_stage(131),
        observed_policy_level=mock.call_args.kwargs['policy_level'],
        observed_policy=satellite_policy(mock.call_args.kwargs['policy_level']))

torch.manual_seed(99)
model,args,proto,ctx,_=core_fixture(131)
start=RNGState.capture();results=[];states=[]
for impl in ('reference','head_grad_only','graph_reuse'):
    m=deepcopy(model);p=deepcopy(proto);context=deepcopy(ctx)
    opt=torch.optim.AdamW(m.parameters(),lr=.0002)
    solver=GameSolver(m,opt,'head_lookahead',b8_impl=impl,max_grad_norm=5.,telemetry_interval=250)
    obj=Core90Objective(m,args,p);start.restore()
    # Same closure selection as runtime.py:548-550 on a telemetry step.
    result=solver.step(obj.prepare_reusable_graph(context))
    results.append(dict(impl=impl,accepted=result.accepted,algorithm=result.algorithm,
        forward_calls=context.forward_calls,telemetry=result.telemetry))
    states.append({n:v.detach().clone() for n,v in m.state_dict().items()})
report['b8_actual_core90_E131_one_step']=results
report['b8_parameter_differences']={impl:max(float((v-states[i][n]).abs().max()) for n,v in states[0].items())
    for i,impl in enumerate(('reference','head_grad_only','graph_reuse'))}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='b8_actual_core90_E131_one_step'},ensure_ascii=False,indent=2))
print([(r['impl'],r['accepted'],r['forward_calls'],r['telemetry']['origin_scope'],r['telemetry']['gradient_component_diagnostics']) for r in results])
