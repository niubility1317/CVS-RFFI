from copy import deepcopy
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from scripts.accept_core90_v2_source import acceptance_args, first_difference
from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.game_tracking.step_context import prepare_context, Core90Objective
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.schedule import build_stage_state

torch.set_num_threads(2)
a=acceptance_args(synthetic=True)
s=build_source(a);m=runtime.build_model(a,len(s.domains),torch.device('cpu')).train()
opt=torch.optim.AdamW(m.parameters(),lr=.0002)
b=next(iter(s.loader('train',18,seed=1,shuffle=True)))
c=prepare_context(b,None,m,None,a,131,1,_loss_weights(a,build_stage_state(131,a)),torch.Generator().manual_seed(3))
assert GameSolver(m,opt).step(lambda:Core90Objective(m,a,None)(c)).accepted
before_params={n:p.detach().clone() for n,p in m.named_parameters()}
before_buffers={n:b.detach().clone() for n,b in m.named_buffers()}
before_opt={n:deepcopy(opt.state.get(p,{})) for n,p in m.named_parameters()}
before_grad={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()}
before_rng=deepcopy(vars(RNGState.capture()))
before_ctx=deepcopy(vars(c));before_flags={n:v.training for n,v in m.named_modules()}
steps,forwards=runtime.head_catchup_v2(m,opt,c,2,a)
differences=[];cleared=[];head_changes=[]
for n,p in m.named_parameters():
    if n.startswith('adv_head.'):
        if not torch.equal(before_params[n],p):head_changes.append(n)
        continue
    for kind,before,after in [('parameter',before_params[n],p),('optimizer',before_opt[n],opt.state.get(p,{}))]:
        d=first_difference(before,after,f'{kind}.{n}')
        if d:differences.append(d)
    if before_grad[n] is not None and p.grad is None:cleared.append(n)
for n,b in m.named_buffers():
    if not n.startswith('adv_head.'):
        d=first_difference(before_buffers[n],b,f'buffer.{n}')
        if d:differences.append(d)
for name,before,after in [('rng',before_rng,vars(RNGState.capture())),('context',before_ctx,vars(c)),
                         ('flags',before_flags,{n:v.training for n,v in m.named_modules()})]:
    d=first_difference(before,after,name)
    if d:differences.append(d)
runtime.json_write(Path(__file__).with_suffix('.json'),dict(status='VERIFIED' if not differences else 'FAILED',
    actual_head_steps=steps,encoder_forwards=forwards,nonhead_parameter_buffer_optimizer_or_rng_differences=differences,
    changed_head_parameters=head_changes,nonhead_gradient_buffers_cleared=cleared,
    scope='read_only_review_of_actual_head_catchup_v2_on_synthetic_CPU_warm_optimizer'))
print('steps',steps,'differences',len(differences),'cleared_nonhead_grad_buffers',len(cleared))
