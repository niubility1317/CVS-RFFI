import argparse,json
from pathlib import Path
import torch
from . import design as d
from .method import alpha,target,ema_update,objective
def run():
 c=d.config(d.rows()[0]);d.validate(c);bad=dict(c,source_U=False)
 if any(d.config(r)['U_cache']!=d.U_CACHE.as_posix() for r in d.rows()):raise AssertionError('Existing truthblind U cache missing')
 try:d.validate(bad)
 except ValueError:pass
 else:raise AssertionError('Changed permission accepted')
 if alpha(1111)!=.8 or alpha(4440)!=0 or alpha(6660)!=0:raise AssertionError('Anchor handoff differs')
 s=torch.nn.Linear(2,6);e=torch.nn.Linear(2,6);e.load_state_dict(s.state_dict());e.requires_grad_(False);before=e.weight.clone();version=e.weight._version
 with torch.no_grad():s.weight.add_(1)
 ema_update(e,s,1)
 if e.weight._version<=version or not torch.allclose(e.weight,before+.5) or e.weight.grad is not None:raise AssertionError('EMA update/cache version differs')
 logits=torch.randn(4,6,requires_grad=True);anchor=torch.randn(4,6).softmax(1);q=target(logits,anchor,1111)
 if q.requires_grad or not torch.allclose(target(logits,anchor,4440),logits.softmax(1)) or not torch.allclose(q,.2*logits.softmax(1)+.8*anchor):raise AssertionError('Pseudo target detach/handoff differs')
 try:target(logits,anchor*2,1200)
 except ValueError:pass
 else:raise AssertionError('Malformed probability accepted')
 from experiments.cvs_stage2_hybrid.semisupervised import VisitPassBank,validate_unlabeled
 try:validate_unlabeled({'y':torch.zeros(1)})
 except ValueError:pass
 else:raise AssertionError('U truth schema accepted')
 x=torch.randn(4,2);y=torch.zeros(4,dtype=torch.long);loss,parts,detail=objective(s,x,y,x,e,anchor,['a','b','c','d'],torch.ones(4,dtype=torch.long),VisitPassBank(2,.90),1111,1);loss.backward()
 if not torch.isfinite(loss) or e.weight.grad is not None or not all(torch.isfinite(p.grad).all() for p in s.parameters()):raise AssertionError('Finite student gradient/teacher isolation')
 return dict(status='PASS',fixed_config=True,truth_schema_closed=True,detached_targets=True,EMA_version_update=True,handoff_zero_after20=True,gradient=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
