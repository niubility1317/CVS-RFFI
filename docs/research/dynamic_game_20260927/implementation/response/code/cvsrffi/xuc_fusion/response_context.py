"""Legal rotating source-L monitoring; no target, U identity or V access."""
from contextlib import contextmanager
from collections import defaultdict
import torch
from cvsrffi.game_tracking.state import RNGState,clone_buffers,restore_buffers

@contextmanager
def passive(model):
    rng=RNGState.capture();buffers=clone_buffers(model);flags={m:m.training for m in model.modules()}
    try:
        model.eval()
        yield
    finally:
        # Do not bump unchanged buffer version counters while a caller still
        # holds an autograd graph (e.g. an XT inner step through BatchNorm).
        with torch.no_grad():
            current=dict(model.named_buffers())
            if current.keys()!=buffers.keys():raise RuntimeError('passive forward changed buffer layout')
            for name,value in buffers.items():
                if not torch.equal(current[name],value):current[name].copy_(value)
        rng.restore()
        for m,flag in flags.items():m.training=flag

class SourceMonitor:
    def __init__(self, source):
        self.dataset=source.train
        self.cells=defaultdict(list)
        for i in range(len(self.dataset)):
            _,y,d,meta=self.dataset[i]
            if int(y)<0:raise ValueError('monitor must be labeled source L')
            self.cells[(int(y),int(meta['rx_i']),int(meta['day_i']))].append(i)
        if len(self.cells)!=90:raise ValueError('requires 6 TX x 5 RX x 3 day source L cells')
        axes=[{key[i] for key in self.cells} for i in range(3)]
        if [len(a) for a in axes]!=[6,5,3]:raise ValueError('wrong TX/RX/day monitoring universe')
    def batch(self, step, device):
        ids=[rows[step%len(rows)] for _,rows in sorted(self.cells.items())]
        rows=[self.dataset[i] for i in ids]
        return dict(x=torch.stack([r[0] for r in rows]).to(device),
            y=torch.tensor([r[1] for r in rows],device=device),
            domain=torch.tensor([r[2] for r in rows],device=device),
            ids=[r[3]['sample_id'] for r in rows])

def risk_vector(model, views, y):
    risks=[]
    for x in views.values():
        logits=model(x,return_aux=True)['tx_logits'];p=logits.softmax(-1)
        true=p.gather(1,y[:,None]).squeeze(1)
        others=p.scatter(1,y[:,None],-1.).max(1).values
        for tx in y.unique(sorted=True):risks.append(-(true-others)[y==tx].mean())
    return torch.stack(risks)

def monitor_views(batch,ctx,args):
    views={'clean':batch['x']}
    if ctx.epoch>=args.sat_cons_start_epoch:
        # Replay the exact current native batch channel, never add inactive scenes.
        from baseline_origin_sat_view import BaselineOriginSatViewAugment
        from cvsrffi.game_tracking.step_context import apply_sat_channel_for_scenario
        augment=BaselineOriginSatViewAugment(schedule=args.joint_sat_schedule,
            seed=args.joint['augmentation_seed'],apply_fn=apply_sat_channel_for_scenario)
        with torch.no_grad():result=augment.transform(batch['x'],args=args,epoch=ctx.epoch,batch_idx=ctx.batch_index)
        if result.applied:views[result.scenario]=result.x.detach()
    return views

def actual_head_input(model,x):
    captured=[]
    handle=model.adv_head.register_forward_pre_hook(lambda m,a:captured.append(a[0]))
    try:model(x,return_aux=True,grl_lambda=1.)
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('ambiguous adversarial input')
    return captured[0]

def source_observation(model,batch,views):
    """Read-only L observation; the gradient change is not a recovery gap."""
    with passive(model):
        with torch.no_grad():
            risks=risk_vector(model,views,batch['y']).detach().clone()
            features=actual_head_input(model,batch['x']).detach().clone()
        loss=torch.nn.functional.cross_entropy(model.adv_head(features),batch['domain'])
        params=list(model.adv_head.parameters())
        gs=torch.autograd.grad(loss,params,allow_unused=True)
        head_gradient=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,gs)])
    return risks,features,head_gradient
