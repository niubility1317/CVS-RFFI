"""Read-only endpoint gradient and restored virtual-step diagnostics on legal L/U."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from experiments.cvs_report_ssl.model import groups
from experiments.cvs_report_ssl.source import objective
from experiments.cvs_report_ssl.sampling import DomainStream

class StudentSnapshot:
    """Frozen exact current student copy; never performs temporal averaging."""
    def __init__(self,student):
        self.model=deepcopy(student).eval().requires_grad_(False)

    @torch.no_grad()
    def probabilities(self,x):
        return self.model(x).softmax(1).detach()


def flatten_grad(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(parameters,gradients)])

def cosine(a,b):
    norm=a.norm()*b.norm()
    return float(a.dot(b)/norm) if float(norm)>1e-20 else None

def audit(config,model,teacher,heads,L,U,device):
    c=config['source_config'];settings=config['gradient'];pg,_=groups(model,c);model.eval();heads.eval()
    snapshot=StudentSnapshot(model)
    names={id(p):n for n,p in model.named_parameters()}
    params=[p for g in pg for p in g['params'] if not names[id(p)].endswith(('low_hz_','band_hz_'))]
    lengths=[p.numel() for p in params];before=[p.detach().clone() for p in params]
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    spans={};offset=0
    for group in pg:
        for p in group['params']:
            if names[id(p)].endswith(('low_hz_','band_hz_')):continue
            spans[names[id(p)]]=(offset,offset+p.numel(),group['name']);offset+=p.numel()
    reference_norm=torch.cat([p.flatten() for p in before]).norm()
    lstream=DomainStream(L['rx'],L['day'],settings['sample_seed']);ustream=DomainStream(U['rx'],U['day'],settings['sample_seed']+1)
    rows=[]
    for batch in range(settings['batches']):
        li,_=lstream.batch(settings['L_batch']);probe,_=lstream.batch(settings['L_probe_batch']);ui,_=ustream.batch(settings['U_batch'])
        # Exclude overlap if a domain permutation wraps between the two draws.
        probe=probe[~torch.isin(probe,li)]
        if len(probe)<96:raise ValueError('Insufficient independent L probe batch')
        # This index only fixes deterministic action/noise draws and full loss weights.
        _,_,weighted,details=objective(c,model,heads,L,li,U,ui,teacher,None,8884+batch*4,device)
        _,_,direct,direct_details=objective(c,model,heads,L,li,U,ui,snapshot,None,8884+batch*4,device)
        tasks={'CE':weighted['ce'],'U_EMA':weighted['soft_identity'],'U_student':direct['soft_identity']}
        if c['enabled']['linear']:
            tasks['linear_action']=weighted['linear'];tasks['linear_identity']=weighted['linear_identity']
        gradients={k:flatten_grad(v,params) for k,v in tasks.items()}
        px=L['x'][probe].to(device);py=L['y'][probe].to(device)
        probe_loss=F.cross_entropy(model(px),py);gprobe=flatten_grad(probe_loss,params)
        base_loss=float(probe_loss.detach());row=dict(batch=batch,L_probe_count=len(probe),L_probe_before=base_loss,tasks={},
            EMA_accepted_U=details['accepted_U'],student_accepted_U=direct_details['accepted_U'],
            U_target_gradient_cosine=cosine(gradients['U_EMA'],gradients['U_student']))
        for name,g in gradients.items():
            norm=g.norm();delta=settings['virtual_relative_step']*reference_norm
            offset=0
            try:
                with torch.no_grad():
                    if float(norm)>1e-20:
                        for p,original,size in zip(params,before,lengths):
                            p.copy_(original-(delta/norm)*g[offset:offset+size].reshape_as(p));offset+=size
                    after=float(F.cross_entropy(model(px),py))
            finally:
                with torch.no_grad():
                    for p,original in zip(params,before):p.copy_(original)
            layers={}
            for n,(a,b,group) in spans.items():
                layers[n]=dict(group=group,norm=float(g[a:b].norm()),cosine_with_CE=cosine(g[a:b],gradients['CE'][a:b]),cosine_with_independent_L=cosine(g[a:b],gprobe[a:b]))
            group_results={}
            for group in pg:
                pieces=[(a,b) for a,b,gn in spans.values() if gn==group['name']]
                gg=torch.cat([g[a:b] for a,b in pieces]);gc=torch.cat([gradients['CE'][a:b] for a,b in pieces]);gl=torch.cat([gprobe[a:b] for a,b in pieces])
                group_results[group['name']]=dict(norm=float(gg.norm()),cosine_with_CE=cosine(gg,gc),cosine_with_independent_L=cosine(gg,gl))
            row['tasks'][name]=dict(loss=float(tasks[name].detach()),norm=float(norm),cosine_with_CE=cosine(g,gradients['CE']),
                cosine_with_independent_L=cosine(g,gprobe),L_probe_after=after,L_probe_change=after-base_loss,
                virtual_delta_l2=float(delta) if float(norm)>1e-20 else 0.,groups=group_results,layers=layers)
        if not all(torch.equal(p.detach(),v) for p,v in zip(params,before)):raise AssertionError('Virtual source parameter step not restored')
        if not all(torch.equal(b,buffers[n]) for n,b in model.named_buffers()):raise AssertionError('Frozen model buffers changed')
        rows.append(row)
    return dict(rows=rows,identity_updates=0,virtual_steps_restored_exactly=True,V_gradient_access=False,
        target_access=False,teacher_updates=0,no_EMA_target='exact frozen current student snapshot; weak clean target, same noisy student view',
        scope='Endpoint deterministic dense gradients; normalized virtual steps do not reproduce historical AdamW or a no-EMA training trajectory')
