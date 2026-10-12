"""Unchanged source-U soft KL; only registered engineering terms differ."""
from .base_objective import objective as base_objective,BASE_WEIGHTED_KEYS,AUDIT_ONLY_KEYS
from .update import rng_state
from .hidden_mix import forward as hidden_forward
from .update import set_rng
from torch.nn import functional as F

def objective(c,model,bank,L,li,U,ui,step,device):
    clean_rng=rng_state() if c['engineering']['hidden_mix'] or c['engineering']['clean_replay'] else None
    total,parts,weighted,detail=base_objective(c,model,bank,L,li,U,ui,step,device)
    if c['engineering']['hidden_mix'] and step>c['semi']['warmup_updates']:
        loss,info=hidden_forward(c,model,L['x'][li].to(device),L['y'][li].to(device),step,clean_rng)
        parts['style_ce']=loss;weighted['style_ce']=loss*c['engineering']['mix_weight']*detail['ramp']
        detail['hidden_mix']=info
        total=sum(weighted[k] for k in BASE_WEIGHTED_KEYS)
    if c['engineering']['clean_replay'] and step>c['semi']['warmup_updates']:
        end=rng_state()
        try:
            set_rng(clean_rng);logits=model(L['x'][li].to(device))
        finally:set_rng(end)
        loss=F.cross_entropy(logits,L['y'][li].to(device))
        parts['style_ce']=loss;weighted['style_ce']=loss*c['engineering']['mix_weight']*detail['ramp']
        detail['clean_replay']=dict(source_role='L_s',extra_forward_calls=1,packets=len(li),main_rng_restored=True,dropout_clean_replay=True,identity_transform=True,loss=float(loss.detach()))
        total=sum(weighted[k] for k in BASE_WEIGHTED_KEYS)
    return total,parts,weighted,detail
