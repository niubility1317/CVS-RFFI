"""Source-only worst-two-view objective for matched physical records."""
import torch
from experiments.phase1_volterra_aligned.objective import aligned_iq_error, source_loss

COUNTS = [0,0,0,0]

def worst_two_view_loss(per_view, groups):
    if per_view.ndim!=1 or groups.shape!=per_view.shape or len(per_view)==0 or len(per_view)%3:
        raise ValueError('Three equal paired source-view batches are required')
    size=len(per_view)//3
    g=groups.reshape(3,size)
    if not torch.equal(g[0],g[1]) or not torch.equal(g[0],g[2]):
        raise ValueError('Source receiver groups must agree across paired views')
    worst=per_view.reshape(3,size).topk(2,dim=0).values.mean(0)
    return torch.stack([worst[g[0]==group].mean() for group in g[0].unique()]).mean()

def viewcvar_source_objective(reader, output, anchor, clean, clean_features, groups):
    _,terms=source_loss(reader,output,clean,clean_features,groups)
    with torch.no_grad():anchor_error=aligned_iq_error(anchor.detach(),clean)
    excess=(terms['aligned']-anchor_error).clamp_min(0)
    per=terms['aligned']+0.05*terms['power']+0.2*terms['feature']+0.2*terms['distill']+0.05*terms['margin']+5.0*excess
    result=worst_two_view_loss(per,groups)
    chosen=per.detach().reshape(3,-1).topk(2,dim=0).indices
    COUNTS[0]+=chosen.shape[1]
    for j in range(3):COUNTS[j+1]+=int((chosen==j).sum())
    return result,dict(terms,view_cvar=result)
