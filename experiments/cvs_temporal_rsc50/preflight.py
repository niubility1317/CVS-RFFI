import json
import torch
from . import design as d
from .checks import checks,exercise
from experiments.cvs_source_signature_order.probe import model

def main():
    torch.set_num_threads(2);local=checks();m=model().train().requires_grad_(True)
    before={k:v.clone() for k,v in m.state_dict().items()};flags=[p.requires_grad for p in m.parameters()]
    rng=torch.get_rng_state().clone()
    result=exercise(m,d.config(next(r for r in d.training_rows() if r['arm']=='TEMPORAL_RSC')),optimizer_update=False)
    if flags!=[p.requires_grad for p in m.parameters()] or any(p.grad is not None for p in m.parameters()):raise ValueError('No-query smoke changed flags or accumulated gradients')
    if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('No-query smoke changed model state')
    torch.set_rng_state(rng)
    path=d.ROOT/'preflight.json'
    if path.exists():raise FileExistsError('Immutable preflight exists')
    value=dict(status='PASS',checks=local,real_checkpoint_no_query_smoke=True,actual_checkpoint_check=result,
        synthetic_IQ_packets=12,source_checkpoint='Fixed source SIG_TIME final50; existing own_model source provenance checked',
        training_inherits_checkpoint=False,query_read=False,head_fit_updates=0,optimizer_updates=0)
    d.write(path,value);print(json.dumps(value))
if __name__=='__main__':main()
