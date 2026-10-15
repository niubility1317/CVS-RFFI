import json,torch
from . import design as d
from .checks import checks
from experiments.cvs_rx_cagrad50.checks import real_gradient_check
from experiments.cvs_receiver_dro50 import design as parent,predict as api
def main():
 torch.set_num_threads(2);local=checks();d.require_source();c=parent.config(next(r for r in parent.training_rows() if r['arm']=='SIG_TIME'));m=api.own_model(c,'cpu').eval().requires_grad_(True);before={k:v.clone() for k,v in m.state_dict().items()};real_gradient_check(m)
 if any(v.grad is not None for v in m.parameters()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Read-only actualcheckpointsmoke changedstate')
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Immutablepreflight exists')
 value=dict(status='PASS',checks=local,real_checkpoint_no_query_smoke=True,training_inherits_checkpoint=False,source_first_model_qualified=True,full195041_gradient_formula=True,original_defaultW_gradient=True,head_fit_updates=0,optimizer_updates=0,query_read=False);d.write(p,value);print(json.dumps(value))
if __name__=='__main__':main()
