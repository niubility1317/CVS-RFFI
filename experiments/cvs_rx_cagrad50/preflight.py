import json,torch
from . import design as d
from .checks import checks,real_gradient_check
from experiments.cvs_receiver_dro50 import design as original,predict as api
def main():
 torch.set_num_threads(2);local=checks();c=original.config(next(r for r in original.training_rows() if r['arm']=='SIG_TIME'));m=api.own_model(c,'cpu').eval().requires_grad_(True);before={k:v.clone() for k,v in m.state_dict().items()};real_gradient_check(m)
 if any(v.grad is not None for v in m.parameters()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Realcheckpoint smoke mutatedstate')
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Immutablepreflight exists')
 v=dict(status='PASS',checks=local,real_checkpoint_no_query_smoke=True,source_checkpoint='OriginalfixedDRO SIG_TIME01final50, source_proof checked',training_inherits_checkpoint=False,full_feature_gradient_correction_verified=True,default_W_gradient_unchanged=True,query_read=False,head_fit_updates=0,optimizer_updates=0);d.write(p,v);print(json.dumps(v))
if __name__=='__main__':main()
