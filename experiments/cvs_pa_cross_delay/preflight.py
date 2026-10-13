import json,torch
from . import design as d

def main():
 from .checks import checks
 from .pipeline_checks import checks as pipeline_checks
 from experiments.cvs_default_head.checks import checks as head_checks
 from experiments.cvs_frequency_context50.predict import source_proof,own_model
 from experiments.cvs_default_head.predict import logits
 torch.set_num_threads(2);integration=checks();pipeline=pipeline_checks();head=head_checks()
 c=d.baseline_config(d.rows()[0]);source_proof(c);model=own_model(c,'cpu')
 z=logits(model,torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real default baseline no-query smoke failed')
 for r in d.rows():d.validate(d.config(r))
 result=dict(status='PASS',real_checkpoint_no_query_smoke=True,old_weights_used='smoke and frozen baseline comparison only',training_inherits_old_weights=False,query_read=False,integration=integration,pipeline=pipeline,default_head=head)
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Existing preflight artifact')
 d.write(p,result);print(json.dumps(result))
if __name__=='__main__':main()
