import json,torch
from . import design as d
def main():
 from .checks import checks
 from experiments.cvs_feature_multiscale50.predict import source_proof,own_model
 from experiments.cvs_default_head.predict import logits
 torch.set_num_threads(2);result=checks()
 for row in d.rows():source_proof(d.source_config(row))
 model=own_model(d.source_config(d.rows()[0]),'cpu')
 with torch.no_grad():value=logits(model,torch.zeros(2,2,256))
 if value.shape!=(2,6) or not torch.isfinite(value).all():raise ValueError('Real default source checkpoint smoke failed')
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing source preflight')
 result=dict(status='PASS',checks=result,real_checkpoint_no_query_smoke=True,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
 d.write(d.ROOT/'preflight.json',result);print(json.dumps(result))
if __name__=='__main__':main()
