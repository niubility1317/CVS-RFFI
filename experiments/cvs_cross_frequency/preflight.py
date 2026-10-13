import json,torch
from . import design as d
def main():
 from .checks import checks
 from .relations_checks import checks as relation_checks
 from .readout_checks import checks as readout_checks
 from experiments.cvs_frequency_context50.predict import source_proof,own_model
 from experiments.cvs_transfer_features.model import from_phase1
 torch.set_num_threads(2)
 primitive=relation_checks();integration=checks();readouts=readout_checks()
 c=d.baseline_config(d.rows()[0]);source_proof(c);model=own_model(c,'cpu')
 with torch.no_grad():
  z=model.forward_identity_only(torch.zeros(2,2,256));pure=from_phase1(model)(torch.zeros(2,2,256))
 if z['tx_logits'].shape!=(2,6) or not torch.isfinite(z['tx_logits']).all() or not torch.equal(pure,z['id_feat_joint']):raise ValueError('Real baseline no-query smoke differs')
 for r in d.rows():d.validate(d.config(r))
 result=dict(status='PASS',real_checkpoint_no_query_smoke=True,old_weights_used='smoke and frozen baseline comparison only',training_inherits_old_weights=False,query_read=False,relations=primitive,integration=integration,readouts=readouts)
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Existing preflight artifact')
 d.write(p,result);print(json.dumps(result))
if __name__=='__main__':main()
