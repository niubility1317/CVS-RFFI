import json,torch
from . import design as d

def main():
 from .checks import checks
 from .predict import source_proof,own_model
 torch.set_num_threads(2);check=checks();c=d.config(d.rows()[0]);source_proof(c);m=own_model(c,'cpu');saved={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():out=m.forward_identity_only(torch.zeros(2,2,256))
 if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or not torch.isfinite(out['tx_logits']).all() or not all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items()):raise ValueError('Real old checkpoint no-query smoke failed')
 p=d.BASE.parent/'20261014-phase1-source-style-profile-manysig-s01-m1-r01'
 done=d.read(p/'completion.json');q=d.read(p/'SOURCE_STYLE-s2026092701'/'results.json')
 if done.get('status')!='ANALYZED' or q['status']!='VERIFIED' or q['source_packets']!=6300 or not q['model_parameters_buffers_unchanged'] or q['query_read'] or q['head_fit'] or q['optimizer_steps']:raise ValueError('Fixed source evidence incomplete')
 value=dict(status='PASS',checks=check,source_evidence_run=q['run_id'],real_checkpoint_no_query_smoke=True,query_read=False,head_fit=False,training_inherits_checkpoint=False)
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing immutable preflight')
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
