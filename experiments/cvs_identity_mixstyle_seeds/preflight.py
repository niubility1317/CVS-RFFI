import json,torch
from . import design as d
from .predict import source_proof,own_model

def qualified():
 from experiments.cvs_identity_mixstyle50 import design as first
 q=d.read(first.BASE/'source_qualification.json')
 if q['status']!='PASS' or q['run_id']!=first.RUN or q['seed']!=2026092701 or not all(q['checks'].values()) or not q['source_only'] or q['target_results_read'] or q['target_truth_read'] or q['U_truth_read'] or q['head_fit']:raise ValueError('Original fixed source criteria did not qualify expansion')
 return dict(status='PASS',source_qualification_run=first.RUN,target_scores_used=False,fixed_decision_commit='9e79dc413967aee78fa87e5d0f6c7408c44e4efc')

def main():
 torch.set_num_threads(2);q=qualified()
 for row in d.rows():d.validate(d.config(row))
 c=d.config(d.rows()[0]);source_proof(c);model=own_model(c,'cpu');saved={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():v=model.forward_identity_only(torch.zeros(2,2,256))
 if v['tx_logits'].shape!=(2,6) or v['id_feat_joint'].shape!=(2,160) or not torch.isfinite(v['tx_logits']).all() or not all(torch.equal(s,model.state_dict()[k]) for k,s in saved.items()):raise ValueError('Correspondingseed real checkpoint no-query smoke failed')
 value=dict(status='PASS',source_qualification=q,real_checkpoint_no_query_smoke=True,source_seed=2026092702,query_read=False,head_fit=False,training_inherits_checkpoint=False)
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing immutable preflight')
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
