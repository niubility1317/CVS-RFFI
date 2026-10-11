import argparse,torch
from . import design as d
from .metrics import margin,summarize
def check():
 x=torch.arange(8).float()[None].repeat(4,1);p=torch.full_like(x,1/8)
 a=summarize(x,x,p,1e-6);b=summarize(-x,x,p,1e-6);z=summarize(torch.zeros_like(x),x,p,1e-6)
 assert a['pair_concordance']==1 and a['top1_hit_on_informative']==1 and a['MSE']==0
 assert b['pair_concordance']==0 and b['top1_hit_on_informative']==0
 assert z['pair_concordance']==.5 and z['MSE_relative_to_zero']==1
 reverse_zero=summarize(torch.zeros_like(x),x.flip(1),p,1e-6)
 assert z['top1_hit_on_informative']==reverse_zero['top1_hit_on_informative']==.125
 assert z['top1_actual_regret']==reverse_zero['top1_actual_regret']==3.5
 assert z['top_prediction_tie_count_mean']==8
 assert a['policy_minus_uniform']==0 and a['fraction_oracle_gap_captured']==0 and a['total_variation_from_uniform_mean']==0
 probs=torch.zeros_like(x);probs[:,-1]=1.;assert summarize(x,x,probs,1e-6)['fraction_oracle_gap_captured']==1
 ties=summarize(x,torch.ones_like(x),p,1e-6);assert ties['eligible_pairs']==0 and ties['pair_concordance'] is None
 assert torch.equal(margin(torch.tensor([[2.,1.,3.]]),torch.tensor([0])),torch.tensor([-1.]))
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert all(c[k].startswith('/home/') and '\\' not in c[k] for k in ('output','source_checkpoint','auxiliary_checkpoint'))
 return dict(status='PASS',rows=3,ranking_perfect_reversed_tie_verified=True,uniform_oracle_expectations_verified=True,query_access=False)
def preflight():
 from .run import load,evaluate_role
 value=check();evidence=[]
 for row in d.rows():
  c=d.config(row);model,heads,L,V=load(c,'cpu');tiny={k:(v[:2] if k!='ids' else v[:2]) for k,v in L.items()};_,s=evaluate_role(model,heads,tiny,True,c,'cpu')
  assert s['CE']['count']==2
  idx=torch.arange(2);baseline=heads.candidates(tiny,idx,False);poison=dict(tiny,y=torch.full_like(tiny['y'],999));assert torch.equal(baseline,heads.candidates(poison,idx,False))
  evidence.append(dict(row_id=row['row_id'],checkpoint=c['source_checkpoint'],real_L_count=2,all8candidate_forward=True,TX_blind_bank_poison_test=True))
 value['frozen_checkpoint_checks']=evidence;return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2);d.write(a.output,preflight() if a.remote_preflight else check())
