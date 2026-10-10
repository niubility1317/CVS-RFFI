"""Integration checks: actual CVS adapter source audit, fixed V, source ranking."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from . import design as d
from .diagnostics import action_audit,reference_config
from .validation import evaluate_source_stress,source_ranking
from experiments.cvs_phase1_overlay.model import native_modules,Phase1IdentityAdapter


def _data(count,generator,device,prefix):
 x=torch.randn(count,2,256,generator=generator).to(device)
 x=x/x.square().sum(1).mean(1).sqrt()[:,None,None]
 return dict(x=x,y=torch.arange(count,device=device)%6,
             rx=torch.tensor([1,3,4,6,8],device=device).repeat((count+4)//5)[:count],
             day=torch.arange(count,device=device)%4,ids=[prefix+str(i) for i in range(count)],
             condition=['clean']*count)


class _FastClassifier(torch.nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('weights',torch.arange(12,dtype=torch.float32).reshape(2,6)/12)
 def forward(self,x):return x.mean(2)@self.weights


def run_checks(device='cpu'):
 torch.set_num_threads(2);native_modules()
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(6431)
  identity=Phase1IdentityAdapter(SimpleNamespace(num_classes=6,input_len=256)).to(device)
 identity.train();before={k:v.clone() for k,v in identity.state_dict().items()}
 generator=torch.Generator().manual_seed(6433)
 fit,audit=_data(12,generator,device,'fit'),_data(12,generator,device,'audit')
 rng=torch.random.get_rng_state().clone()
 cuda_rng=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
 reports=[]
 for heldout,edges in [(None,False),(None,True),(0,True)]:
  report=action_audit(identity,fit,audit,6437,heldout=heldout,edges=edges,steps=2)
  assert {r['parameter_source'] for r in report['action_records']}=={'true','analytic','learned'}
  assert {r['kind'] for r in report['action_records']}=={'linear','temporal'}
  assert len(report['fit_logs'])==2 and len(report['proposal_records'])==2
  assert all(abs(r['measured_added_noise_snr_db']-r['snr_db'])<1e-4 for r in report['noise_records'])
  assert report['fit_logs'][0]['order']=='LT' and report['fit_logs'][1]['order']=='TL'
  if heldout is not None:assert heldout not in report['fit_tx'] and heldout in report['audit_tx']
  reports.append(report)
 assert identity.training and all(torch.equal(v,identity.state_dict()[k]) for k,v in before.items())
 assert torch.equal(rng,torch.random.get_rng_state())
 if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
 for seed in d.SEEDS:
  config=reference_config(seed)
  assert config['run_id']=='20261009-phase1-multi-state-action-manysig-m36-r02'
  assert config['arm']=='native' and config['model_seed']==seed
 # Every existing V packet is counted; cheap fixture isolates validation from
 # identity compute while the audit above exercises the actual identity path.
 fixture=_FastClassifier().train()
 def loader(count=27000):
  private=torch.Generator().manual_seed(711)
  for start in range(0,count,1000):
   n=min(1000,count-start)
   x=torch.randn(n,2,256,generator=private)
   labels=(torch.arange(n)+start)%6
   meta=dict(rx_i=torch.tensor([1,3,4,6,8]).repeat((n+4)//5)[:n],
             day_i=(torch.arange(n)+start)%4)
   yield x,labels,torch.zeros(n,dtype=torch.long),meta
 c=d.config(d.rows()[0]);expected_views=set(d.STRESS['views'])
 with tempfile.TemporaryDirectory(prefix='cvs-risk-source-check-') as temporary:
  out=Path(temporary)
  result=evaluate_source_stress(fixture,loader(),'cpu',c,out)
  assert fixture.training and set(result['views'])==expected_views
  assert result['views']['clean']['applied_packets']==0
  assert all(v['count']==27000 for v in result['views'].values())
  assert result['views']['LTL_heldout']['heldout_composition']
  assert result['views']['curvature_pressure']['artificial_curvature_pressure']
  failed=False
  try:evaluate_source_stress(fixture,loader(5),'cpu',c,out)
  except ValueError as error:failed='Incomplete source V' in str(error)
  assert failed and fixture.training
  # Exercise the real ranking reader using only source files and all 3 seeds.
  rows=[dict(row_id=a+'-s'+str(s),arm=a,model_seed=s) for s in d.SEEDS for a in ('native','L')]
  for row in rows:
   root=out/row['row_id'];root.mkdir()
   v=copy.deepcopy(result);v.update(row)
   if row['arm']=='L':
    for name,metrics in v['views'].items():
     if name!='clean':metrics['accuracy']+=.02;metrics['worst_rx']+=.02
   d.write(root/'source_stress.json',v)
   d.write(root/'auxiliary_cost.json',dict(auxiliary_parameters=0 if row['arm']=='native' else 500))
  with patch.object(d,'rows',lambda:rows),patch.object(d,'config',lambda row:dict(output_root=str(out/row['row_id']))),patch.object(d,'BASE',out):
   ranked=source_ranking()
  assert ranked['selected']=='L' and not ranked['target_scores_consumed']
 return dict(status='PASS',device=device,actual_identity_adapter=True,
             parameter_sources=['true','analytic','learned'],heldout_tx=0,
             conditional_and_main_edges=True,noise_snr_verified=True,proposal_true_IQ=True,
             identity_and_RNG_unchanged=True,source_V_count=27000,
             incomplete_V_rejected=True,source_ranking_all_three_seeds=True,
             pressure_views=sorted(expected_views),old_native_checkpoint_config_unchanged=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
 args=parser.parse_args();print(json.dumps(run_checks(args.device),indent=2))
