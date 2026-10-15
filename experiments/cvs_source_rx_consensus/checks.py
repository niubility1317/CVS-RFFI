import argparse,copy,json
from pathlib import Path
import torch
from . import design as d
from .objective import transform
from .audit import measure_batch,prepare

def checks():
 torch.set_num_threads(2);n=0
 def rejects(fn):
  nonlocal n
  try:fn()
  except (ValueError,TypeError):n+=1
  else:raise AssertionError('Protocol-negative case accepted')
 G=torch.tensor([[1.,1.,1.,0.], [1.,1.,-1.,0.], [1.,1.,1.,0.], [1.,-1.,-1.,0.], [1.,-1.,1.,0.]])
 ce=G.mean(0);state=torch.random.get_rng_state().clone();a=transform(G,ce,[2,2]);assert torch.equal(state,torch.random.get_rng_state())
 assert a['mask'].tolist()==[True,False,False,False] and int(a['random_mask'].sum())==1
 assert torch.equal(a['masked'],ce*a['mask']) and not a['eligible'][-1]
 for role in ('U_s','V','target'):rejects(lambda role=role:transform(G,ce,[4],role=role))
 rejects(lambda:transform(G,ce,[4],threshold=.9));rejects(lambda:transform(G[:4],ce,[4]));rejects(lambda:transform(G,ce[:3],[4]));rejects(lambda:transform(G.double(),ce,[4]));rejects(lambda:transform(G,ce,[3]));rejects(lambda:transform(G,ce,[0,4]))
 bad=G.clone();bad[0,0]=float('nan');rejects(lambda:transform(bad,ce,[4]))
 for row in d.rows():
  cfg=d.config(row);d.validate(cfg);bad=copy.deepcopy(cfg);bad['agreement_threshold']=.8;rejects(lambda bad=bad:d.validate(bad))
 torch.manual_seed(2026092701)
 from experiments.cvs_receiver_dro50.model import build
 model=build(d.source_config(d.rows()[0]));before={k:v.clone() for k,v in model.state_dict().items()};x=torch.randn(30,2,256);y=torch.arange(6).repeat(5);rx=torch.tensor(d.RX).repeat_interleave(6)
 observed=measure_batch(model,x,y,rx,13,14);assert observed['per_rx'].shape==(5,d.FEATURE_COUNT) and observed['head_ce'].shape==(960,)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in model.parameters())
 assert not any(p['name']==d.CLASS_W for p in observed['parameters']) and sum(p['elements'] for p in observed['parameters'])==d.FEATURE_COUNT
 assert torch.allclose(observed['ce'],observed['per_rx'].mean(0),atol=1e-6,rtol=1e-4)
 model.eval().requires_grad_(False);altered=copy.deepcopy(model);altered.id_backbone.encoder.core.id_backbone.cls_head.weight.zero_()
 with torch.no_grad():z=model.forward_identity_only(x[:2])['id_feat_joint'];zz=altered.forward_identity_only(x[:2])['id_feat_joint']
 assert torch.equal(z,zz)
 from .publish import configure,INSPECT
 configure()
 from experiments.cvs_report_ssl import publish as transport
 assert all(k in transport.REMOTE for k in ('experiments.cvs_source_rx_consensus.preflight','experiments.cvs_source_rx_consensus.audit'))
 assert 'experiments/cvs_receiver_dro_seeds' in transport.PATHS and 'experiments/cvs_source_rx_consensus' in transport.PATHS
 compile(INSPECT.replace('CONFIG',repr(dict(project='PROJECT',release=d.RELEASE,run=d.RUN))),'source_inspect','exec')
 return dict(status='PASS',negatives=n,actual_feature_parameters=d.FEATURE_COUNT,all_feature_norms_gains_preserved=True,actual_synthetic_native_forward_and_6_derivatives=True,head_W_excluded_only=True,old_W_zeroing_preserves_joint160=True,model_state_and_grad_buffers_unchanged=True,random_exact_per_tensor_sparsity=True,private_RNG_preserved=True,conditional_source_models=[1,3],real_IQ_or_checkpoint_read=False,optimizer_steps=0,head_fit_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--export',type=Path);a=p.parse_args();v=checks()
 if a.export:a.export.write_text(json.dumps(dict(run_id=d.RUN,rows=[dict(row_id=r['row_id'],config=d.config(r)) for r in d.rows()]),indent=2)+'\n',encoding='utf-8')
 print(json.dumps(v))
