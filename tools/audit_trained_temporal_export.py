"""Real fixed-final temporal weights and exact class-weight-free export."""
import argparse,json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
REMOTE=r'''
import sys,json,time,torch
from pathlib import Path
c=CONFIG;root=Path(c['project'])/'releases'/c['release'];sys.path[:0]=[str(root),str(root/'code')]
from experiments.cvs_time_factorial50 import design as d
from experiments.cvs_time_factorial50.provenance import source_proof,own_model
from experiments.cvs_time_factorial50.model import classifier
from experiments.cvs_feature_restitution.features import layout
torch.set_num_threads(1);x=torch.randn(8,2,256,generator=torch.Generator().manual_seed(2026101413));records=[]
for row in d.rows():
 if row['arm']=='BASE_REF':continue
 cfg=d.config(row);source=Path(cfg['output_root'])
 if not (source/'completion.json').is_file():continue
 source_proof(cfg);initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];m=own_model(cfg,'cpu');core,prefix,_=layout(m);before={n:v.detach().clone() for n,v in m.state_dict().items()};updates={}
 for layer in ('t1','t2','t3'):
  params={}
  for short,p in getattr(core,layer).named_parameters():
   name=prefix+'.'+layer+'.'+short;q=initial[name];delta=p.detach()-q
   params[short]=dict(values=p.numel(),changed=int((p!=q).sum()),l2_change=float(delta.norm()),max_abs_change=float(delta.abs().max()))
   if params[short]['changed']==0:raise ValueError('Temporal parameter tensor did not update '+row['row_id']+'/'+name)
  updates[layer]=params
 with torch.no_grad():
  actual=m.forward_identity_only(x)['id_feat_joint'];pure=m.id_backbone.encoder.features(x)
  if not torch.equal(actual,pure):raise ValueError('Actual z differs from pure export')
  h=classifier(m);W=h.weight.detach().clone();h.weight.fill_(1);modified=m.id_backbone.encoder.features(x);del h.weight;deleted=m.id_backbone.encoder.features(x)
  if not torch.equal(actual,modified) or not torch.equal(actual,deleted):raise ValueError('z depends on class W')
  h.weight=torch.nn.Parameter(W,requires_grad=False)
  for name,value in m.state_dict().items():
   if not torch.equal(value,before[name]):raise ValueError('Export diagnostic changed persistent feature state '+name)
  shadow=own_model(cfg,'cpu');shadow.load_state_dict(initial,strict=True);z0=shadow.forward_identity_only(x)['id_feat_joint'];movement=float((actual-z0).norm(dim=1).mean())
  if movement<=1e-7:raise ValueError('Actual trained feature did not change')
 records.append(dict(**row,updates=updates,temporal_parameter_count=sum(p.numel() for l in ('t1','t2','t3') for p in getattr(core,l).parameters()),pure_export_equal_actual=True,W_modified_deleted_invariant=True,all_feature_params_buffers_unchanged_by_diagnostic=True,trained_vs_initial_z_delta_on_fixed_artificial_packets=movement,head_fit_updates=0))
print(json.dumps(dict(run_id=d.RUN,status='VERIFIED',execution_commit=(root/'release_commit.txt').read_text().strip(),rows=records,expected_new_models=12,all_models_complete=len(records)==12,query_or_source_IQ_read=False,truth_read=False,optimizer_updates=0,head_fit_updates=0,scope='actual initial/final source weights; exact feature boundary and movement on fixed artificial packets, not recognition or transfer proof',read_at=time.time())))
'''
def audit(output):
 from experiments.cvs_time_factorial50 import design as d
 payload=REMOTE.replace('CONFIG',repr(dict(project=d.PROJECT,release=d.RELEASE)))
 wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES=''))\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
 v=json.loads(ssh(wrapper));p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=v['status'],completed_models=len(v['rows']),all_models_complete=v['all_models_complete'])));return v
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output)
