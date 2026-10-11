import argparse,json
import torch
from . import design as d
from .anchors import fit,predict

def check():
 torch.manual_seed(321);y=torch.arange(6).repeat_interleave(40).repeat(5);rx=torch.arange(5).repeat_interleave(240);f=torch.randn(len(y),10)*.1;f[:,:6]+=torch.nn.functional.one_hot(y,6)*2
 settings=dict(d.config(d.rows()[0])['anchor']);settings['k_per_RX']=8
 for mode in settings['metric_modes']:
  state=fit(f,y,rx,settings,mode);q=torch.ones(18,6)/6;query=f[:18].clone();r=rx[:18];before=state['bank'].clone()
  p,diag=predict(state,query,r,q);assert torch.equal(before,state['bank']) and diag['cross_RX_count']==4
  changed=dict(state,bank=state['bank'].clone(),y=state['y'].clone());changed['y'][rx==0]=(changed['y'][rx==0]+1)%6
  p2,_=predict(changed,query,r,q);assert torch.equal(p['cross_RX'],p2['cross_RX']) and not torch.equal(p['same_RX'],p2['same_RX'])
  p3,_=predict(state,query[:1],r[:1],q[:1]);assert torch.allclose(p['cross_RX'][:1],p3['cross_RX'],atol=1e-6)
  assert int(p['cross_RX'].argmax(1).eq(y[:18]).sum())==18
 return dict(status='PASS',same_RX_anchor_exclusion=True,query_does_not_fit=True,query_batch_independent=True,metric_modes=2,target_access=False)

def preflight():
 from experiments.cvs_stage2_teacher_factorial.source import load_source
 from experiments.cvs_stage2_teacher_factorial.model import build
 result=check();records=[]
 for row in d.rows():
  c=d.config(row);L,U,V,contract,p=load_source(c['parent_build_config']);m=build(c['parent_build_config'],'cpu').eval()
  with torch.no_grad():out=m.forward_identity_only(L['x'][:8]);assert out['id_feat_joint'].shape==(8,160)
  records.append(dict(row_id=row['row_id'],parent=p,counts=dict(L=len(L['ids']),U=len(U['ids']),V=len(V['ids'])),E40_loaded=False))
 result['sources']=records;return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2);v=preflight() if a.remote_preflight else check();d.write(a.output,v);print(json.dumps(v))
