import json,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .readout import fit,logits,support_indices
from experiments.cvs_transfer_features.model import from_phase1
from experiments.cvs_frequency_context50.predict import own_model,source_proof
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_wave_geometry.run import score

def data():
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_nocfo_foundation.source import load_source
 config=nd.config(nd.rows()[0]);L,U,V,contract=load_source(config)
 if U is not None or sorted(L['rx'].unique().tolist())!=list(d.SOURCE_RX):raise ValueError('Probe reads L/V only')
 return L,V,contract

@torch.no_grad()
def extract(model,features,data,device):
 values=[];original=[];maximum=0.
 for start in range(0,len(data['x']),256):
  x=data['x'][start:start+256].to(device);v=model.forward_identity_only(x);z=features.forward_features(x)
  maximum=max(maximum,float((z-v['id_feat_joint']).abs().max()))
  if not torch.equal(z,v['id_feat_joint']):raise ValueError('Class-free feature boundary differs on actual source IQ')
  values.append(F.normalize(z,dim=1,eps=1e-4).cpu());original.append(v['tx_logits'].cpu())
 return torch.cat(values),torch.cat(original),maximum

def run(rowid):
 torch.set_num_threads(2);row=next(r for r in d.rows() if r['row_id']==rowid);c=d.config(row);d.validate(c)
 out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();device='cuda:0'
 L,V,contract=data();source_proof(c['source_config'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=own_model(c['source_config'],device);features=from_phase1(model).to(device).eval().requires_grad_(False)
  before={k:t.clone() for k,t in features.state_dict().items()}
  if any(n.endswith('cls_head.weight') for n,_ in features.named_parameters()):raise ValueError('Old class weights exported')
  z,unused,el=extract(model,features,L,device);q,native,ev=extract(model,features,V,device)
  y=L['y'].long();records=[];head_logs=[]
  records.append(dict(readout='ORIGINAL_HEAD',**score(native,V['y'],V['rx'],V['day'])))
  w,audit=fit(z,y,row['model_seed']+60013);head_logs.append(dict(readout='FRESH_COSINE_ALL_L',**audit))
  records.append(dict(readout='FRESH_COSINE_ALL_L',**score(logits(q,w),V['y'],V['rx'],V['day'])))
  proto=torch.stack([z[y==cls].mean(0) for cls in range(6)])
  records.append(dict(readout='COSINE_PROTO_ALL_L',**score(logits(q,proto),V['y'],V['rx'],V['day'])))
  for held in d.SOURCE_RX:
   query=V['rx']==held
   for support_seed in d.SUPPORT_SEEDS:
    for k in d.K_VALUES:
     take=support_indices(y,L['rx'],held,k,support_seed)
     w,audit=fit(z[take],y[take],support_seed+held*100+k,batch_size=6*k)
     head_logs.append(dict(readout='FRESH_COSINE_CROSS_RX_K',held_rx=held,support_seed=support_seed,k=k,fit_ids=[L['ids'][i] for i in take.tolist()],**audit))
     records.append(dict(readout='FRESH_COSINE_CROSS_RX_K',held_rx=held,support_seed=support_seed,k=k,**score(logits(q[query],w),V['y'][query],V['rx'][query],V['day'][query])))
  if any(not torch.equal(t,features.state_dict()[k]) for k,t in before.items()):raise ValueError('Frozen feature parameters/buffers changed')
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,records=records,head_training=head_logs,feature_updates=0,U_opened=False,target_access=False,old_class_weights_absent=True,feature_parameters_buffers_unchanged=True,actual_source_feature_max_difference=max(el,ev),fit_L_count=len(y),V_count=len(q),feature_parameters=sum(p.numel() for p in features.parameters()),elapsed_seconds=time.perf_counter()-start)
  d.write(out/'source_contract.json',contract);d.write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),checkpoint=str(Path(c['source_config']['output_root'])/'final_ssdg.pth'),old_W_discarded=True))
  d.write(out/'completion.json',done);print(json.dumps(dict(row_id=rowid,status=done['status'],seconds=done['elapsed_seconds'],target_access=False)),flush=True)
 return done
