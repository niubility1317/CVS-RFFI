"""Source-only inference and nonparametric scatter; no training or target imports."""
import importlib.util,json,sys,time,traceback,types
from pathlib import Path

def bootstrap():
 root=Path(__file__).resolve().parents[2];rt=json.loads((root/'runtime.json').read_text());ref=Path(rt['reference_release'])
 if (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned source diagnostic runtime differs')
 spec=importlib.util.spec_from_file_location('_prepool_runtime',ref/'experiments/cvs_feature_head_joint30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 namespace=types.ModuleType('experiments.cvs_prepool_source');namespace.__path__=[str(root/'experiments/cvs_prepool_source')];sys.modules[namespace.__name__]=namespace

def main():
 bootstrap()
 import numpy as np
 import torch
 from experiments.cvs_prepool_source import design as d
 from experiments.cvs_prepool_source.probe import Probe
 from experiments.cvs_prepool_source.scatter import halves,scatter
 from experiments.cvs_feature_head_joint30 import model as api
 from experiments.cvs_feature_adapters30.data import diagnostic_indices
 from experiments.cvs_F_nontrue20.channel import tensor_iq
 from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
 torch.set_num_threads(2);d.BASE.mkdir(parents=True,exist_ok=False);d.write(d.BASE/'config.json',d.config());start=time.perf_counter()
 try:
  bank=d.joint.FROZEN_BASE/'observation_bank';L=torch.load(bank/'labeled.pt',map_location='cpu',weights_only=False);di=diagnostic_indices(L);ids=[L['ids'][i] for i in di];y=np.asarray(L['y'][di].tolist(),np.int64);rx=np.asarray(L['rx'][di].tolist(),np.int64);day=np.asarray(L['day'][di].tolist(),np.int64);split=halves(y,rx,day,ids,d.DIAGNOSTIC_SEED)
  if len(L['ids'])!=6300 or set(L['ids'])!=set(d.read(d.joint.SOURCE)['role_ids']['L_s']) or ids!=d.read(bank/'source_roles.json')['L_diagnostic_ids']:raise ValueError('Reuse matching source L physical role and exact saved panel order only')
  raw=L['x'][di];fresh=np.load(bank/'same_L_new_channel.npy',mmap_mode='r',allow_pickle=False)
  if fresh.shape!=(2,2,1800,2,256):raise ValueError('Existing source L mid/low observation shape differs')
  inputs=[('raw',raw)]+[(f'{name}_draw{a}',fresh[a,b]) for a in range(2) for b,name in enumerate(('mid','low'))];rows={};NN=0
  with numerical_context(d.joint.FULL_FP32_POLICY),torch.no_grad():
   flags=actual_flags()
   for name in d.config()['models']:
    m=api.parent('cpu') if name=='B0_FINAL20' else api.own_model(d.joint.config(next(r for r in d.joint.rows() if r['arm']=='FULL_As')),'cpu');m.eval();initial={k:v.clone() for k,v in m.state_dict().items()};probe=Probe(m.id_backbone.encoder.core.id_backbone);views={};saved={}
    for view,x in inputs:
     descriptors=dict(pooled=[],removed=[]);pred=[]
     for i in range(0,1800,180):
      z=x[i:i+180] if torch.is_tensor(x) else tensor_iq(x[i:i+180],'cpu');output=m.forward_identity_only(z);pred.extend(output['tx_logits'].argmax(1).tolist());v=probe.take()
      for key in descriptors:descriptors[key].extend(v[key])
      NN+=len(z)
     views[view]=dict(count=1800,default_head_accuracy=float((np.asarray(pred)==y).mean()),**{key:scatter(values,split) for key,values in descriptors.items()})
     for key,values in descriptors.items():saved[view+'_'+key]=np.asarray(values,np.float32)
     d.write(d.BASE/'progress.json',dict(model=name,view=view,NN=NN,seconds=time.perf_counter()-start));print(json.dumps(dict(model=name,view=view,NN=NN)),flush=True)
    probe.close()
    if any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or probe.calls!=50:raise ValueError('Source inference mutated state or call budget differs')
    np.savez_compressed(d.BASE/(name+'_descriptors.npz'),ids=np.asarray(ids),y=y,rx=rx,day=day,**saved)
    rows[name]=dict(views=views,pool_reconstruction_max_abs_error=probe.max_error,actual_prepool_shape=[48,256],actual_t1_input_shape=[48,128],fit_updates=0,parameters_unchanged=True,head='original checkpoint default cosine head')
    del m,initial
  if NN!=18000:raise ValueError('Fixed diagnostic NN budget differs')
  supportive=all(rows[model]['views'][view]['removed']['cross_half_TX_scatter_sum']>0 for model in rows for view in ('low_draw0','low_draw1'))
  result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,rows=rows,NN=NN,model_seed_count=1,fit_updates=0,new_physical_simulations=0,U_used=False,V_used=False,target_access=False,formal_scores_read=False,backend_flags=flags,supports_local_removed_TX_variation=supportive,interpretation='Local reproducible TX-related variation removed by first time pooling; does not establish unique necessary identity information or raw-bypass performance' if supportive else 'Fixed diagnostic does not support the preregistered local removed-TX-variation condition;no descriptor/split/draw sweep',wall_seconds=time.perf_counter()-start,goal_complete=False)
  d.write(d.BASE/'results.json',result);d.write(d.BASE/'complete.json',dict(status=result['status'],NN=NN,fit_updates=0,target_access=False,goal_complete=False));print(json.dumps(dict(status=result['status'],NN=NN,supports_local_removed_TX_variation=supportive)),flush=True)
 except Exception:
  d.write(d.BASE/'failure.json',dict(status='TECHNICAL_FAILURE',traceback=traceback.format_exc(),target_access=False));raise

if __name__=='__main__':main()
