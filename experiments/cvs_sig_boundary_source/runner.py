"""Source-only inference and nonparametric scatter; no training or target imports."""
import importlib.util,json,sys,time,traceback,types
from pathlib import Path

def bootstrap():
 root=Path(__file__).resolve().parents[2];rt=json.loads((root/'runtime.json').read_text());ref=Path(rt['reference_release'])
 if (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned source diagnostic runtime differs')
 spec=importlib.util.spec_from_file_location('_sig_boundary_runtime',ref/'experiments/cvs_feature_head_joint30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 namespace=types.ModuleType('experiments.cvs_sig_boundary_source');namespace.__path__=[str(root/'experiments/cvs_sig_boundary_source')];sys.modules[namespace.__name__]=namespace

def main():
 bootstrap()
 import numpy as np
 import torch
 from experiments.cvs_sig_boundary_source import design as d
 from experiments.cvs_sig_boundary_source.probe import Probe
 from experiments.cvs_sig_boundary_source.metrics import score,decision
 from experiments.cvs_feature_head_joint30 import model as api
 from experiments.cvs_feature_adapters30.data import diagnostic_indices
 from experiments.cvs_F_nontrue20.channel import tensor_iq
 from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
 torch.set_num_threads(2);d.BASE.mkdir(parents=True,exist_ok=False);d.write(d.BASE/'config.json',d.config());start=time.perf_counter();NN=0;rows={}
 try:
  with numerical_context(d.joint.FULL_FP32_POLICY),torch.no_grad():
   models={name:api.parent('cpu') if name=='B0_FINAL20' else api.own_model(d.joint.config(next(r for r in d.joint.rows() if r['arm']=='FULL_As')),'cpu') for name in d.MODELS}
   for name,m in models.items():
    with Probe(m) as p:p.take(m.forward_identity_only(torch.zeros(2,2,256)))
    NN+=2
   d.write(d.BASE/'smoke.json',dict(status='PASS',NN=4,source_IQ_read=False,query_read=False,strict_native_base_pool_joint_head_replay=True));print('SMOKE PASS real two fixed source checkpoints,4NN,no source/query IQ.Continue.',flush=True)
   bank=d.joint.FROZEN_BASE/'observation_bank';L=torch.load(bank/'labeled.pt',map_location='cpu',weights_only=False);di=diagnostic_indices(L);ids=[L['ids'][i] for i in di];y=np.asarray(L['y'][di].tolist(),np.int64);rx=np.asarray(L['rx'][di].tolist(),np.int64);day=np.asarray(L['day'][di].tolist(),np.int64);meta=d.read(bank/'actual_first_training_metadata.json');scene=np.asarray([v['scene'] for v in meta])
   if len(L['ids'])!=6300 or set(L['ids'])!=set(d.read(d.joint.SOURCE)['role_ids']['L_s']) or ids!=d.read(bank/'source_roles.json')['L_diagnostic_ids'] or ids!=[v['physical_id'] for v in meta]:raise ValueError('Exact validated legal L role and existing observations required')
   raw=L['x'][di];fresh=np.load(bank/'same_L_new_channel.npy',mmap_mode='r',allow_pickle=False);actual=np.load(bank/'actual_first_training_As.npy',mmap_mode='r',allow_pickle=False)
   if fresh.shape!=(2,2,1800,2,256) or actual.shape!=(1800,2,256):raise ValueError('Existing exact source observations differ')
   inputs=[('raw',raw),('actual_first',actual)]+[(f'{name}_draw{a}',fresh[a,b]) for a in range(2) for b,name in enumerate(('mid','low'))]
   for name,m in models.items():
    initial={k:v.clone() for k,v in m.state_dict().items()};views={};saved={}
    with Probe(m) as p:
     for view,x in inputs:
      logits={mode:[] for mode in d.MODES};changes={mode:[] for mode in d.MODES[1:]}
      for i in range(0,1800,180):
       z=x[i:i+180] if torch.is_tensor(x) else tensor_iq(x[i:i+180],'cpu');v=p.take(m.forward_identity_only(z));NN+=len(z)
       for mode in logits:logits[mode].extend(v['logits'][mode].tolist())
       for mode in changes:changes[mode].append(v['joint_L2_mean'][mode])
      arrays={mode:np.asarray(v,np.float32) for mode,v in logits.items()};views[view]=dict(count=1800,conditions={mode:score(v,y,rx,day,scene if view=='actual_first' else None) for mode,v in arrays.items()},joint_L2_mean={mode:float(np.mean(v)) for mode,v in changes.items()})
      for mode,v in arrays.items():saved[view+'_'+mode]=v
      d.write(d.BASE/'progress.json',dict(model=name,view=view,NN=NN,seconds=time.perf_counter()-start));print(json.dumps(dict(model=name,view=view,NN=NN)),flush=True)
    if not p.closed or p.calls!=60 or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()):raise ValueError('Fixed model state/capture restore/call budget differs')
    np.savez_compressed(d.BASE/(name+'_source_logits.npz'),ids=np.asarray(ids),y=y,rx=rx,day=day,training_scene=scene,**saved);rows[name]=dict(views=views,native_base_pool_joint_head_bit_equal=True,parameters_unchanged=True,displacement_max_error=p.max_displacement,time_area_max_error=p.max_area,capture_closed=True)
   if NN!=21604:raise ValueError('Fixed source complete NN budget differs')
   q=decision(rows);result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,rows=rows,decision=q,NN=NN,source_encoder_NN=21600,additional_alternative_head_decisions=64800,additional_native_replay_head_decisions=21600,additional_default_head_decisions=86400,total_source_head_decisions=108000,additional_pool_output_projections=86400,smoke_total_head_decisions=20,fit_updates=0,new_physical_simulations=0,U_used=False,V_used=False,target_access=False,formal_scores_read=False,model_seed_count=1,backend_flags=actual_flags(),wall_seconds=time.perf_counter()-start,goal_complete=False)
   d.write(d.BASE/'results.json',result);d.write(d.BASE/'complete.json',dict(status=result['status'],NN=NN,target_access=False,fit_updates=0,decision=q,goal_complete=False));print(json.dumps(dict(status=result['status'],NN=NN,decision=q)),flush=True)
 except Exception:
  d.write(d.BASE/'failure.json',dict(status='TECHNICAL_FAILURE',traceback=traceback.format_exc(),target_access=False));raise
if __name__=='__main__':main()
