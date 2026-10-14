"""Real L720 continuous-signature geometry; no optimizer/classifier fitting."""
import json,time,traceback
from pathlib import Path
import torch
from . import design as d
from .algebra import components
from experiments.cvs_source_dropout.probe import objects,capture,select
from experiments.cvs_moment_area50.predict import source_proof,own_model
from experiments.cvs_equivariant_identity.precision import numerical_context
def statistics(c,rx,day):
 if len(rx)!=720 or len(day)!=720:raise ValueError('Fixed720 source context required')
 ij=torch.triu_indices(7,7,offset=1);parts={'full28':torch.arange(28),'increment6':torch.arange(6),'time_increment1':torch.tensor([6]),'feature_area15':7+torch.where(ij[1]!=6)[0],'time_area6':7+torch.where(ij[1]==6)[0]}
 variants={'missing_bridges':c['no_bridges'],'reset_clock':c['reset_clock_joined'],'omitted_cross_area':c['within_only']}
 result=[]
 for name,values in variants.items():
  for part,ix in parts.items():
   ref=c['direct'][:,ix].double();alternative=values[:,ix].double();error=alternative-ref
   if not torch.isfinite(ref).all() or not torch.isfinite(error).all():raise ValueError('Nonfinite source geometry')
   for dim,strata in [('overall',[None]),('receiver',d.RX),('day',d.DAYS)]:
    for value in strata:
     mask=torch.ones(720,dtype=torch.bool) if value is None else (rx if dim=='receiver' else day)==value;n=int(mask.sum());expected=720 if dim=='overall' else 144 if dim=='receiver' else 240
     if n!=expected:raise ValueError('Source stratum differs')
     r=ref[mask];e=error[mask];signal=float(r.square().mean().sqrt());noise=float(e.square().mean().sqrt())
     result.append(dict(variant=name,component=part,dimension=dim,stratum=value,count=n,width=len(ix),direct_RMS=signal,error_RMS=noise,error_to_direct_RMS=noise/signal if signal>1e-12 else None,error_L2_mean=float(e.norm(dim=1).mean()),maximum_absolute_error=float(e.abs().max())))
 return result
def smoke(model):
 saved={k:v.clone() for k,v in model.state_dict().items()};h=capture(model,torch.zeros(2,2,256));_,_,pool=objects(model)
 with torch.no_grad():u=torch.tanh(pool.views[0].local_projection(h)).transpose(1,2).double();c=components(u)
 if (c['joined']-c['direct']).abs().max()>1e-12 or any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()):raise ValueError('No-query geometry smoke failed')
 return dict(status='PASS',zero_IQ_packet_forwards=2,query_read=False,head_fit=False,optimizer_steps=0,continuous_composition_exact=True)
def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2);c=d.config(d.rows()[0]);d.validate(c)
 if d.BASE.exists():raise FileExistsError('Existing immutable geometry root; reconcile')
 d.BASE.mkdir(parents=True);out=Path(c['output_root']);out.mkdir();start=time.time()
 try:
  proof=source_proof(c['source_config'])
  if d.read(proof/'resolved_config.json')['commit']!=c['source_execution_commit']:raise ValueError('Source execution differs')
  model=own_model(c['source_config'],'cpu');objects(model);saved={k:v.clone() for k,v in model.state_dict().items()}
  L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;source=select(L);del L
  with numerical_context(d.FULL_FP32_POLICY),torch.no_grad():
   h=torch.cat([capture(model,source['x'][j:j+64]) for j in range(0,720,64)]);pool=model.id_backbone.encoder.sequence_pool;u=torch.tanh(pool.views[0].local_projection(h)).transpose(1,2);a=components(u.double());a32=components(u)
  err=float((a['joined']-a['direct']).abs().max());halferr=float((a['halves']-a['direct_halves']).abs().max());err32=float((a32['joined']-a32['direct']).abs().max())
  if err>1e-11 or halferr>1e-11 or err32>5e-5 or any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Composition mismatch or model mutation')
  raw=dict(projected_path=u,components=a,rx=source['rx'],day=source['day'],config=c);torch.save(raw,out/'source_geometry.pt');records=statistics(a,raw['rx'],raw['day']);reload=torch.load(out/'source_geometry.pt',map_location='cpu',weights_only=False)
  if reload['config']!=c or statistics(reload['components'],reload['rx'],reload['day'])!=records:raise ValueError('Saved geometry recount differs')
  result=dict(status='VERIFIED',run_id=d.RUN,row_id=d.ROW,model_seed=d.SEED,source_packets=720,packet_forwards=720,source_pool_packets=6300,records=records,joined_direct_error64=err,half_direct_error64=halferr,joined_direct_error32=err32,boundary_increments=3,global_time=True,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,head_fit=False,head_updates=0,optimizer_steps=0,training_epochs=0,model_state_unchanged=True,saved_geometry_recount='VERIFIED',source_execution_commit=c['source_execution_commit'],checkpoint=(proof/'final_ssdg.pth').as_posix(),seconds=time.time()-start,scope='Fixed source geometry only. No trained hierarchy, causal order/context attribution, target performance or newTX transfer claim. Current linear/normalized pooling may implicitly use other cross-window relations; explicit signature gaps do not prove fullmodel blindness.')
  d.write(out/'results.json',result);d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES='',CPU_threads=2))
  done=dict(status='ANALYZED',models=1,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,source_packets=720,source_geometry_recount='VERIFIED')
  d.write(out/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as e:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),outputs_preserved=True));raise
if __name__=='__main__':main()
