import csv,json,os,sys,time,math,torch
from pathlib import Path
from . import design as d
from .observations import load,extract
from .splits import make
from .predictors import fit,predict
from .metrics import metric,coverage
from experiments.cvs_nocfo_engineering.teacher import load as load_reference
from .state import snapshot
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def flatten(obs,kind,level,indices,points):
 k=obs['targets'][kind];np=len(k['points']);h=obs['h'][level][indices].repeat_interleave(np,0);q=obs['q'][indices].repeat_interleave(np,0);a=torch.tensor([points[j] for j in k['points']],dtype=torch.float32).repeat(len(indices),1)
 return h,q,a,k['mean'][level][indices].flatten(0,1),k['variance'][level][indices].flatten(0,1)
def run(row_id):
 c=d.validate(d.config(next(r for r in d.rows() if r['row_id']==row_id)));out=Path(c['output']);out.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();L,U,contract=load(c);device='cuda:0'
 with numerical_context(d.FULL_FP32_POLICY):
  model,proof=load_reference(c['model_seed'],device)
  if proof['checkpoint']!=c['checkpoint'] or proof['execution_commit']!=c['teacher_commit']:raise ValueError('Fixed source encoder differs')
  before=snapshot(model);torch.cuda.reset_peak_memory_stats();d.write(out/'resolved_config.json',dict(config=c,checkpoint_provenance=proof,source_contract=contract,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],backend_flags=actual_flags(),identity_updates=0,target_access=False,U_fit=False))
  print('RESOLVED '+json.dumps(d.read(out/'resolved_config.json')),flush=True)
  def progress(value):print('FINITE_OBSERVATION '+json.dumps(value),flush=True);d.write(out/'observation_progress.json',value)
  observed_L=extract(model,L,c,device,'L_s',progress);observed_U=extract(model,U,c,device,'U_s',progress)
  torch.save(dict(L=observed_L,U=observed_U),out/'source_observations.pt');observations_seconds=time.perf_counter()-start
  split,split_proof=make(L,observed_L['q'],c);records=[];coverage_records=[];fits=[];updates=0
  with (out/'epoch_metrics.jsonl').open('x',encoding='utf-8') as logs,(out/'epoch_metrics.csv').open('x',encoding='utf-8',newline='') as csvstream:
   writer=csv.DictWriter(csvstream,fieldnames=['split','kind','level','epoch','loss','mean_error_term','repeat_variance_term','log_variance_term','gradient_norm_last','optimizer_steps','fit_observations','fit_packets','lr','epoch_seconds','identity_updates','U_fit']);writer.writeheader()
   for si,(name,(fit_ids,eval_ids)) in enumerate(split.items()):
    if name.startswith('TX') and int(name[2:]) in L['y'][fit_ids].tolist():raise ValueError('Held TX entered model fitting')
    for ki,kind in enumerate(d.KINDS):
     for li,level in enumerate(d.LEVELS):
      inputs=flatten(observed_L,kind,level,fit_ids,c['points'][kind]);test=flatten(observed_L,kind,level,eval_ids,c['points'][kind]);seed=c['fit_seed']+si*100003+ki*1009+li*101
      def epoch(v):
       value=dict(split=name,kind=kind,level=level,**v);logs.write(json.dumps(value)+'\n');logs.flush();writer.writerow(value);csvstream.flush();print('ACTION_EPOCH '+json.dumps(value),flush=True)
      fitted=fit(*inputs,c,kind,seed,device,epoch);updates+=fitted['updates']
      state=dict(state=fitted['state'],parameter=fitted['parameter'],bilinear=fitted['bilinear'],lowrank=fitted['lowrank'],net=fitted['net'].state_dict(),pooled_var=fitted['pooled_var'],noise=kind=='noise',split=name,kind=kind,level=level,config=c,fit_IDs=[L['ids'][i] for i in fit_ids.tolist()],eval_IDs=[L['ids'][i] for i in eval_ids.tolist()])
      torch.save(state,out/f'{name}-{kind}-{level.replace(".","_")}.pt')
      fits.append(dict(split=name,kind=kind,level=level,fit_packets=len(fit_ids),eval_packets=len(eval_ids),fit_observations=len(inputs[0]),epochs=c['fit_epochs'],optimizer_steps=fitted['updates'],trainable_aux_parameters=fitted['parameters'],fit_eval_PID_disjoint=True,U_fit=False,fit_normalization_only=True))
      for method in c['methods']:
       mu,var=predict(fitted,*test[:3],method,device);records.append(dict(split=name,kind=kind,level=level,method=method,role='L_s',**metric(mu,var,test[3],test[4],kind)))
       if name=='PID80_20':
        ut=flatten(observed_U,kind,level,torch.arange(len(U['ids'])),c['points'][kind]);um,uv=predict(fitted,*ut[:3],method,device);records.append(dict(split='U_coverage',kind=kind,level=level,method=method,role='U_s',TX_label_used=False,U_fit=False,**metric(um,uv,ut[3],ut[4],kind)))
        fm,fv=predict(fitted,*inputs[:3],method,device);coverage_records.append(dict(kind=kind,level=level,method=method,**coverage(fm,observed_L['h'][level],L,fit_ids,eval_ids,c['coverage_rank'])))
      d.write(out/'progress.json',dict(completed_fits=len(fits),expected_fits=117,auxiliary_optimizer_steps=updates,last_fit=fits[-1],identity_updates=0,target_access=False))
      del fitted
  after=snapshot(model)
  if before.keys()!=after.keys() or not all(torch.equal(before[k],after[k]) for k in before) or any(p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Reference encoder changed')
  if len(fits)!=117 or len(records)!=630 or len(coverage_records)!=45 or sum(v['counts']['packet_occurrences'] for v in (observed_L,observed_U))!=491400:raise ValueError('Incomplete fixed diagnostic matrix')
  results=dict(config=c,records=records,coverage=coverage_records,fits=fits,split_proof=split_proof,L_operations=observed_L['operation_metrics'],U_operations=observed_U['operation_metrics'],L_clean_head_accuracy=observed_L['clean_head_accuracy'],identity_updates=0,U_fit=False,U_truth=False,V_forwards=0,target_access=False)
  d.write(out/'results.json',results)
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,row_id=row_id,config=c,identity_updates=0,head_updates=0,target_access=False,U_fit=False,U_truth=False,auxiliary_fits=117,auxiliary_optimizer_steps=updates,records=630,coverage_records=45,budget=dict(L=observed_L['counts'],U=observed_U['counts'],packet_occurrences=491400,reference_parameters_buffers_modes_gradients_unchanged=True),checkpoint_provenance=proof,seconds=time.perf_counter()-start,observations_seconds=observations_seconds,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),source_observations_bytes=(out/'source_observations.pt').stat().st_size)
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(done),flush=True)
 return done
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);run(p.parse_args().row)
