import csv,json,os,sys,time,math
from pathlib import Path
import torch
from . import design as d
from .core import errors,jacobian
from experiments.cvs_e20_response import design as rd
from experiments.cvs_e20_response.source import load_source
from experiments.cvs_e20_response.response import Bank
from experiments.cvs_e20_response.model import build
from experiments.cvs_e20_response.evaluate import source_provenance
from experiments.cvs_e20_response.objective import eval_forward
from experiments.cvs_nocfo_engineering.teacher import load as load_parent
from experiments.cvs_finite_action.operators import apply,quality
from experiments.cvs_finite_action.state import snapshot
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def model_source(tag,seed,device):
 if tag=='E20':return load_parent(seed,device)
 c=rd.config(next(r for r in rd.rows() if r['model_seed']==seed and r['arm']==tag));source=source_provenance(c);resolved=rd.read(source/'resolved_config.json')
 if resolved['commit']!=d.E40_EXECUTION_COMMIT:raise ValueError('E40 source execution differs')
 model=build(c,device,source/'final_ssdg.pth').eval().requires_grad_(False)
 return model,dict(tag=tag,source=str(source),execution_commit=d.E40_EXECUTION_COMMIT,initialization=rd.read(source/'initialization.json'),source_config=c,target_training_or_selection=False)

def partitions(L,bank,c):
 s=torch.load(bank.proof['state_paths'][0],map_location='cpu',weights_only=False);index={pid:i for i,pid in enumerate(L['ids'])}
 fi=torch.tensor([index[p] for p in s['fit_IDs']]);ev=torch.tensor([index[p] for p in s['eval_IDs']])
 if len(fi)!=5040 or len(ev)!=1260 or set(fi.tolist())&set(ev.tolist()) or set(fi.tolist())|set(ev.tolist())!=set(range(6300)):raise ValueError('Original source PID split differs')
 return dict(L_s=dict(L_fit=fi,L_eval=ev),U_s=dict(U_all=torch.arange(56700)))

@torch.no_grad()
def extract(model,data,c,device,role):
 if role=='U_s' and any(k in data for k in ('y','label','labels','tx','TX','truth')):raise ValueError('U TX forbidden')
 h={k:[] for k in ('time_down','t2.pool')};delta={(kind,k):[] for kind,k,m in c['tasks']};qs=[];head={kind:[0,0.] for kind in ('clean','spectral','timing')};calls=packets=0
 for start in range(0,len(data['ids']),c['batch_size']):
  x=data['x'][start:start+c['batch_size']].to(device);logits,features=eval_forward(model,x,features=True);calls+=1;packets+=len(x);qs.append(quality(x).cpu())
  for level in h:h[level].append(features[level].cpu())
  def label_measure(kind,out):
   if role!='L_s':return
   y=data['y'][start:start+len(x)].to(device);head[kind][0]+=int((out.argmax(1)==y).sum());margin=out.gather(1,y[:,None]).squeeze(1)-out.masked_fill(torch.nn.functional.one_hot(y,6).bool(),float('-inf')).max(1).values;head[kind][1]+=float(margin.sum())
  label_measure('clean',logits)
  for kind in ('spectral','timing'):
   acted=apply(x,kind,c['known_parameters'][kind],c['model_seed']);out,other=eval_forward(model,acted,features=True);calls+=1;packets+=len(x);label_measure(kind,out)
   for level in h:delta[kind,level].append((other[level]-features[level]).cpu())
 return dict(h={k:torch.cat(v) for k,v in h.items()},delta={k:torch.cat(v) for k,v in delta.items()},q=torch.cat(qs),model_forward_calls=calls,source_packet_occurrences=packets,head={k:dict(accuracy=v[0]/len(data['ids']),margin=v[1]/len(data['ids'])) for k,v in head.items()} if role=='L_s' else None,U_truth=False)

def run(row,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY):
  L,U,V,contract=load_source(c['response_config']);del V
  bank=Bank(c['response_config'],L,U,device);groups=partitions(L,bank,c);fixed={key:snapshot(reader) for key,reader in bank.readers.items()}
  parameters={tuple(task[:2]):torch.load(path,map_location='cpu',weights_only=False)['parameter'].to(device) for task,path in zip(c['tasks'],bank.proof['state_paths'])}
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',backend_flags=actual_flags(),response_provenance=bank.proof,source_only=True,identity_updates=0,auxiliary_updates=0,target_access=False,U_truth=False,V_forwards=0)
  d.write(out/'resolved_config.json',resolved);print('RESOLVED '+json.dumps(resolved),flush=True)
  if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
  reference={};records=[];observations=[];sources=[];packets=calls=0
  with (out/'model_metrics.jsonl').open('x',encoding='utf-8') as logs:
   for mi,tag in enumerate(d.TAGS):
    model,proof=model_source(tag,c['model_seed'],device);before=snapshot(model);sources.append(dict(tag=tag,provenance=proof))
    for ri,(role,data) in enumerate([('L_s',L),('U_s',U)]):
     t=time.perf_counter();obs=extract(model,data,c,device,role);packets+=obs['source_packet_occurrences'];calls+=obs['model_forward_calls'];order=bank.orders[role];cached=bank.observed[role]
     if not torch.allclose(obs['q'],cached['q'][order],rtol=1e-5,atol=1e-6):raise ValueError('Actual source quality cache differs')
     if tag=='E20':
      reference[role]=obs
      for kind,level,method in c['tasks']:
       target=cached['targets'][kind]['mean'][level][order,cached['targets'][kind]['points'].index(1)]
       if not torch.allclose(obs['delta'][kind,level],target,rtol=1e-5,atol=1e-5):raise ValueError('Actual E20 response/cache differs')
     shuffled=torch.arange(len(data['ids']));gen=torch.Generator().manual_seed(c['shuffle_seed']+ri*100003)
     for group,idx in groups[role].items():shuffled[idx]=idx[torch.randperm(len(idx),generator=gen)]
     for ti,(kind,level,method) in enumerate(c['tasks']):
      reader=bank.readers[kind,level];a=torch.tensor(c['known_parameters'][kind],device=device);base=reference[role];metrics=errors(reader,parameters[kind,level],obs['h'][level],base['h'][level],obs['q'],a,obs['delta'][kind,level],base['delta'][kind,level],shuffled,device)
      for gi,(group,idx) in enumerate(groups[role].items()):
       generator=torch.Generator().manual_seed(c['jacobian_seed']+ri*100003+gi*101);sel=idx[torch.randperm(len(idx),generator=generator)[:c['jacobian_samples_per_stratum']]]
       j=jacobian(reader,obs['h'][level][sel].to(device),obs['q'][sel].to(device),a.expand(len(sel),-1),c['jacobian_seed']+ri*100003+gi*101+ti*1009,c['jacobian_probes'])
       records.append(dict(model_tag=tag,role=role,stratum=group,kind=kind,level=level,method=method,count=len(idx),metrics={k:float(v[idx].mean()) for k,v in metrics.items()},jacobian=j))
     measured=dict(model_tag=tag,role=role,count=len(data['ids']),forward_calls=obs['model_forward_calls'],source_packet_occurrences=obs['source_packet_occurrences'],seconds=time.perf_counter()-t,head=obs['head'],U_truth=False,target_access=False,identity_updates=0)
     observations.append(measured);logs.write(json.dumps(measured)+'\n');logs.flush();print('MODEL_ROLE '+json.dumps(measured),flush=True)
    after=snapshot(model)
    if any(not torch.equal(t,after[n]) for n,t in before.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen source identity changed')
    del model
  if packets!=c['source_model_packet_occurrences'] or len(records)!=96 or len(observations)!=16:raise ValueError('Incomplete fixed source diagnostic')
  after_readers={key:snapshot(reader) for key,reader in bank.readers.items()}
  if any(not torch.equal(t,after_readers[key][n]) for key,reader in bank.readers.items() for n,t in fixed[key].items()) or any(p.grad is not None for reader in bank.readers.values() for p in reader.parameters()):raise ValueError('Frozen readout changed')
  flat=[dict(**{k:v for k,v in r.items() if k not in ('metrics','jacobian')},**r['metrics']) for r in records]
  with (out/'metrics.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
  result=dict(config=c,records=records,observations=observations,source_provenance=sources,response_provenance=bank.proof,identity_updates=0,auxiliary_updates=0,U_fit=False,U_truth=False,V_forwards=0,target_access=False,parameters_buffers_unchanged=True,source_packet_occurrences=packets,model_forward_calls=calls,source_only=True)
  d.write(out/'results.json',result);done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,identity_updates=0,auxiliary_updates=0,target_access=False,source_packet_occurrences=packets,model_forward_calls=calls,records=96,model_count=8,parameters_buffers_unchanged=True,E20_actual_cache_check=True,seconds=time.perf_counter()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0)
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(done),flush=True)
 return done
