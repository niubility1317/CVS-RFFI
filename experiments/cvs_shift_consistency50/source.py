import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .model import build,groups
from .objective import objective
from experiments.cvs_nocfo_foundation.source import load_source as native_source
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load(c):
 d.validate(c);L,closed,V,contract=native_source(c['source_loader_config'])
 if closed is not None or len(L['ids'])!=6300 or len(V['ids'])!=27000:raise ValueError('Existing source roles differ')
 U=None
 if c['source_U']:
  U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('HiddenU physical role differs')
 return L,U,contract

def expected_counts(c):
 active=d.UPDATES-d.WARMUP;multi=c['source_shift']['enabled'];stats=c['shift_statistics']['enabled'];u=c['source_U']
 return dict(identity_updates=d.UPDATES,L_packets=d.UPDATES*128,U_packets=active*128 if u else 0,accepted_U=None,teacher_forwards=active*2 if u else 0,U_student_forwards=active if u else 0,L_model_forwards=d.UPDATES+active if multi else d.UPDATES,shift_updates=active if multi else 0,statistics_updates=active if stats else 0)

def validate_counts(c,counts,distinct_U):
 expected=expected_counts(c)
 if any(counts[k]!=v for k,v in expected.items() if v is not None) or not 0<=counts['accepted_U']<=counts['U_packets'] or distinct_U!=(56700 if c['source_U'] else 0):raise ValueError('E50 fixed update/view/role budget differs')
 return True

def train(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();L,U,contract=load(c);seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device);pg,names=groups(model,c);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=c['weight_decay'])
  ls=DomainStream(L['rx'],L['day'],c['model_seed']+101);us=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U else None;bank=VisitPassBank(2,.95) if U else None
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy,checkpoint_loaded=False,ancestors=[],trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,U_iterator=U is not None,teacher='owncurrentstudent twoevalweakviews' if U else 'none',EMA=False,satellite_generator=False,L_count=6300,U_count=56700 if U else 0,V_forward=False,source_performance_gate=False,target_access=False,U_truth_read=False,actual_statistic_layers=list(d.LEVELS) if c['shift_statistics']['enabled'] else [],source_augment_non_circular=True,teacher_branch_after_update=d.WARMUP if U else None)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True)
  counts={k:0 for k in expected_counts(c)};visited=set();epochs=[];seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,d.EPOCHS+1):
    model.train();sums={};lr=c['min_lr']+(c['peak_lr']-c['min_lr'])*(1+math.cos(math.pi*(epoch-1)/(d.EPOCHS-1)))/2
    for g in pg:g['lr']=lr
    for local in range(d.STEPS):
     step=(epoch-1)*d.STEPS+local+1;li,_=ls.batch(128);ux=ids=passes=None
     if U is not None and step>d.WARMUP:
      ui,passes=us.batch(128);visited.update(ui.tolist());ux=U['x'][ui].to(device);ids=[U['ids'][i] for i in ui.tolist()]
     opt.zero_grad(set_to_none=True);loss,metrics,detail=objective(model,L['x'][li].to(device),L['y'][li].to(device),ux,ids,passes,bank,c,step)
     if not torch.isfinite(loss):raise FloatingPointError('E50 source objective')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128
     counts['L_model_forwards']+=metrics['L_model_forwards'];counts['teacher_forwards']+=metrics['teacher_forward_calls'];counts['shift_updates']+=metrics['shift_active'];counts['statistics_updates']+=metrics['statistics_active']
     for k in ('U_packets','accepted_U','U_student_forwards'):counts[k]+=metrics[k]
     metrics['gradient_norm']=float(norm)
     for k,v in metrics.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=metrics,detail=detail))+'\n')
    log.flush();record=dict(epoch=epoch,updates=step,lr=lr,**{k:v/d.STEPS for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited),validation=None);epochs.append(record);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in epochs),encoding='utf-8');print('EPOCH '+json.dumps(record),flush=True)
  validate_counts(c,counts,len(visited))
  torch.save(dict(model=model.state_dict(),config=c,epoch=d.EPOCHS,updates=d.UPDATES,ancestors=[],initialization='scratch',target_access=False,U_truth_read=False,checkpoint_selection='fixed_final_epoch50'),out/'identity.pth')
  done=dict(status='SOURCE_TRAINED',config=c,epoch=d.EPOCHS,updates=d.UPDATES,logged_steps=d.UPDATES,counts=counts,distinct_U=len(visited),target_access=False,U_truth_read=False,V_forward=False,ancestors=[],elapsed_seconds=time.perf_counter()-start,parameters=resolved['trainable_parameters'])
  d.write(out/'completion.json',done);d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(out/'identity.pth'),epoch=d.EPOCHS,updates=d.UPDATES,selection='fixed_final_epoch50',target_access=False));print('SOURCE_TRAINED '+rowid,flush=True)
 return done
