import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .model import build,groups
from .method import gradients,parameter_selection
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_cross_rx_feedback.sampling import RotatingStreams
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def load(c):
 d.validate(c);L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None or len(L['ids'])!=6300 or len(V['ids'])!=27000:raise ValueError('Fullsource role contract')
 U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
 if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('HiddenU physical roles')
 return L,U,contract
def train(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();L,U,contract=load(c);seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  student,teacher=build(c,device);sg,snames=groups(student,c);tg,tnames=groups(teacher,c);sparams=[p for g in sg for p in g['params']];tparams=[p for g in tg for p in g['params']];sopt=torch.optim.AdamW(sg,weight_decay=.0001);topt=torch.optim.AdamW(tg,weight_decay=.0001);spnames,sp=parameter_selection(student,sg);tpnames,tp=parameter_selection(teacher,tg);ls=RotatingStreams(L,c['model_seed']+101,True);us=RotatingStreams(U,c['model_seed']+901337);bank=VisitPassBank(2,.95);rxs=sorted(ls.indices);visited=set();epochs=[];counts=dict(identity_updates=0,teacher_updates=0,L_packets=0,teacher_L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0,student_virtual_forwards=0,teacher_aux_gradients=0)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=student._identity_input_policy,teacher_identity_input_policy=teacher._identity_input_policy,checkpoint_loaded=False,ancestors=[],trainable_parameters=dict(STUDENT=sum(p.numel() for p in sparams),TEACHER=sum(p.numel() for p in tparams)),parameter_groups=dict(STUDENT=snames,TEACHER=tnames),feedback_parameters=dict(STUDENT=spnames,TEACHER=tpnames),actual_init_seeds=dict(STUDENT=c['model_seed'],TEACHER=c['feedback_teacher']['teacher_init_seed']),U_opened=True,U_truth_read=False,V_forward=False,source_performance_gate=False,target_access=False,classifier=c['classifier'],EMA=False,models_per_row=2)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    student.train();teacher.train();sums={};lr=.000001+(.0002-.000001)*(1+math.cos(math.pi*(epoch-1)/29))/2
    for g in sg+tg:g['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;mr=rxs[(step-1)%5];parts=[ls.batch(r,n)[0] for r,n in zip(rxs,(26,26,26,25,25))];li=torch.cat(parts);mi=parts[rxs.index(mr)];ux=ids=passes=None
     if step>1110:
      batches=[us.batch(r,32) for r in rxs if r!=mr];ui=torch.cat([a for a,b in batches]);passes=torch.cat([b for a,b in batches]);visited.update(ui.tolist());ux=U['x'][ui].to(device);ids=[U['ids'][i] for i in ui.tolist()]
     sopt.zero_grad(set_to_none=True);topt.zero_grad(set_to_none=True);metrics,detail=gradients(student,teacher,L['x'][li].to(device),L['y'][li].to(device),L['x'][mi].to(device),L['y'][mi].to(device),ux,ids,passes,bank,sopt,sp,tp,c,step);snorm=torch.nn.utils.clip_grad_norm_(sparams,10.,error_if_nonfinite=True);tnorm=torch.nn.utils.clip_grad_norm_(tparams,10.,error_if_nonfinite=True);sopt.step();topt.step();counts['identity_updates']+=1;counts['teacher_updates']+=1;counts['L_packets']+=128;counts['teacher_L_packets']+=128;counts['U_packets']+=metrics['U_packets'];counts['accepted_U']+=metrics['accepted_U'];counts['teacher_forwards']+=1+int(ux is not None);counts['student_virtual_forwards']+=2*int(ux is not None);counts['teacher_aux_gradients']+=int(ux is not None)
     values=dict(**metrics,CE=metrics['student_CE'],L_accuracy=metrics['student_train_accuracy'],student_gradient_norm=float(snorm),teacher_gradient_norm=float(tnorm),**{k:v for k,v in detail.items() if isinstance(v,(float,int))})
     if any(not math.isfinite(v) for v in values.values()):raise FloatingPointError('Teacherfeedback actualstep metrics')
     for k,v in values.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,feedback_RX=mr,detail={k:v for k,v in detail.items() if not isinstance(v,(float,int))}))+'\n')
    log.flush();r=dict(epoch=epoch,updates=step,teacher_updates=step,lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited));epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if counts['identity_updates']!=6660 or counts['teacher_updates']!=6660 or counts['U_packets']!=710400 or counts['L_packets']!=852480 or counts['teacher_L_packets']!=852480 or len(visited)!=56700:raise ValueError('Eachmodel fixed30budget')
  for name,model in (('STUDENT',student),('TEACHER',teacher)):
   torch.save(dict(model=model.state_dict(),config=c,readout=name,epoch=30,updates=6660,actual_init_seed=resolved['actual_init_seeds'][name],ancestors=[],fit_ids=L['ids'],fit_U_ids=U['ids'],target_access=False,U_truth_read=False,initialization='scratch'),out/(name+'.pth'))
  done=dict(status='SOURCE_TRAINED',config=c,epoch=30,updates=6660,teacher_updates=6660,combined_updates=13320,logged_steps=6660,epochs=epochs,counts=counts,distinct_U=len(visited),ancestors=[],target_access=False,U_truth_read=False,V_forward=False,source_performance_gate=False,readouts=['STUDENT','TEACHER'],elapsed_seconds=time.perf_counter()-start);d.write(out/'completion.json',done);d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,readouts=['STUDENT','TEACHER'],epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False));print('SOURCE_TRAINED '+rowid,flush=True)
 return done
