import pathlib,sys,torch,numpy as np,json,time
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for p in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release']:
 sys.path.insert(0,str(root/'releases'/p))
sys.path.insert(0,str(root/'releases/20261102-phase1-phase-shift-student-s2026092701-r01-release/experiments/phase1_phase_shift_restorer'))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);torch.manual_seed(2026111654);cache,clean,rx,train,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,11);ids=np.random.default_rng(2026111655).choice(train[scene[train]!='clean'],size=2304,replace=False);training=ids[:2048];hold=ids[2048:];model,gate=s.initial_models(root,'cpu');dura=prior.frozen_dura(root,'cpu');teacher,_=core.teacher(root,'cpu');opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.0001,weight_decay=.001)
def evaluate():
 model.eval();rows=[]
 with torch.no_grad():
  for begin in range(0,len(hold),16):
   idx=hold[begin:begin+16];x=core.tensor(received[idx],'cpu');y=core.tensor(clean[idx],'cpu');d=dura(x);out=model(x);a=model.anchor(x);target=teacher.classify_features(teacher.features(y)).softmax(1);valid=target.max(1).values>=.95;label=target.argmax(1);pd=teacher.classify_features(teacher.features(d)).argmax(1);pa=teacher.classify_features(teacher.features(a)).argmax(1);pm=teacher.classify_features(teacher.features(out)).argmax(1);den=d[:,:,64:-64].square().mean((1,2)).clamp_min(1e-8);err=(out[:,:,64:-64]-d[:,:,64:-64]).square().mean((1,2))/den;err_a=(a[:,:,64:-64]-d[:,:,64:-64]).square().mean((1,2))/den;rows.append(torch.stack((valid.float(),(pd==label).float(),(pa==label).float(),(pm==label).float(),err,err_a),1))
 v=np.from_dlpack(torch.cat(rows).contiguous());good=v[:,0]>.5;return {'count':len(v),'pseudo_valid':int(good.sum()),'dura_pseudo_correct':int(v[good,1].sum()),'anchor_pseudo_correct':int(v[good,2].sum()),'student_pseudo_correct':int(v[good,3].sum()),'central_teacher_nmse':float(v[:,4].mean()),'anchor_teacher_nmse':float(v[:,5].mean())}
def forensic_objective(model,dura,teacher,x,y):
 from torch.nn import functional as F
 with torch.no_grad():
  ref=dura(x);aligned,_=s.align(y,ref);dl=teacher.classify_features(teacher.features(ref));yl=teacher.classify_features(teacher.features(y));ds=F.softmax(dl/2,1);ys=F.softmax(yl/2,1)
 out=model(x);logits=teacher.classify_features(teacher.features(out));norm=ref.square().mean((1,2)).clamp_min(1e-8);imit=(out-ref).square().mean((1,2))/norm;dk=F.kl_div(F.log_softmax(logits/2,1),ds,reduction='none').sum(1)*4;yk=F.kl_div(F.log_softmax(logits/2,1),ys,reduction='none').sum(1)*4;inv=s.complex_gain_invariant_error(out,aligned)
 value=(imit+.5*dk+.2*yk+.2*inv).mean();parts=(imit.mean(),(.5*dk).mean(),(.2*yk).mean(),(.2*inv).mean());return value,parts,out
before=evaluate();model.train();losses=[];active_counts=[]
for i in range(64):
 idx=training[i*32:(i+1)*32];x=core.tensor(received[idx],'cpu');y=core.tensor(clean[idx],'cpu')
 with torch.no_grad():active=gate(x).sigmoid()>.8
 if not active.any():raise ValueError('active required')
 loss,terms,signal=forensic_objective(model,dura,teacher,x[active],y[active]);opt.zero_grad(set_to_none=True);loss.backward(retain_graph=True)
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];rows=[(float(p.grad.detach().double().abs().max()),name,int((~torch.isfinite(p.grad)).sum())) for name,p in named if p.grad is not None];maxval=max(v[0] for v in rows);print('STEP',i,'loss',float(loss.detach()),'maxgrad',maxval,flush=True)
 if maxval>100 or any(v[2] for v in rows):
  print('TOTAL_PARAMS',sorted(rows,reverse=True)[:6],flush=True)
  for tag,term in [('total',loss)]+list(zip(('waveform','teacher_KL','clean_KL','invariant'),terms)):
   gs=torch.autograd.grad(term,[signal]+[p for _,p in named],retain_graph=True,allow_unused=True);outg=gs[0];top=[]
   for (name,p),q in zip(named,gs[1:]):
    if q is not None:top.append((float(q.detach().double().abs().max()),name,int((~torch.isfinite(q)).sum())))
   print('SAME_GRAPH_TERM',tag,'value',float(term.detach()),'output_grad_max',float(outg.detach().double().abs().max()),'output_nonfinite',int((~torch.isfinite(outg)).sum()),'param_top',sorted(top,reverse=True)[:4],flush=True)
  break
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()));active_counts.append(int(active.sum()))
after=evaluate();print(json.dumps({'source_train_only':True,'train':2048,'heldout_from_train':256,'steps':64,'before':before,'after':after,'losses':losses,'active':active_counts,'TX_labels_read':0,'U_s_truth_read':0,'V_reads':0,'test_reads':0,'classification':'clean F high-confidence pseudo targets, not ground truth','scope':'tiny diagnostic; no checkpoint or Phase1 target predictions'},allow_nan=False))
