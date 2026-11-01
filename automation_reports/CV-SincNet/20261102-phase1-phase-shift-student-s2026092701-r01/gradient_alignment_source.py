import pathlib,sys,numpy as np,torch,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r01-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);torch.manual_seed(2026111854);cache,clean,rx,train,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,13);rng=np.random.default_rng(2026111855);views=['practical_high','practical_mid_urban','practical_low_urban'];ids=np.concatenate([rng.choice(train[scene[train]==v],256,replace=False) for v in views]);m,g=s.initial_models(root,'cpu');d=prior.frozen_dura(root,'cpu');f,_=core.teacher(root,'cpu');groups={'teacher_agrees':[],'teacher_disagrees':[]};counts={v:{'valid':0,'agree':0,'disagree':0} for v in views}
with torch.no_grad():
 for begin in range(0,len(ids),16):
  ix=ids[begin:begin+16];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');q=f.classify_features(f.features(y)).softmax(1);target=q.argmax(1);valid=q.max(1).values>=.95;dp=f.classify_features(f.features(d(x))).argmax(1)
  for i in range(len(ix)):
   if bool(valid[i]):
    disagree=bool(dp[i]!=target[i]);key='teacher_disagrees' if disagree else 'teacher_agrees';groups[key].append(int(ix[i]));counts[scene[ix[i]]]['valid']+=1;counts[scene[ix[i]]]['disagree' if disagree else 'agree']+=1
results={};params=[m.body.temporal_output.weight,m.body.temporal_output.bias,m.body.output.weight,m.body.output.bias,m.body.phase_head[2].weight,m.body.phase_head[2].bias]
def flatten(loss,retain=True):
 gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True);return torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(params,gs)])
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))
for key in groups:
 selected=np.asarray(groups[key][:32],dtype=np.int64)
 if len(selected)<4:raise ValueError('insufficient confidence-qualified source cases')
 x=core.tensor(received[selected],'cpu');y=core.tensor(clean[selected],'cpu')
 with torch.no_grad():
  ref=d(x);aligned,_=s.align(y,ref);dl=f.classify_features(f.features(ref));yl=f.classify_features(f.features(y));ds=F.softmax(dl/2,1);ys=F.softmax(yl/2,1)
 out=m(x);logits=f.classify_features(f.features(out));norm=ref.square().mean((1,2)).clamp_min(1e-8);imit=((out-ref).square().mean((1,2))/norm).mean();kd=F.kl_div(F.log_softmax(logits/2,1),ds,reduction='none').sum(1).mean()*4;ky=F.kl_div(F.log_softmax(logits/2,1),ys,reduction='none').sum(1).mean()*4;inv=s.complex_gain_invariant_error(out,aligned).mean();gw=flatten(imit);gd=flatten(kd);gy=flatten(ky);gb=flatten(imit+.5*kd+.2*inv,False)
 results[key]={'records':len(selected),'teacher_pool':len(groups[key]),'waveform_loss':float(imit.detach()),'teacher_KL':float(kd.detach()),'clean_KL':float(ky.detach()),'waveform_vs_clean_gradient_cos':cos(gw,gy),'teacher_KL_vs_clean_gradient_cos':cos(gd,gy),'teacher_objective_vs_clean_gradient_cos':cos(gb,gy),'grad_norms':[float(gw.norm()),float(gd.norm()),float(gy.norm()),float(gb.norm())]}
print(json.dumps({'source_train_only':True,'records_examined':len(ids),'counts':counts,'classification_target':'clean F pseudo target conf >=.95; disagreement is not known error','parameters':'six output head weights and biases at random body initialization','no_TX':True,'no_U_truth':True,'no_V':True,'no_test':True,'results':results},allow_nan=False))
