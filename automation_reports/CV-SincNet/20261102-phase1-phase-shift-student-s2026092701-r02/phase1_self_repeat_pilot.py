import pathlib,sys,torch,numpy as np,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);cache,clean,rx,train,pv,ph=s.source_contract(root);_,held=s.new_splits(cache,rx,train,pv,ph);received,scene=s.epoch_data(cache,15);rng=np.random.default_rng(2026113055);views=['clean','practical_high','practical_mid_urban','practical_low_urban'];ids=np.concatenate([rng.choice(held[scene[held]==v],128,replace=False) for v in views]);phase,gate=s.initial_models(root,'cpu');phase.eval().requires_grad_(False);f,_=core.teacher(root,'cpu');d=prior.frozen_dura(root,'cpu');alphas=(.1,.25,.5);thresholds=(.8,.95);rows=[]
with torch.no_grad():
 for start in range(0,len(ids),16):
  ix=ids[start:start+16];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');a=phase.anchor(x);z=torch.complex(a[:,0],a[:,1]);u=z[:,:64];v=z[:,80:144];dot=(u*v.conj()).sum(1);rot=dot/dot.abs().clamp_min(1e-12);corr=dot.abs()/(u.abs().square().sum(1)*v.abs().square().sum(1)).clamp_min(1e-20).sqrt();gate_on=gate(x).sigmoid()>.8;cleanp=f.classify_features(f.features(y)).softmax(1);target=cleanp.argmax(1);valid=cleanp.max(1).values>=.95;dura=d(x);raw=f.classify_features(f.features(x)).argmax(1);base=f.classify_features(f.features(dura)).argmax(1);anch=torch.where(gate_on[:,None,None],a,x);pa=f.classify_features(f.features(anch)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(dura,y);_,ae=s.align(anch,y);cols=[valid.float(),(raw==target).float(),(base==target).float(),(pa==target).float(),de/den,ae/den,gate_on.float()]
  for th in thresholds:
   act=gate_on&(corr>=th)
   for alpha in alphas:
    w=z.clone();w[:,:64]=(1-alpha)*u+alpha*rot[:,None]*v;w[:,80:144]=(1-alpha)*v+alpha*rot.conj()[:,None]*u;wi=torch.stack((w.real,w.imag),1);wi=torch.where(act[:,None,None],wi,anch);pred=f.classify_features(f.features(wi)).argmax(1);_,e=s.align(wi,y);cols.extend(((pred==target).float(),e/den,act.float()))
  rows.append(torch.stack(cols,1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for view in views:
 a=v[scene[ids]==view];g=a[a[:,0]>.5];candidates={}
 for ti,th in enumerate(thresholds):
  for ai,alpha in enumerate(alphas):
   j=7+3*(ti*len(alphas)+ai);candidates[f'{th}:{alpha}']={'correct':int(g[:,j].sum()),'aligned_nmse':float(a[:,j+1].mean()),'modified':int(a[:,j+2].sum())}
 out[view]={'count':len(a),'valid':len(g),'gate_active':int(a[:,6].sum()),'raw_correct':int(g[:,1].sum()),'DURA_correct':int(g[:,2].sum()),'anchor_correct':int(g[:,3].sum()),'DURA_aligned_nmse':float(a[:,4].mean()),'anchor_aligned_nmse':float(a[:,5].mean()),'candidate':candidates}
print(json.dumps({'source_only':True,'source_rx8_day3_records_per_scene':128,'input_only':'complex phase and correlation from first two 64-sample windows 80 samples apart; protected gate bypass','target':'F(clean) confidence >=.95 pseudo-label, not true accuracy','phase_prior':'previously trained with DURA source teacher; DURA not used in inference','TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'limitations':'small source diagnostic; receiver/day identities previously examined; no trained model, checkpoint or Phase1 test','results':out},allow_nan=False))
