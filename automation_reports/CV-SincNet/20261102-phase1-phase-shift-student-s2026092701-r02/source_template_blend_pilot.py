import pathlib,sys,torch,numpy as np,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;cache,clean,rx,train,pv,ph=s.source_contract(root);valid,held=s.new_splits(cache,rx,train,pv,ph);received,scene=s.epoch_data(cache,14);rng=np.random.default_rng(2026112355);bank_ids=rng.choice(train,size=8192,replace=False);views=['clean','practical_high','practical_mid_urban','practical_low_urban'];ids=np.concatenate([rng.choice(held[scene[held]==v],128,replace=False) for v in views]);assert not np.intersect1d(bank_ids,ids).size
model,gate=s.initial_models(root,'cpu');model.eval();f,_=core.teacher(root,'cpu');d=prior.frozen_dura(root,'cpu');alphas=[0.,.05,.1,.25,.5,1.];rows=[]
with torch.no_grad():
 chunks=[]
 for start in range(0,len(bank_ids),256):
  y=core.tensor(clean[bank_ids[start:start+256]],'cpu');chunks.append(torch.complex(y[:,0],y[:,1]))
 bank=torch.cat(chunks,0);bc=bank[:,64:-64];bn=bc.abs().square().sum(1).clamp_min(1e-12);unit=bc/bn.sqrt()[:,None]
 for start in range(0,len(ids),16):
  ix=ids[start:start+16];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');anchor=model.anchor(x);az=torch.complex(anchor[:,0],anchor[:,1]);ac=az[:,64:-64];an=ac.abs().square().sum(1).clamp_min(1e-12);score=(ac@unit.conj().T).abs()/an.sqrt()[:,None];best=score.argmax(1);template=bank[best];gain=(template[:,64:-64].conj()*ac).sum(1)/bn[best];templ=template*gain[:,None];p=f.classify_features(f.features(y)).softmax(1);target=p.argmax(1);valid=p.max(1).values>=.95;active=gate(x).sigmoid()>.8;baseline=d(x);raw=f.classify_features(f.features(x)).argmax(1);dp=f.classify_features(f.features(baseline)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,dr=s.align(baseline,y);_,rr=s.align(x,y);cols=[valid.float(),(raw==target).float(),(dp==target).float(),active.float(),rr/den,dr/den]
  for alpha in alphas:
   z=(1.-alpha)*az+alpha*templ;w=torch.stack((z.real,z.imag),1);w=torch.where(active[:,None,None],w,x);pred=f.classify_features(f.features(w)).argmax(1);_,err=s.align(w,y);cols.extend(((pred==target).float(),err/den))
  rows.append(torch.stack(cols,1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for view in views:
 a=v[scene[ids]==view];good=a[:,0]>.5;g=a[good];cells={}
 for i,alpha in enumerate(alphas):cells[str(alpha)]={'correct':int(g[:,6+2*i].sum()),'aligned_nmse':float(a[:,7+2*i].mean())}
 out[view]={'count':len(a),'valid':len(g),'active':int(a[:,3].sum()),'raw_correct':int(g[:,1].sum()),'DURA_correct':int(g[:,2].sum()),'raw_aligned_nmse':float(a[:,4].mean()),'DURA_aligned_nmse':float(a[:,5].mean()),'candidate':cells}
print(json.dumps({'scope':'source only; 8192 clean training-ID dictionary, 128 disjoint RX8/day3 source holdout records per scene','method':'input-only nearest clean IQ central correlation, complex gain fit to phase-shift anchor, fixed blend alphas, gate bypass on clean','disclaimer':'templates belong to different physical IDs and can inject other transmitters fingerprints; small exploratory diagnostic, not deployed or a full restoration; phase anchor previously source-trained with DURA','target':'high-confidence F(clean) pseudo class, not true labels','TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':out},allow_nan=False))
