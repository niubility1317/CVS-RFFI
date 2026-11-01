import pathlib,sys,torch,numpy as np,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;cache,clean,rx,train,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,13);rng=np.random.default_rng(2026112055);views=[x for x in s.VIEWS if x!='clean'];ids=np.concatenate([rng.choice(train[scene[train]==v],128,replace=False) for v in views]);f,_=core.teacher(root,'cpu');d=prior.frozen_dura(root,'cpu');rows=[]
with torch.no_grad():
 for start in range(0,len(ids),32):
  idx=ids[start:start+32];x=core.tensor(received[idx],'cpu');y=core.tensor(clean[idx],'cpu');pad=F.pad(x,(64,64),mode='reflect');full=torch.complex(pad[:,0],pad[:,1]).unfold(-1,256,1);part=full[:,:,64:-64];target=torch.complex(y[:,0,64:-64],y[:,1,64:-64])[:,None,:];gain=(part.conj()*target).sum(-1)/part.abs().square().sum(-1).clamp_min(1e-20);err=(gain[:,:,None]*part-target).abs().square().mean(-1);k=err.argmin(1);sel=gain.gather(1,k[:,None]).squeeze(1);g=sel/sel.abs().clamp_min(1e-8);w=full[torch.arange(len(idx)),k]*g[:,None];wave=torch.stack((w.real,w.imag),1);p=f.classify_features(f.features(y)).softmax(1);valid=p.max(1).values>=.95;target=p.argmax(1);raw=f.classify_features(f.features(x)).argmax(1);adjusted=f.classify_features(f.features(wave)).argmax(1);dura=f.classify_features(f.features(d(x))).argmax(1);r=err.gather(1,k[:,None]).squeeze(1)/torch.complex(y[:,0,64:-64],y[:,1,64:-64]).abs().square().mean(1).clamp_min(1e-12);rows.append(torch.stack((valid.float(),(raw==target).float(),(adjusted==target).float(),(dura==target).float(),(k-64).float(),sel.abs(),r),1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for name in views:
 a=v[scene[ids]==name];z=a[a[:,0]>.5];out[name]={'sample_count':len(a),'valid':len(z),'raw_correct':int(z[:,1].sum()),'oracle_sync_correct':int(z[:,2].sum()),'DURA_correct':int(z[:,3].sum()),'delay_median':float(np.median(a[:,4])),'delay_p10_p90':np.percentile(a[:,4],[10,90]).tolist(),'gain_magnitude_median':float(np.median(a[:,5])),'aligned_nmse_mean':float(a[:,6].mean())}
print(json.dumps({'scope':'source train only; 128 records per impaired scene, no TX/U_s truth/V/test','target':'high-confidence F(clean) pseudo labels','oracle':'gain phase and shift fitted using paired clean IQ, unavailable at deployment; phase-only gain preserves amplitude','results':out},allow_nan=False))
