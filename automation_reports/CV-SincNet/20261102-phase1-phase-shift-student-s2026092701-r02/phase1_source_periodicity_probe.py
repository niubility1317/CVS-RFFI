import pathlib,sys,numpy as np,torch,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
torch.set_num_threads(2);cache,clean,rx,train,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,14);rng=np.random.default_rng(2026112855);views=[v for v in s.VIEWS if v!='clean'];ids=np.concatenate([rng.choice(train[scene[train]==v],128,replace=False) for v in views]);lags=[1,2,4,8,16,24,32,48,64,80,96,112];rows=[]
def cor(z,lag):
 a=z[:,:-lag];b=z[:,lag:];return (a.conj()*b).sum(1).abs()/(a.abs().square().sum(1)*b.abs().square().sum(1)).clamp_min(1e-20).sqrt()
with torch.no_grad():
 for start in range(0,len(ids),64):
  ix=ids[start:start+64];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');x=F.pad(x,(64,64),mode='reflect')[:,:,88:344];zx=torch.complex(x[:,0],x[:,1]);zy=torch.complex(y[:,0],y[:,1]);rows.append(torch.stack([*(cor(zy,k) for k in lags),*(cor(zx,k) for k in lags)],1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for name in views:
 a=v[scene[ids]==name];out[name]={'records':len(a),'clean_median':{str(k):float(np.median(a[:,i])) for i,k in enumerate(lags)},'received_median':{str(k):float(np.median(a[:,len(lags)+i])) for i,k in enumerate(lags)}}
print(json.dumps({'source_train_only':True,'records':len(ids),'metric':'magnitude normalized complex lag correlation over 256 samples','TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':out},allow_nan=False))
