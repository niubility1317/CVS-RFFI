import pathlib,sys,numpy as np,torch,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;cache,clean,rx,training,pv,ph=s.source_contract(root);valid,held=s.new_splits(cache,rx,training,pv,ph);r,scene=s.epoch_data(cache,14);rng=np.random.default_rng(2026112255);memory=rng.choice(training,size=8192,replace=False);views=[v for v in s.VIEWS if v!='clean'];idx=np.concatenate([rng.choice(held[scene[held]==v],96,replace=False) for v in views]);assert len(np.intersect1d(memory,idx))==0
with torch.no_grad():
 buf=[]
 for start in range(0,len(memory),256):
  y=core.tensor(clean[memory[start:start+256]],'cpu')[:,:,64:-64];z=torch.complex(y[:,0],y[:,1]);buf.append(z/z.abs().square().sum(1,keepdim=True).clamp_min(1e-12).sqrt())
 bank=torch.cat(buf,0);rows=[]
 for start in range(0,len(idx),32):
  ii=idx[start:start+32];y=core.tensor(clean[ii],'cpu')[:,:,64:-64];x=core.tensor(r[ii],'cpu');x=F.pad(x,(64,64),mode='reflect')[:,:,88:344][:,:,64:-64];z=torch.complex(x[:,0],x[:,1]);truth=torch.complex(y[:,0],y[:,1]);z=z/z.abs().square().sum(1,keepdim=True).clamp_min(1e-12).sqrt();truth=truth/truth.abs().square().sum(1,keepdim=True).clamp_min(1e-12).sqrt();corr=z@bank.conj().T;dist=truth@bank.conj().T;near=corr.abs().argmax(1);by_y=dist.abs().max(1).values;retrieved=dist[torch.arange(len(ii)),near].abs();matching=(z.conj()*truth).sum(1).abs();noisy=corr.abs().max(1).values;rows.append(torch.stack((matching,by_y,retrieved,noisy),1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for name in views:
 a=v[scene[idx]==name];out[name]={'count':len(a),'matching_pair_coherence_median':float(np.median(a[:,0])),'clean_library_oracle_median':float(np.median(a[:,1])),'retrieved_to_paired_clean_median':float(np.median(a[:,2])),'noisy_to_retrieved_median':float(np.median(a[:,3]))}
print(json.dumps({'status':'SOURCE_IQ_LIBRARY_GEOMETRY','reference_source_clean_ids':len(memory),'heldout_source_ids':len(idx),'cache_epoch':14,'metric':'magnitude normalized complex inner product, central 128; observed shifted by +24','oracle':'clean-library maximum compares paired clean directly and is unavailable at inference; bounds only this fixed library','no_F_loaded':True,'TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'caveat':'source RX8 physical IDs previously seen under other channel views; no generated restoration or classification','results':out},allow_nan=False))
