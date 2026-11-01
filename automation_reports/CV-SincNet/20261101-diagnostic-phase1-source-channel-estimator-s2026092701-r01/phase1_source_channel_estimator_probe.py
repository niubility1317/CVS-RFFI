"""Source-only supervised simulator-channel identifiability pilot."""
import subprocess
from pathlib import Path

inner=r'''
from pathlib import Path
import json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
from leo_practical.equalization import linear_filter
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
rng=np.random.default_rng(2026110351);groups=[]
for role,take in (('L_s',512),('U_s',1536)):
 available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==2048
source_rx=rx[fixed];source_day=day[fixed];split=np.full(len(fixed),-1,dtype=np.int64)
split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0
split[np.isin(source_rx,[4,6])&(source_day==3)]=1
split[(source_rx==8)&(source_day==3)]=2
assert all(int((split==i).sum())>100 for i in range(3))
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');namespace='phase1-source-channel-estimator-2026110352';data={}
with torch.no_grad():
 for view in views:
  received=[];kernels=[];freqs=[]
  if view!='clean':cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
  for j in fixed:
   v=np.asarray(clean[j],dtype=np.float64);signal,clean_rms=normalize_input(v[0]+1j*v[1])
   if view=='clean':
    result=(v[0]+1j*v[1]).astype(np.complex64);h=np.zeros(62,dtype=np.complex128);h[0]=1.;freq=0.
   else:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110353,session)
    result,meta,trace=ResidualChannel(cfg,stable_seed(2026110352,namespace,view,pid),hw).process(signal,return_traces=True)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied'];h=np.asarray(trace['residual_kernel'],dtype=np.complex128);assert len(h)==62
    t=np.arange(len(signal))/cfg.fs_hz;rotation=np.exp(-2j*np.pi*meta['output_frequency_hz']*t)
    ideal,_=linear_filter(signal,h,cfg.boundary);noiseless=np.asarray(trace['pre_agc_signal'])*rotation
    gamma=np.vdot(ideal,noiseless)/max(float(np.vdot(ideal,ideal).real),1e-20)
    h=h*gamma*(10**(meta['agc_gain_db']/20))/clean_rms;freq=float(meta['output_frequency_hz'])
   rms=max(float(np.sqrt(np.mean(abs(result)**2))),1e-12);h=h/rms;hnorm=max(float(np.sqrt(np.sum(abs(h)**2))),1e-12)
   received.append(np.stack((result.real,result.imag)).astype(np.float32));kernels.append(np.concatenate((h.real,h.imag)).astype(np.float32)/hnorm);freqs.append((np.log(hnorm),freq/10000.))
  received=np.stack(received);anchors=[]
  for start in range(0,len(fixed),32):anchors.append(dura(core.tensor(received[start:start+32],'cpu')).cpu())
  anchors=torch.cat(anchors);x=core.tensor(received,'cpu');rms=x.square().sum(1).mean(1).sqrt().clamp_min(1e-8)
  # Model sees only normalized received IQ, DURA IQ and received amplitude.
  data[view]=dict(x=torch.cat((x/rms[:,None,None],anchors/rms[:,None,None]),1),log_rms=rms.log()[:,None],kernel=core.tensor(np.stack(kernels),'cpu'),aux=core.tensor(np.asarray(freqs,dtype=np.float32),'cpu'))
  print('CHANNEL_ESTIMATOR_SOURCE_PAIRS_READY',view,flush=True)
print('SOURCE_SIMULATOR_TARGETS_ONLY_NO_TX_OR_TEST_LABELS',flush=True)
n=len(fixed);x=torch.cat([data[v]['x'] for v in views]);amp=torch.cat([data[v]['log_rms'] for v in views]);kernel=torch.cat([data[v]['kernel'] for v in views]);aux=torch.cat([data[v]['aux'] for v in views]);all_split=np.tile(split,len(views));train_ix=np.flatnonzero(all_split==0)
class ChannelNet(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(4,32,7,stride=2,padding=3),nn.GELU(),nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU(),nn.Conv1d(64,64,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Linear(65,128),nn.GELU(),nn.Linear(128,126))
 def forward(self,x,amp):return self.head(torch.cat((self.net(x).squeeze(-1),amp),1))
torch.manual_seed(2026110354);net=ChannelNet();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001);rng=np.random.default_rng(2026110355);validation=[];best=None;state=None
def evaluate(part,group=None):
 mask=split==part
 if group is not None:mask &= source_rx==group
 selected=np.flatnonzero(mask);out={}
 with torch.no_grad():
  for v in views:
   pred=[]
   for start in range(0,len(selected),128):
    ix=torch.tensor(selected[start:start+128].tolist(),dtype=torch.long);pred.append(net(data[v]['x'][ix],data[v]['log_rms'][ix]).cpu())
   pred=torch.cat(pred);truth=data[v]['kernel'][selected.tolist()];true_aux=data[v]['aux'][selected.tolist()]
   nmse=(pred[:,:124]-truth).square().sum(1);baseline=truth.square().sum(1);logamp=(pred[:,124]-true_aux[:,0]).abs();freq=(pred[:,125]-true_aux[:,1]).abs()*10000.
   out[v]=dict(count=len(selected),kernel_nmse=float(nmse.mean()),zero_baseline_nmse=float(baseline.mean()),kernel_nmse_median=float(nmse.median()),log_amplitude_mae=float(logamp.mean()),frequency_mae_hz=float(freq.mean()))
 return out
for epoch in range(1,16):
 order=rng.permutation(train_ix)
 for start in range(0,len(order),64):
  ix=torch.tensor(order[start:start+64].tolist(),dtype=torch.long);pred=net(x[ix],amp[ix]);loss=(pred[:,:124]-kernel[ix]).square().sum(1).mean()+.05*F.mse_loss(pred[:,124],aux[ix,0])+.01*F.mse_loss(pred[:,125],aux[ix,1])
  if not torch.isfinite(loss):raise FloatingPointError('Finite channel estimator loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
 if epoch in (2,5,10,15):
  val=evaluate(1);groups_v={str(group):evaluate(1,group) for group in (4,6)};score=sum(val[v]['kernel_nmse'] for v in views[1:])/3
  validation.append(dict(epoch=epoch,mean_impaired_nmse=score,views=val,groups=groups_v))
  if best is None or score<best['validation_nmse']:
   best=dict(epoch=epoch,validation_nmse=score);state={k:v.detach().clone() for k,v in net.state_dict().items()}
  print('CHANNEL_ESTIMATOR_SOURCE_VALIDATION_COMPLETE',epoch,flush=True)
net.load_state_dict(state);heldout=evaluate(2)
print('SOURCE_CHANNEL_ESTIMATOR_PROBE '+json.dumps(dict(status='SOURCE_ONLY_CHANNEL_ESTIMATOR_PROBE',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),excluded_ids=int((split<0).sum()),train_receivers=[1,3,4],validation_receivers=[4,6],heldout_receivers=[8],train_days=[1,2],heldout_days=[3],validation=validation,selected=best,heldout=heldout,model_state={k:v.detach().tolist() for k,v in state.items()},U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='Simulator latent channel, AGC and source-clean/noiseless trace are privileged source training targets only; inference uses received IQ and frozen DURA. The heldout receiver/day is disjoint only for this estimator; frozen experts saw source records and IDs may overlap prior probes. Simulator is not validated real satellite data. Prior observed Phase1 test results motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER',repr(inner),1)
proc=subprocess.Popen(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_CHANNEL_ESTIMATOR_PROBE ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('Source result marker missing')
 dest=Path(__file__).parent/'phase1_source_channel_estimator_probe.json'
 with dest.open('x',encoding='utf8') as f:f.write(payload)
 print('SAVED_SOURCE_CHANNEL_ESTIMATOR_RESULT',dest,flush=True)
raise SystemExit(rc)
