import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from experiments.phase1_hankel_posterior.hankel import hankel_shrink

torch.set_num_threads(2)
torch.manual_seed(2026110321)
n=48; t=torch.arange(256, dtype=torch.float32)
frequencies=torch.randint(1,125,(n,4))
phases=2*math.pi*torch.rand(n,4)
angles=2*math.pi*frequencies[:,:,None]*t/256+phases[:,:,None]
tones=torch.polar(torch.ones_like(angles),angles).sum(1)
qpsk=torch.complex(torch.where(torch.rand(n,256)>.5,1.,-1.),torch.where(torch.rand(n,256)>.5,1.,-1.))
symbols=torch.complex(torch.where(torch.rand(n,64)>.5,1.,-1.),torch.where(torch.rand(n,64)>.5,1.,-1.))
shaped=symbols.repeat_interleave(4,dim=1)
waveforms={'four_tones':tones,'iid_qpsk':qpsk,'repeated_qpsk':shaped}
results={}
for name,z in waveforms.items():
 z=z/z.abs().square().mean(1,keepdim=True).sqrt()
 clean=torch.stack((z.real,z.imag),1)
 results[name]={}
 for snr in (0,5,10):
  variance=.5*10**(-snr/10)
  noise=torch.randn_like(clean)*math.sqrt(variance)
  observed=clean+noise
  restored=hankel_shrink(observed,torch.full((n,),variance))
  energy=clean.square().sum((1,2))
  raw=((observed-clean).square().sum((1,2))/energy)
  new=((restored-clean).square().sum((1,2))/energy)
  false_clean=hankel_shrink(clean,torch.full((n,),variance))
  distortion=(false_clean-clean).square().sum((1,2))/energy
  results[name][str(snr)]=dict(count=n,raw_nmse=float(raw.mean()),restored_nmse=float(new.mean()),relative_change_percent=float((new.mean()/raw.mean()-1)*100),improved=int((new<raw).sum()),clean_false_active_nmse=float(distortion.mean()))
print(json.dumps(results,indent=2))
