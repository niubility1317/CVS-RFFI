"""Synthetic exact IQ/label/domain/meta and original RNG/U-path checks; no NN."""
import gzip,json,random,types
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Dataset
from .pairing import Order,scope,private_L_loader,identities
from .checks import fixture
ROOT=Path(__file__).resolve().parents[2]
def batches():
 output=[]
 for slot,row in enumerate(fixture()):
  x=torch.arange(128*2*256,dtype=torch.float32).reshape(128,2,256)/1000+slot
  y=torch.tensor([r[0] for r in row]);domain=torch.tensor([(1,3,4,6,8).index(r[2])*3+r[3]-1 for r in row]);keys=('eq_i','rx_i','day_i','sig_i','base_index')
  meta={k:torch.tensor([r[i+1] for r in row]) for i,k in enumerate(keys)};meta['sample_id']=['synthetic-%d-%d'%(slot,i) for i in range(128)];meta['role']=['labeled']*128
  output.append((x,y,domain,meta))
 return output

def main():
 out=ROOT/'local_artifacts/F_L_batch_order_20261017/synthetic_pairing_r01';out.mkdir(parents=True,exist_ok=False);b=batches();torch_before=torch.get_rng_state().clone();random_before=random.getstate();outputs={}
 for arm in ('O0','O1'):
  target=out/arm;target.mkdir();c=Order(arm,target);q,old,new=c.transform(b);c.close();outputs[arm]=q
  if arm=='O0':
   for source,dest in zip(b,q):
    for a,z in zip(source[:3],dest[:3]):assert torch.equal(a,z)
    for key,a in source[3].items():assert torch.equal(a,dest[3][key]) if torch.is_tensor(a) else a==dest[3][key]
  flat={p[3]['sample_id'][i]:(p[0][i],int(p[1][i]),int(p[2][i]),{k:(int(v[i]) if torch.is_tensor(v) else v[i]) for k,v in p[3].items()}) for p in b for i in range(128)}
  for row in q:
   for i,key in enumerate(row[3]['sample_id']):
    sx,sy,sd,sm=flat[key];assert torch.equal(sx,row[0][i]) and sy==int(row[1][i]) and sd==int(row[2][i])
    assert all((int(v[i]) if torch.is_tensor(v) else v[i])==sm[k] for k,v in row[3].items())
 assert torch.equal(torch_before,torch.get_rng_state()) and random_before==random.getstate()
 target=out/'epoch';target.mkdir();c=Order('O1',target)
 def parent(*a,**kw):
  for i in range(222):yield b[i%6],None
 native=types.SimpleNamespace(_muse_epoch_pairs=parent)
 with scope(native,c):
  q=list(native._muse_epoch_pairs());assert len(q)==222 and all(u is None for _,u in q)
 assert native._muse_epoch_pairs is parent and c.steps==222 and c.blocks==37 and c.epochs==1
 with gzip.open(target/'L_batch_order_steps.jsonl.gz','rt',encoding='utf8') as f:assert len([json.loads(x) for x in f])==222
 class Visible(Dataset):
  hide=False
  def __len__(self):return 5040
  def __getitem__(self,i):return i
 loaders=[DataLoader(Visible(),batch_size=128,shuffle=True,num_workers=0,drop_last=True) for _ in range(2)]
 for ld in loaders:private_L_loader(ld,2026092701)
 snapshots=[list(iter(ld.sampler)) for ld in loaders];assert snapshots[0]==snapshots[1] and len(set(snapshots[0]))==5040 and torch.equal(torch_before,torch.get_rng_state())
 illegal=DataLoader(Visible(),batch_size=64,shuffle=True)
 try:private_L_loader(illegal,2026092701)
 except ValueError:pass
 else:raise AssertionError('Wrong batch accepted')
 print('PASS: control IQ bit-exact; reordered per-packet IQ/labels/domains/all metadata bit-exact; 222 steps/37 blocks; original function restored; no U batches; global Torch/Python RNG unchanged; private L sampling matched. Synthetic tensors only,NN=0,realIQ=0,CP=0,updates=0')
if __name__=='__main__':main()
