from pathlib import Path
import json,tempfile,py_compile
from unittest.mock import patch
import numpy as np
import torch
import source_data as sd
import train
P=Path(__file__).parent
for p in P.glob('*.py'):py_compile.compile(str(p),doraise=True)
seen=[]
y=np.repeat(np.arange(10),100)
class Guard:
    def __init__(self,allowed):self.allowed=set(allowed)
    def __getitem__(self,idx):
        idx=np.asarray(idx);assert set(idx.tolist())<=self.allowed,'unlabeled IQ accessed'
        seen.extend(idx.tolist());return np.zeros((len(idx),2,2048),np.float32)
def guarded_load(path,**kwargs):
    rx=int(Path(path).name.split('_')[1])
    if str(path).endswith('_y.npy'):return y.copy()
    l,v,u=sd._split_domain_indices(y,.1,.1,299+rx,10)
    return Guard(np.concatenate([l,v]))
with tempfile.TemporaryDirectory() as td,patch.object(sd.np,'load',side_effect=guarded_load):
    xl,yl,xv,yv,xu,yu=sd.load_source('unused',td,False)
    assert xu is None and yu is None and len(xl)==450 and len(xv)==50
rng=np.random.default_rng(7);raw=rng.normal(size=(2,2,256))+1j*rng.normal(size=(2,2,256))
pad=np.pad(raw,((0,0),(0,0),(0,256)));fft=np.fft.fft(pad)/np.sqrt(512)
z=np.concatenate([pad[:,0],pad[:,1],fft[:,0],fft[:,1]],axis=1)
clean=np.stack([z.real,z.imag],1).astype(np.float32)
warm=sd.warm_view(clean);assert warm.shape==clean.shape and np.isfinite(warm).all()
wz=warm[:,0]+1j*warm[:,1]
assert np.max(np.abs(wz[:,256:512]))==0 and np.max(np.abs(wz[:,768:1024]))==0
np.testing.assert_allclose(wz[:,1024:1536],np.fft.fft(wz[:,:512])/np.sqrt(512),atol=1e-6)
np.testing.assert_array_equal(warm,sd.warm_view(clean))
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.manual_seed(299)
model=train.create_model('riei',device);opts=train.make_optimizers(model,'riei')
x=torch.randn(2,2,2,2048,device=device);labels=torch.tensor([[0,0],[1,1]],device=device)
checks=[]
for epoch,u in [(19,None),(20,None),(19,x),(20,x)]:
    m=train.train_step(model,'riei',opts,x,labels,u,None,epoch,threshold=0.)
    assert m['pseudo_active']==(u is not None and epoch>=20)
    assert np.isfinite(m['loss']) and np.isfinite(m['disentangle_loss'])
    if u is None or epoch<20:assert m['selected']==0 and m['pseudo_ce']==0
    else:assert m['selected']==4 and m['pseudo_ce']>0
    checks.append(dict(epoch=epoch,has_u=u is not None,active=m['pseudo_active'],selected=m['selected']))
result=dict(status='PASS',device=str(device),compile=True,no_u_guard=True,warm_layout=True,warm_deterministic=True,boundary_steps=checks)
(P/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
