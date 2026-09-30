"""One real-source IQ batch; scratch models only, no target or checkpoint access."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--method',required=True,choices=['asknet','wavemlp','difl'])
p.add_argument('--release',type=Path,required=True)
p.add_argument('--data-root',type=Path,required=True)
args=p.parse_args()
torch.set_num_threads(2)
folder=args.release/'sources'/args.method/('WiSig' if args.method=='asknet' else '')
sys.path.insert(0,str(folder))
os.environ['WISIG_DATA_ROOT']=str(args.data_root/'author_layout')
if args.method in ('asknet','wavemlp'):
 import main
 sys.argv=['main.py']
 conf=main.parse_args()
 model=main.build_model(conf,6) if args.method=='asknet' else main._build_model_from_conf(conf,6,16)
 x,y=main.load_single_dataset('ManySig',3,1,6,'non_equalized')
 x=torch.tensor(x[:4]).float(); y=torch.tensor(y[:4]).long()
 output=model(x)
 if isinstance(output,tuple): output=output[0]
 loss=torch.nn.functional.cross_entropy(output,y)
else:
 import Network1, teacherNet
 x=torch.tensor(np.load(args.data_root/'difl/rx_2-19.npy')[:4].transpose(0,2,1)).float()
 model=Network1.DNN()
 features,output=model(x)
 assert features.shape==(4,512)
 teacher=teacherNet.DNN()
 phase=torch.angle(torch.fft.fft(torch.complex(x[:,0],x[:,1]),dim=1)).unsqueeze(1)
 teacher.eval()
 with torch.no_grad(): tf,to=teacher(phase)
 loss=torch.nn.functional.cross_entropy(output,torch.zeros(4,dtype=torch.long))+torch.nn.functional.mse_loss(features[:,:256],tf)
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None)
print(f'SOURCE_SMOKE_PASS method={args.method} loss={loss.item():.6f} checkpoint=scratch target_access=false')
