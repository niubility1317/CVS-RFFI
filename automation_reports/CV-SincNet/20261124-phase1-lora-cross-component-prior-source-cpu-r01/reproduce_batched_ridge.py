
import json, subprocess, sys
script = '''
import torch,json,sys
torch.set_num_threads(2)
g=torch.Generator().manual_seed(1)
x=torch.randn((6,25,384),dtype=torch.float64,generator=g)
b=x.mT @ torch.randn((6,25,128),dtype=torch.float64,generator=g)
a=x.mT @ x+.1*torch.eye(384,dtype=torch.float64)
mode=sys.argv[1]
w=(torch.linalg.solve(a/25,b/25) if mode=='lu' else torch.cholesky_solve(b,torch.linalg.cholesky(a)))
ratio=((a@w-b).norm()/b.norm()).item()
print(json.dumps(dict(mode=mode,finite=bool(torch.isfinite(w).all()),relative_residual=ratio)),flush=True)
'''
for mode in ('lu','chol'):
    try:
        q=subprocess.run([sys.executable,'-c',script,mode],capture_output=True,text=True,timeout=12)
        print(mode,q.returncode,q.stdout[-1000:],q.stderr[-1200:],flush=True)
    except subprocess.TimeoutExpired as e:
        print(mode,'TIMEOUT',str(e.stderr)[-1200:],flush=True)
