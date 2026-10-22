"""The launcher's one source-checkpoint/no-query compatibility smoke."""
import copy
from pathlib import Path
import numpy as np
import torch
from .io import read,write


def run(rt,output):
    from .bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    from .g_lineage import load,construct,objective,trainable
    from .teacher_targets import EpochTargets
    jobs=read(rt['source_jobs'])
    job=next(j for j in jobs if j['kind']=='G' and '/label01-G-L-s2026092701' in j['id'])
    config=read(job['config']);contract=read(config['cache']['source_contract'])
    checkpoint=Path(config['output_root'])/'r17/final.pt'
    teacher,_=load(checkpoint,contract,'L');teacher.eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026102201)
        initial=construct('r38',teacher);a=copy.deepcopy(initial);b=copy.deepcopy(initial)
        x=torch.randn(2,2,256);clean=x*.9
        cache=EpochTargets(teacher,2,1,{'diagnostic':'synthetic-source-only','checkpoint':str(checkpoint)})
        desired=cache.get(np.arange(2),x)
        online=objective(a,teacher,'r38',x,clean,1)
        cached=objective(b,teacher,'r38',x,clean,1,desired=desired,skip_zero_spectral=True)
        torch.testing.assert_close(online,cached,atol=2e-6,rtol=2e-5)
        online.mean().backward();cached.mean().backward()
        max_abs=0.
        for left,right in zip(trainable(a,'r38'),trainable(b,'r38')):
            if (left.grad is None)!=(right.grad is None):raise ValueError('Gradient presence differs')
            if left.grad is not None:
                torch.testing.assert_close(left.grad,right.grad,atol=3e-5,rtol=1e-4)
                max_abs=max(max_abs,float((left.grad-right.grad).abs().max()))
        cache.get(np.arange(2),x)
        if cache.teacher_samples!=2:raise ValueError('Teacher input was recomputed')
    write(output,dict(status='PASS',torch=torch.__version__,device='cpu',checkpoint=str(checkpoint),
        gradient_max_abs=max_abs,teacher_samples=cache.teacher_samples,query_reads=0,TX_truth_reads=0,F_reads=0))
