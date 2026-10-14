"""Synthetic no-query CE update and checkpoint roundtrip for every architecture."""
import argparse
import gc
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.standard_ce_baselines.common import write
from experiments.standard_ce_baselines.model import VARIANTS,build


def smoke(output,device='cpu',batch=2,sigtime=False):
    torch.set_num_threads(2);torch.manual_seed(37)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    result=[]
    if sigtime:
        # One existing real checkpoint, legal source contract, synthetic IQ only.
        from experiments.standard_ce_baselines.common import PROJECT,RUN,read
        from experiments.standard_ce_baselines.predict import check
        source=Path(PROJECT)/'runs'/RUN/'residual_fusion-s2026092701/source'
        c=read(source/'resolved_config.json')
        ic=dict(variant='residual_fusion',model_seed=2026092701,source_output=str(source))
        payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
        check(ic,read(source/'completion.json'),read(source/'initialization.json'),
              read(source/'source_contract.json'),read(c['source_contract']),c,payload,read(source/'freeze.json'))
        reference=build('residual_fusion');reference.load_state_dict(payload['model'],strict=True)
        reference.eval()
        with torch.no_grad():scores=reference(torch.zeros(2,2,256))
        if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Real checkpoint no-query smoke failed')
        del reference,payload
    for variant in (('sig_time',) if sigtime else VARIANTS):
        model=build(variant).to(device);model.train();x=torch.randn(batch,2,256,device=device);y=torch.arange(batch,device=device)%6
        scores=model(x)
        if scores.shape!=(batch,6) or not torch.isfinite(scores).all():raise ValueError('Forward invalid: '+variant)
        loss=F.cross_entropy(scores,y);loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):raise ValueError('CE gradient invalid: '+variant)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);optimizer.step()
        model.eval()
        with torch.no_grad():expected=model(x)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt';torch.save({'model':model.state_dict(),'scratch_only':True},path)
            payload=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(payload['model'],strict=True)
            with torch.no_grad():actual=model(x)
            if not torch.equal(expected,actual):raise ValueError('Checkpoint roundtrip changed output')
        item=dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
            gradient_used_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
            synthetic_ce=float(loss.detach()),batch=batch,device=device,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            no_query=True,checkpoint_roundtrip=True)
        result.append(item);print(item,flush=True)
        del model,optimizer,payload,x,y,scores,expected,actual,grads,loss;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    write(output,dict(status='PASS',models=result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=2);p.add_argument('--sigtime',action='store_true');a=p.parse_args();smoke(a.output,a.device,a.batch,a.sigtime)
