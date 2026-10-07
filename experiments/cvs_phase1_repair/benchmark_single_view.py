"""Synthetic timing only; no dataset/checkpoint/query/truth read."""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
import sys
import torch
from experiments.cvs_phase1_overlay.contract import build_model
from experiments.cvs_phase1_overlay.augmentation import OriginalLEO,loss_for_batch
from experiments.cvs_phase1_repair.single_view import SingleViewLEO,single_view_loss
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY


def run(output,device='cpu'):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261008)
    device=torch.device(device)
    if device.type!='cpu':raise ValueError('This preregistration is local CPU only')
    x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;domains=torch.arange(128,device=device)%5
    batch=dict(meta=[dict(sample_id=f'synthetic:{i}',rx_i=i%5,day_i=1) for i in range(128)])
    records=[]
    with numerical_context(FULL_FP32_POLICY):
        for epoch in (1,100):
            for mode in ('ce','concat','single_view'):
                torch.manual_seed(20261008)
                model=build_model('ce').to(device)
                (out/mode).mkdir(exist_ok=True)
                cp=out/mode/f'e{epoch}-own-scratch.pt'
                torch.save(model.state_dict(),cp)
                model.load_state_dict(torch.load(cp,map_location=device,weights_only=False),strict=True)
                model.eval()
                with torch.no_grad():assert model(x[:2]).shape==(2,6)
                model.train();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
                folder=out/mode/f'e{epoch}'
                aug=SingleViewLEO(folder,seed=20261008) if mode=='single_view' else OriginalLEO(folder) if mode=='concat' else None
                timings=[];changed=[]
                for step in range(1,8):
                    start=time.perf_counter();opt.zero_grad(set_to_none=True)
                    if mode=='single_view':loss,info=single_view_loss(model,x,y,domains,batch,aug,epoch,step)
                    else:loss,info=loss_for_batch(model,x,y,domains,batch,aug,epoch,step)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite synthetic loss')
                    loss.backward();opt.step();duration=time.perf_counter()-start
                    if step>2:
                        timings.append(duration)
                        changed.append(info['rendered_samples'] if mode=='single_view' else 128 if info['channel_applied'] else 0)
                record=dict(mode=mode,epoch=epoch,batch_size=128,model_input_samples=128 if mode!='concat' else 256,
                    measured_steps=5,warmup_steps=2,step_seconds=timings,median_seconds=statistics.median(timings),
                    min_seconds=min(timings),max_seconds=max(timings),rendered_samples=changed,
                    total_parameters=sum(p.numel() for p in model.parameters()),peak_memory_bytes=None)
                records.append(record);print(json.dumps(record),flush=True)
    value=dict(status='BENCHMARK_COMPLETE',scope='synthetic local CPU execution cost only; no accuracy evidence',
        python=sys.executable,torch_version=torch.__version__,device=str(device),threads=2,
        hardware=os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),os=platform.platform(),
        source_dataset_read=False,target_access=False,checkpoint_ancestry='own scratch only',rows=records)
    (out/'results.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();run(args.output)
