"""Prepare one frozen-DURA output per canonical clean physical record."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .clean_pairing import RECEIVERS
from .io import read, require, write


def build(clean_root, output_root, restore_batch, *, batch_size, dura_spec, receivers=RECEIVERS,
          validate_final=None, source_parity=None):
    clean_root=Path(clean_root).resolve();output_root=Path(output_root)
    source=read(clean_root/'completion.json')
    require(source.get('status')=='CLEAN_PAIRED_COMPLETE','Clean input incomplete')
    require(batch_size>0 and set(receivers)<=set(source['receivers']),'Invalid batch or RX')
    output_root.mkdir(parents=True,exist_ok=False);(output_root/'banks').mkdir()
    started=time.perf_counter();records=0
    for rx in receivers:
        raw=np.load(clean_root/'banks'/rx/'clean.npy',mmap_mode='r',allow_pickle=False)
        require(raw.shape==(source['receivers'][rx],2,256) and raw.dtype==np.float32,'Wrong clean bank')
        folder=output_root/'banks'/rx;folder.mkdir()
        result=np.lib.format.open_memmap(folder/'restored.npy',mode='w+',dtype=np.float32,shape=raw.shape)
        for start in range(0,len(raw),batch_size):
            x=np.asarray(raw[start:start+batch_size]);y=np.asarray(restore_batch(x))
            require(y.shape==x.shape and y.dtype==np.float32 and np.isfinite(y).all(),'Wrong DURA output')
            result[start:start+len(y)]=y
        result.flush();records+=len(raw)
        print('CLEAN_DURA_RX_COMPLETE',rx,len(raw),flush=True)
    require(records==sum(source['receivers'][r] for r in receivers),'Wrong output count')
    if validate_final is not None:validate_final()
    result=dict(status='CLEAN_DURA_COMPLETE' if set(receivers)==set(RECEIVERS) else 'CLEAN_DURA_SUBSET',
        clean_root=str(clean_root),restorer_id='dura_v1',restorer_spec=dura_spec,
        receivers=list(receivers),physical_records=records,batch_size=batch_size,
        model_updates=0,identity_model_used=False,truth_read=False,query_statistics_fit=False,
        source_batch_parity=source_parity,seconds=time.perf_counter()-started)
    write(output_root/'completion.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();rt=read(args.runtime)
    import torch
    from .io import activate
    from .features import load_G,tensor
    from .clean_source_smoke import source_samples
    spec=rt['DURA'];require(spec.get('kind')=='original_F0_LU' and spec.get('roles')=='LU' and
        spec.get('checkpoint','').endswith('20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'),
        'DURA v1 checkpoint required')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(gpu.isdigit(),'Single explicit physical GPU required')
    free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    require(free>=6000,'Insufficient free GPU memory')
    # Preserve other owners; a free-memory reading alone does not establish that
    # a currently active process will remain unaffected by a long preprocessing job.
    active=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    require(not active.strip(),'GPU has existing compute processes')
    activate(rt);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=load_G(rt,'cuda:0');model=engine.pipeline.model
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    sample=tensor(source_samples(rt,128),'cuda:0')
    with torch.inference_mode():
        individual=torch.cat([engine(sample[i:i+1])[0] for i in range(len(sample))])
        parity={}
        for size in sorted({1,5,10,20,int(rt.get('batch_size',128))}):
            combined=torch.cat([engine(sample[i:i+size])[0] for i in range(0,len(sample),size)])
            require(torch.allclose(combined,individual,rtol=2e-4,atol=2e-5),'DURA source batch parity failed')
            parity[str(size)]=float((combined-individual).abs().max())
    def restore(x):
        with torch.inference_mode():return engine(tensor(x,'cuda:0'))[0].detach().cpu().numpy().copy()
    def validate_final():
        require(all(torch.equal(state[k],v) for k,v in model.state_dict().items()),'DURA state changed')
    result=build(rt['clean_root'],args.output,restore,batch_size=rt.get('batch_size',128),dura_spec=spec,
                 validate_final=validate_final,source_parity=parity)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
