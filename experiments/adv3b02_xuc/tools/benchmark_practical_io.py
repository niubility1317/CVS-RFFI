"""Additional input-copy and telemetry benchmark; synthetic data only."""
import argparse,json,statistics,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.practical_adapter import _input_array
from cvsrffi.incremental_telemetry import IncrementalTelemetry
from scripts.train_rc4_practical import native

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);device=torch.device(a.device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    def measure(fn,n=5):
        fn();times=[]
        for _ in range(n):
            sync();t=time.perf_counter();value=fn();sync();times.append(time.perf_counter()-t)
        return dict(median_s=statistics.median(times),samples_s=times),value
    x=torch.randn(128,2,256,device=device)
    old,ref=measure(lambda:_input_array(x,False),31)
    new,value=measure(lambda:_input_array(x,True),31)
    assert np.array_equal(ref,value)
    result=dict(scope='synthetic IO components, not whole epoch',input_copy=dict(baseline=old,optimized=new,
        speedup=old['median_s']/new['median_s'],exact=True))
    rows=[dict(epoch=e,**{f'metric{k}':e/(k+1) for k in range(500)}) for e in range(200)]
    with tempfile.TemporaryDirectory(prefix='cvs-telemetry-benchmark-') as directory:
        root=Path(directory)
        def legacy():
            for i in range(1,len(rows)+1):native._write_ssdg_epoch_telemetry(root/'old.csv',root/'old.jsonl',rows[:i])
        def fast():
            writer=IncrementalTelemetry(root/'new.csv',root/'new.jsonl')
            for i in range(1,len(rows)+1):writer.write(rows[:i])
        old,_=measure(legacy,3);new,_=measure(fast,3)
        assert all((root/f'old.{s}').read_bytes()==(root/f'new.{s}').read_bytes() for s in ('csv','jsonl'))
        result['telemetry']=dict(baseline=old,optimized=new,speedup=old['median_s']/new['median_s'],
            exact=True,epochs=200,fields=501,saved_total_s=old['median_s']-new['median_s'])
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
