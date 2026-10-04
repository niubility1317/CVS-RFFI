"""Focused equivalence checks, synthetic tensors only."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

def run(root,device):
    sys.path[:0]=[str(Path(__file__).parent),str(root),str(root/'code')]
    import torch
    from experiments.cvs_phase1_stack.design import rows,config,make_args
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.source import clean
    from fast_execution import batch_clean,configure,incremental_native_writer
    c=config(next(r for r in rows() if r['stage']=='r3' and r['arm']=='base'),['leo'])
    checks=[]
    with installed(c) as n:
        from cvsrffi.a1_fast_runtime import GradientSnapshot
        from cvsrffi.source_validation_reuse import SourceValidationReuse
        torch.set_num_threads(2);torch.manual_seed(117);args=configure(make_args(c,device));args.num_domains=15;args.input_len=256;model=n.build_baseline_model(args,torch.device(device))
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.randn_like(p)
        for clipped in [False,True]:
            if clipped:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            snap=GradientSnapshot.capture(model)
            assert snap.norm()==n._grad_norm(model)
            assert snap.first_nonfinite()==n._first_nonfinite_gradient(model)
        checks.append('gradient_norm_before_after_clip_exact')
        next(model.parameters()).grad.flatten()[0]=float('nan')
        assert GradientSnapshot.capture(model).first_nonfinite()==n._first_nonfinite_gradient(model)
        checks.append('nonfinite_gradient_exact')
        v={'f':torch.tensor(.123,device=device),'nan':torch.tensor(float('nan'),device=device),'int':torch.tensor(7,device=device),'arr':torch.arange(3,device=device),'tuple':(None,False,3.)}
        assert batch_clean(v)==clean(v);checks.append('nested_log_values_exact')
        model.eval();x=torch.randn(4,2,256,device=device)
        with torch.no_grad():
            features=model(x,return_aux=True)
        # Exercise the cache as a value store; current model type/adapter must be allowed.
        z=next(v for v in features.values() if torch.is_tensor(v) and v.ndim==2) if isinstance(features,dict) else features[0]
        cache=SourceValidationReuse(model);assert cache.allowed
        cache.record(0,x,z);assert torch.equal(cache.take(0,x),z)
        cache.record(1,x,z);assert cache.take(1,x+1) is None
        cache.record(2,x,z)
        with torch.no_grad():next(model.parameters()).add_(.001)
        assert cache.take(2,x) is None;checks.append('cache_exact_and_input_parameter_invalidation')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);rows1=[{'epoch':1,'loss':1.,'details':[1,2]}];rows2=rows1+[{'epoch':2,'loss':.5,'extra':3.,'details':[2,3]}]
            n._write_ssdg_epoch_telemetry(tmp/'old.csv',tmp/'old.jsonl',rows2)
            with incremental_native_writer(n):
                n._write_ssdg_epoch_telemetry(tmp/'new.csv',tmp/'new.jsonl',rows1)
                n._write_ssdg_epoch_telemetry(tmp/'new.csv',tmp/'new.jsonl',rows2)
            assert [json.loads(s) for s in (tmp/'old.jsonl').read_text().splitlines()]==[json.loads(s) for s in (tmp/'new.jsonl').read_text().splitlines()]
            import csv
            assert list(csv.DictReader((tmp/'old.csv').open()))==list(csv.DictReader((tmp/'new.csv').open()))
            checks.append('incremental_jsonl_csv_values_schema_exact')
    return dict(status='VERIFIED',device=device,torch=torch.__version__,checks=checks,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path)
    a=p.parse_args();r=run(a.worker_root,a.device)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r))
