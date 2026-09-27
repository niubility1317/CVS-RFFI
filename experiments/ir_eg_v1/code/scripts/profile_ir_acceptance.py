"""Bounded synthetic profiling, not a ManySig training speed claim."""
import argparse
import gc
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tests')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ['IR_ACCEPTANCE_DEVICE']=a.device
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    from ir_fixtures import make_case,run
    records=[]
    for name,impl,method,kappa in [('IR_ref','reference','IR_EG',.25),('IR_cached','cached','IR_EG',.25),
                                 ('BR_EG','cached','BR_IR_EG',0.),('BR_IR','cached','BR_IR_EG',.25)]:
        setup_started=time.perf_counter()
        warmup=make_case(impl,method,kappa)
        for _ in range(2):
            warmup.ctx.pending=None;run(warmup)
        if a.device=='cuda':torch.cuda.synchronize()
        del warmup;gc.collect()
        setup_seconds=time.perf_counter()-setup_started
        case=make_case(impl,method,kappa)
        if a.device=='cuda':
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();rows=[]
        for i in range(10):
            case.ctx.pending=None
            result=run(case);rows.append(result.telemetry)
        if a.device=='cuda':torch.cuda.synchronize()
        record=dict(method=name,device=a.device,accepted=10,physical_exposure=100,
                    warmup_steps_excluded=2,setup_and_warmup_seconds=setup_seconds,
                    elapsed_seconds=time.perf_counter()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                    history_bytes=case.solver.history.bytes,
                    model_forward_calls=sum(r['model_forward_calls'] for r in rows),
                    full_backward_calls=sum(r.get('full_backward_calls',0) for r in rows),
                    head_only_backward_calls=sum(r.get('head_only_backward_calls',0) for r in rows),
                    response_applied=sum(bool(r.get('applied')) for r in rows),
                    source_convergence_cost=None)
        records.append(record)
        del case;gc.collect()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'profiling.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(scope='ten-step synthetic joint toy; includes solver snapshot, head replay, CG and external commit; excludes real data, teacher and evaluation',
                rffi_efficiency_claim=False,source_convergence_cost_reason='no real training or frozen convergence run',records=records)
    (a.output/'profiling.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
