"""Isolated synthetic CUDA decision equivalence/timing, never an E200 claim."""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.decision import decision_margin_loss_reference,decision_margin_loss_vectorized


def profile_stages(fn, invoke):
    """Synchronized wall-time sections of the actual production function.

    A source-identical function copy has four boundary calls inserted; no line
    tracing overhead is paid inside the CPU mask loop. These sections diagnose
    cost. Separately measured uninstrumented end-to-end time is authoritative.
    GPU sections include host dispatch and explicit device/host synchronization,
    not a claim about pure kernel execution duration.
    """
    lines,start=inspect.getsourcelines(fn)
    markers={}
    for offset,line in enumerate(lines):
        stripped=line.strip()
        if stripped=='selection = []':
            markers[start+offset]='cpu_reference_mask'
        elif stripped.startswith('margins = logits.gather'):
            markers[start+offset]='gpu_reference_and_loss'
        elif stripped.startswith('selected = gap[valid].detach()'):
            markers[start+offset]='diagnostics'
    if set(markers.values()) != {'cpu_reference_mask','gpu_reference_and_loss','diagnostics'}:
        raise RuntimeError('production stage anchors changed; update benchmark boundaries')
    samples={}
    active='input_validation_and_metadata_transfer'
    torch.cuda.synchronize()
    last=time.perf_counter()
    def checkpoint(next_stage):
        nonlocal active,last
        torch.cuda.synchronize()
        now=time.perf_counter()
        samples[active]=(now-last)*1000
        active=next_stage
        last=now
    instrumented=[]
    for offset,line in enumerate(lines):
        if start+offset in markers:
            instrumented.append('    _benchmark_checkpoint('+repr(markers[start+offset])+')\n')
        if line.startswith('    return '):
            instrumented.append('    _benchmark_checkpoint(None)\n')
        instrumented.append(line)
    namespace=dict(fn.__globals__,_benchmark_checkpoint=checkpoint)
    exec(compile(''.join(instrumented),'<instrumented production decision>','exec'),namespace)
    torch.cuda.synchronize()
    last=time.perf_counter()
    invoke(namespace[fn.__name__])
    return samples


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this explicitly GPU benchmark')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn(128,6)
    labels=torch.arange(128)%6
    x[torch.arange(128),labels]+=2
    rx=(torch.arange(128)%5).tolist()
    physical=[i if i%17 else 'duplicate' for i in range(128)]
    condition=['clean']*128
    labels=labels.cuda()
    payload=dict(kind='isolated_synthetic_decision_benchmark_not_E200_speed',device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,shape=[128,6],receivers=5,seed=42,warmup=args.warmup,repeats=args.repeats,
        scope='fixed logits only; no backbone, optimizer, dataset, checkpoint or target',modes={})
    payload['legacy_fp16_compatibility']='Original CUDA half quantile raised dtype RuntimeError; reference now explicitly promotes quantile input to FP32 and casts result back. FP32 legacy math is unchanged.'
    for name,dtype,amp in [('fp32',torch.float32,False),('autocast_fp16',torch.float16,True)]:
        data=x.to(device='cuda',dtype=dtype)
        def run(fn,backward=False):
            leaf=data.detach().clone().requires_grad_(True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                loss,diagnostics=fn(leaf,labels,rx,physical,condition)
            grad=torch.autograd.grad(loss,leaf)[0] if backward else None
            return loss,diagnostics,grad
        expected=run(decision_margin_loss_reference,True)
        actual=run(decision_margin_loss_vectorized,True)
        atol,rtol=(2e-6,2e-5) if not amp else (1e-4,2e-3)
        checks={}
        for key,a,b in [('loss',expected[0],actual[0]),('reference',expected[1]['reference'],actual[1]['reference']),
                        ('original_logits_gradient',expected[2],actual[2])]:
            a,b=a.detach(),b.detach()
            diff=(a.float()-b.float()).abs()
            checks[key]=dict(max_abs=float(diff.max()),mean_abs=float(diff.mean()),
                max_relative_on_abs_reference_gt_atol=float((diff/a.float().abs().clamp_min(atol)).max()),
                allclose=bool(torch.allclose(a,b,atol=atol,rtol=rtol)))
        checks['valid_mask_equal']=bool(torch.equal(expected[1]['valid_mask'],actual[1]['valid_mask']))
        mode=dict(dtype=str(dtype),autocast=amp,atol=atol,rtol=rtol,equivalence=checks,
            reference_loss=float(expected[0].detach()),vectorized_loss=float(actual[0].detach()),timing={})
        for method,fn in [('reference',decision_margin_loss_reference),('vectorized',decision_margin_loss_vectorized)]:
            for backward in (False,True):
                for _ in range(args.warmup):
                    run(fn,backward)
                torch.cuda.synchronize()
                durations=[]
                for _ in range(args.repeats):
                    before=time.perf_counter()
                    run(fn,backward)
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter()-before)*1000)
                mode['timing'][method+('_forward_backward' if backward else '_forward_all_diagnostics')]=dict(
                    median_ms=statistics.median(durations),min_ms=min(durations),max_ms=max(durations),samples_ms=durations)
        stages=[profile_stages(decision_margin_loss_vectorized,lambda instrumented:run(instrumented)) for _ in range(args.repeats)]
        mode['vectorized_synchronized_stage_ms']={key:statistics.median([s[key] for s in stages]) for key in stages[0]}
        mode['stage_timing_caveat']='Source-identical function copy plus boundary synchronization only; GPU stage includes host dispatch. No per-line tracing overhead. Uninstrumented timings above are end-to-end.'
        mode['passed']=checks['valid_mask_equal'] and all(checks[k]['allclose'] for k in ('loss','reference','original_logits_gradient'))
        payload['modes'][name]=mode
    payload['passed']=all(m['passed'] for m in payload['modes'].values())
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError('preserve prior acceptance artifact; choose a new output')
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),passed=payload['passed'],modes=payload['modes']),allow_nan=False))
    if not payload['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
