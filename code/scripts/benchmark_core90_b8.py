"""Paired fixed-source-context B8 benchmark. No target access or checkpoint load."""
import os
# Must precede any CUDA context initialization, including module imports.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking.runtime import build_model,configure_determinism
from cvsrffi.game_tracking.step_context import prepare_context,Core90Objective
from cvsrffi.game_tracking.head_lookahead import Core90ReusableGraph
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.schedule import build_stage_state


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def compare_endpoint(actual,reference,path=''):
    if torch.is_tensor(actual):
        if path.endswith('/step') or not actual.is_floating_point():
            assert torch.equal(actual,reference),path
        else:torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5,msg=path)
    elif isinstance(actual,dict):
        assert actual.keys()==reference.keys(),path
        for key in actual:compare_endpoint(actual[key],reference[key],path+'/'+str(key))
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(reference),path
        for i,(a,b) in enumerate(zip(actual,reference)):compare_endpoint(a,b,path+'/'+str(i))
    else:assert actual==reference,path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=['all','reference','head_grad_only','graph_reuse'],default='all')
    p.add_argument('--stage',choices=['all','1','80','131'],default='all')
    p.add_argument('--warmup',type=int,default=20); p.add_argument('--steps',type=int,default=100)
    p.add_argument('--repeats',type=int,default=3); p.add_argument('--batch-size',type=int,default=90)
    p.add_argument('--device',default='cuda'); p.add_argument('--wisig-pkl')
    p.add_argument('--synthetic',action='store_true'); p.add_argument('--isolated-device',action='store_true')
    p.add_argument('--allow-nondeterministic',action='store_true',
                   help='Diagnostic reproduction only; default benchmark requires deterministic algorithms.')
    p.add_argument('--telemetry-interval',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    cli=p.parse_args()
    if min(cli.warmup,cli.steps,cli.repeats)<1: p.error('positive budgets required')
    torch.set_num_threads(2)
    configure_determinism(argparse.Namespace(game_deterministic=not cli.allow_nondeterministic))
    if cli.allow_nondeterministic:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic=False
        torch.backends.cudnn.benchmark=False
    torch.manual_seed(12345)
    device=torch.device(cli.device)
    args=parse_args(['--output_dir','unused','--num_workers','0']+(['--game_synthetic'] if cli.synthetic else [])+(['--wisig_pkl',cli.wisig_pkl] if cli.wisig_pkl else []))
    source=build_source(args)
    batch=next(iter(source.loader('train',cli.batch_size,seed=921,shuffle=True)))
    ub=next(iter(source.loader('unlabeled',cli.batch_size,seed=921,shuffle=True)))
    stages=[1,80,131] if cli.stage=='all' else [int(cli.stage)]
    implementations=['reference','head_grad_only','graph_reuse'] if cli.implementation=='all' else [cli.implementation]
    report={'schema':'core90_b8_benchmark_v2','device':str(device),'torch':torch.__version__,
            'source_kind':'SYNTHETIC_FUNCTIONAL' if cli.synthetic else 'REAL_SOURCE_FIXED_CONTEXT',
            'checkpoint':'scratch_only','budgets':vars(cli)|{'output':str(cli.output)},
            'determinism':{'algorithms':torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
                'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
                'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32},
            'source_counts':source.info['counts'],
            'wallclock_attribution':'VERIFIED_ISOLATED_BY_OPERATOR' if cli.isolated_device else 'UNKNOWN_SHARED_LOAD',
            'rows':[]}
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    def write():
        cli.output.parent.mkdir(parents=True,exist_ok=True)
        cli.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for epoch in stages:
            torch.manual_seed(12345)
            base=build_model(args,len(source.domains),device).train()
            teacher=copy.deepcopy(base).eval(); teacher.requires_grad_(False)
            proto=PrototypeMemoryBank(args.num_classes,len(source.domains)); proto._lazy_init(160,device,torch.float32)
            template=prepare_context(batch,ub if epoch>=131 else None,base,teacher,args,epoch,1,
                                     _loss_weights(args,build_stage_state(epoch,args)),torch.Generator(device=device).manual_seed(777))
            initial_rng=RNGState.capture()
            reference_state=reference_opt=reference_rng=reference_mask=None
            # Equivalence always includes reference even for a single timed implementation.
            for impl in dict.fromkeys(['reference']+implementations):
                model=copy.deepcopy(base); bank=copy.deepcopy(proto); ctx=copy.deepcopy(template)
                opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                solver=GameSolver(model,opt,'head_lookahead',b8_impl=impl,max_grad_norm=5)
                objective=Core90Objective(model,args,bank); initial_rng.restore()
                solver.step(Core90ReusableGraph(objective,ctx) if impl=='graph_reuse' else lambda:objective(ctx))
                if impl=='reference':
                    reference_state=copy.deepcopy(model.state_dict()); reference_opt=copy.deepcopy(opt.state_dict()); reference_rng=RNGState.capture()
                    reference_mask=None if ctx.strong_mask is None else ctx.strong_mask.clone()
                else:
                    for key,value in model.state_dict().items(): torch.testing.assert_close(value,reference_state[key],atol=1e-6,rtol=1e-5)
                    for key,state in opt.state_dict()['state'].items():
                        for name,value in state.items(): torch.testing.assert_close(value,reference_opt['state'][key][name],atol=1e-6,rtol=1e-5)
                    assert torch.equal(reference_rng.cpu,RNGState.capture().cpu)
                    for a,b in zip(reference_rng.cuda,RNGState.capture().cuda): assert torch.equal(a,b)
                    if reference_mask is not None: assert torch.equal(reference_mask,ctx.strong_mask)
                for name,value in vars(proto).items():
                    if torch.is_tensor(value): assert torch.equal(value,vars(bank)[name])
            reference_endpoints=[]
            for impl in implementations:
                timings=[]; peaks=[]; counters=[]
                for repeat in range(cli.repeats):
                    model=copy.deepcopy(base); bank=copy.deepcopy(proto); ctx=copy.deepcopy(template)
                    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                    solver=GameSolver(model,opt,'head_lookahead',b8_impl=impl,max_grad_norm=5,telemetry_interval=cli.telemetry_interval)
                    objective=Core90Objective(model,args,bank); initial_rng.restore()
                    counts={'model_forward':0,'head_forward':0,'field_backward':0,'telemetry_backward':0}
                    def count_model(*unused): counts['model_forward']+=1
                    def count_head(*unused): counts['head_forward']+=1
                    mh=model.register_forward_hook(count_model); hh=model.adv_head.register_forward_hook(count_head)
                    def step():
                        ctx.origin_features=None; ctx.origin_terms={}
                        wrapped=impl=='graph_reuse' or (cli.telemetry_interval>0 and solver.steps%cli.telemetry_interval==0)
                        r=solver.step(Core90ReusableGraph(objective,ctx) if wrapped else lambda:objective(ctx))
                        counts['field_backward']+=r.field_evaluations
                        counts['telemetry_backward']+=(r.telemetry or {}).get('telemetry_backward_calls',0)
                    for _ in range(cli.warmup): step()
                    for key in counts: counts[key]=0
                    sync()
                    if device.type=='cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                        baseline_allocated=torch.cuda.memory_allocated(device)
                        baseline_reserved=torch.cuda.memory_reserved(device)
                    start=time.perf_counter()
                    for _ in range(cli.steps): step()
                    sync(); timings.append((time.perf_counter()-start)/cli.steps)
                    peaks.append({'allocated':torch.cuda.max_memory_allocated(device),'reserved':torch.cuda.max_memory_reserved(device),
                                  'baseline_allocated':baseline_allocated,'baseline_reserved':baseline_reserved,
                                  'incremental_allocated':torch.cuda.max_memory_allocated(device)-baseline_allocated} if device.type=='cuda' else {'allocated':None,'reserved':None})
                    mh.remove(); hh.remove(); counters.append(dict(counts))
                    # Outside the timed region: catch accumulated trajectory
                    # divergence without adding reverse passes to wall-clock work.
                    end_rng=RNGState.capture()
                    endpoint=cpu_copy({'model':model.state_dict(),'optimizer':opt.state_dict(),
                                       'mask':ctx.strong_mask,'rng_cpu':end_rng.cpu,'rng_cuda':end_rng.cuda,
                                       'rng_python':end_rng.python,
                                       'rng_numpy':(end_rng.numpy[0],end_rng.numpy[1].tolist(),*end_rng.numpy[2:]),
                                       'prototype':vars(bank)})
                    if impl=='reference':reference_endpoints.append(endpoint)
                    elif reference_endpoints:
                        try:compare_endpoint(endpoint,reference_endpoints[repeat])
                        except Exception:
                            artifact=cli.output.with_name(cli.output.stem+f'_E{epoch}_{impl}_repeat{repeat}_endpoint_failure.pt')
                            torch.save({'reference':reference_endpoints[repeat],'actual':endpoint},artifact)
                            report['endpoint_failure_artifact']=str(artifact)
                            raise
                report['rows'].append({'stage':epoch,'implementation':impl,'equivalence':'PASSED_ATOL_1e-6_RTOL_1e-5',
                    'equivalence_scope':'single_same_initial_state_step_before_timing',
                    'timed_endpoint_equivalence':'REFERENCE_BASELINE' if impl=='reference' else ('PASSED_ATOL_1e-6_RTOL_1e-5' if reference_endpoints else 'UNAVAILABLE_REFERENCE_NOT_TIMED'),
                    'seconds_per_step':timings,'median':statistics.median(timings),'range':[min(timings),max(timings)],
                    'peaks':peaks,'counts':counters,'telemetry_interval':cli.telemetry_interval,
                    'pseudo_selected':ctx.pseudo_selected,'origin_terms':ctx.origin_terms})
                write(); print(json.dumps(report['rows'][-1]),flush=True)
    except Exception as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        write(); raise
    write()


if __name__=='__main__': main()
