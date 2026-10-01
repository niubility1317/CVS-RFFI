"""Synthetic-only CPU software timing; no fitting, data loading, or accuracy claims.

The literal scoring expansions below are not trained models or KKT solutions.
This tool is deliberately separate from every real benchmark launcher.
"""
from __future__ import annotations
import argparse
import ctypes
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

THREAD_ENV = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
PLAN = dict(old_class_count=6,new_class_count=20,ks=[1,5,10,20],query_count=8,
            warmup_repetitions=1,timed_repetitions=3,
            model_seed=2026100101,data_seed=2026100102,evaluation_seed=2026100103,
            model_streams='SeedSequence([model_seed,0]) for B; [model_seed,1] for C',
            split_seed=None,augmentation_seed=None,
            absent_seed_reason='No split construction or augmentation; scoring expansions only')
RUN_ID='20261001-d92-margin-single-query-cost-synthetic-r01'
WORK_KEYS = ('raw_distance_evaluation_count','raw_distance_pair_count',
             'reference_distance_evaluation_count','reference_distance_pair_count',
             'kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count',
             'dictionary_physical_evaluation_count','intercept_addition_count')
EXTRA_SECONDS = ('packet_preparation_seconds','cache_validation_seconds')


def check_thread_environment():
    values={key:os.environ.get(key) for key in THREAD_ENV}
    if any(value!='2' for value in values.values()):
        raise ValueError('All thread environment variables must equal 2 before NumPy import: '+repr(values))
    return values


def load_runtime():
    check_thread_environment()
    if 'numpy' in sys.modules:
        raise RuntimeError('CLI requires a fresh process: NumPy was already imported')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    import numpy as np
    from cvsrffi import d92_margin_joint_local_ridge as mj
    from threadpoolctl import threadpool_info
    pools=threadpool_info()
    if any(pool.get('num_threads')!=2 for pool in pools if pool.get('user_api') in ('blas','openmp')):
        raise RuntimeError('Loaded BLAS/OpenMP pool does not use exactly 2 threads')
    return np,mj,pools


def process_peak_working_set():
    """One process-lifetime peak, including imports, construction and warmup."""
    if os.name!='nt':
        return dict(bytes=None,reason='Windows PeakWorkingSetSize unavailable on this platform')
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    try:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=(wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo failed')
        return dict(bytes=int(counters.PeakWorkingSetSize),reason=None)
    except (OSError,AttributeError) as exc:
        return dict(bytes=None,reason=type(exc).__name__+': '+str(exc))


def hardware_metadata(pools):
    return dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor() or None,
        processor_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),logical_cpu_count=os.cpu_count(),
        executable=sys.executable,python=sys.version,thread_environment=check_thread_environment(),
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','threadpoolctl')},
        threadpools=pools)


def make_features(np,mj,n,seed):
    rng=np.random.default_rng(seed)
    return {name:rng.normal(size=(n,width)) for name,width in zip(mj._NAMES,(160,96,160,160,160))}


def build_states(np,mj,k):
    if k not in PLAN['ks']:
        raise ValueError('K outside fixed synthetic plan')
    classes=tuple('synthetic-class-'+str(i).zfill(2) for i in range(26));old=classes[:6]
    raw=make_features(np,mj,26*k,np.random.SeedSequence([PLAN['data_seed'],k]))
    ids=tuple('synthetic-support-'+str(i).zfill(4) for i in range(26*k))
    def state(names,features,support_ids,prior,seed):
        n=len(support_ids);c=len(names);rng=np.random.default_rng(seed)
        b,a=mj.interaction._blocks(**features);context=mj.fcr._context(b,a)
        held={key:value[:0] for key,value in context.items()}
        U=rng.normal(size=(736,8))*.002
        ab,aa,_=mj.fcr._adapt(context,U)
        oi=np.arange(6*k);ni=np.arange(6*k,n);labels=np.repeat(np.arange(c),k)
        q=np.zeros(n);q[oi]=1/len(oi)
        problem=mj.MarginJointProblem(context,held,labels,np.empty(0,dtype=np.int64),names,q,oi,ni,
            np.zeros((n,n)),np.empty((0,n)),1.3,.7,2.,np.zeros((n,c)),np.empty((0,c)),
            'B' if prior is None else 'C_seq',128,1000000,{})
        numeric=dict(alpha=rng.normal(size=(n,c))*.05,reference=np.zeros(n),reference_self=0.,
                     mean=np.zeros(n),grand=0.)
        numeric['intercept' if prior is None else 'b']=np.linspace(-.23,.41,c)
        return mj.MarginJointState(U,np.empty((736,0)),np.empty((8,0)),np.zeros_like(U),
            np.zeros((n,8)),np.zeros(8),features,labels,support_ids,names,old,problem,
            dict(numeric=numeric,b=ab,a=aa),prior,{},dict(mode=problem.mode))
    b=state(old,{name:value[:6*k] for name,value in raw.items()},ids[:6*k],None,np.random.SeedSequence([PLAN['model_seed'],0]))
    c=state(classes,raw,ids,b,np.random.SeedSequence([PLAN['model_seed'],1]))
    if c.prior is not b or c.ids[:6*k]!=b.ids or any(
            not np.array_equal(c.raw[name][:6*k],b.raw[name]) for name in mj._NAMES):
        raise RuntimeError('Synthetic old support binding failed')
    return b,c


def assert_bitwise_equal(np,left,right):
    if len(left)!=len(right):raise RuntimeError('SCORE_EQUIVALENCE_FAILED: output count')
    for a,b in zip(left,right):
        if a.dtype!=np.float64 or b.dtype!=np.float64 or a.shape!=b.shape or not np.array_equal(a.view(np.uint64),b.view(np.uint64)):
            raise RuntimeError('SCORE_EQUIVALENCE_FAILED: full float64 score bits differ')


def measure_arm(b,c,queries,*,case,optional):
    if case not in ('PAIR','C_ONLY'):raise ValueError('Unknown timing case')
    totals={key:0 for key in WORK_KEYS}
    totals.update(public_score_call_count=0,residual_score_call_count=0,prior_residual_score_call_count=0,
                  reused_prior_count=0,composition_addition_count=0,score_physical_count=0)
    overhead={key:[] for key in EXTRA_SECONDS};outputs=[]
    start=time.perf_counter()
    for query in queries:
        if optional:
            bs,ba,packet=b.score_single_for_reuse(**query)
            cs,ca=c.score_single_with_reused_prior(packet,**query)
            audits=(ba,ca);totals['reused_prior_count']+=int(ca['prior_reused'])
            totals['public_score_call_count']+=2
        elif case=='PAIR':
            bs,ba=b.score_with_audit(**query);cs,ca=c.score_with_audit(**query)
            audits=(ba,ca);totals['public_score_call_count']+=2
        else:
            cs,ca=c.score_with_audit(**query);audits=(ca,)
            totals['public_score_call_count']+=1
        totals['composition_addition_count']+=len(b.classes)
        if case=='PAIR':outputs.append(bs)
        outputs.append(cs)
        for audit in audits:
            totals['residual_score_call_count']+=int(bool(audit['residual']))+int(bool(audit['prior']))
            totals['prior_residual_score_call_count']+=int(bool(audit['prior']))
            # Top-level work already includes nested residual/prior; never sum
            # those nested copies or add reference-subset work to raw totals.
            for key in WORK_KEYS:totals[key]+=int(audit.get(key,0))
            totals['score_physical_count']+=int(audit['score_physical_count'])
            for key in EXTRA_SECONDS:
                if key in audit:overhead[key].append(float(audit[key]))
    seconds=time.perf_counter()-start
    return outputs,dict(seconds=seconds,work=totals,overhead_seconds={key:dict(
        sum=sum(values) if values else None,max=max(values) if values else None,
        measured_call_count=len(values),reason=None if values else 'API_NOT_EXECUTED') for key,values in overhead.items()})


def measure_case(np,b,c,queries,case,*,warmups=1,repetitions=3,record=None,checkpoint=None):
    result=record if record is not None else dict(case=case,samples=[])
    for _ in range(warmups):
        original,_=measure_arm(b,c,queries,case=case,optional=False)
        reused,_=measure_arm(b,c,queries,case=case,optional=True)
        assert_bitwise_equal(np,original,reused)
    result['warmup_complete']=True
    for repeat in range(repetitions):
        arms={};scores={};order=('original','reused') if repeat%2==0 else ('reused','original')
        for name in order:
            scores[name],arms[name]=measure_arm(b,c,queries,case=case,optional=name=='reused')
        assert_bitwise_equal(np,scores['original'],scores['reused'])
        result['samples'].append(dict(repeat=repeat,order=list(order),bitwise_equal=True,**arms))
        if checkpoint:checkpoint()
    medians={name:statistics.median(row[name]['seconds'] for row in result['samples']) for name in ('original','reused')}
    result.update(status='COMPLETE',median_seconds=medians,
        reused_over_original_ratio=medians['reused']/medians['original'],
        paired_reused_over_original_ratios=[row['reused']['seconds']/row['original']['seconds'] for row in result['samples']])
    return result


def _write_json(stream,value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    stream.seek(0);stream.write(text);stream.truncate();stream.flush()


def run(output_root,*,run_id,code_commit):
    # Exclusive directory creation precedes imports. Only our new files can be
    # updated; completed rows survive a later technical failure.
    run_start=time.perf_counter()
    started_at_utc=datetime.now(timezone.utc).isoformat()
    path=Path(output_root)
    if not path.is_absolute():raise ValueError('output-root must be absolute')
    if run_id!=RUN_ID:raise ValueError('run-id differs from the fixed synthetic preregistration')
    if not re.fullmatch('[0-9a-fA-F]{40}',code_commit):raise ValueError('code-commit must be the actual full 40-hex HEAD supplied by launch owner')
    path.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',synthetic_only=True,plan=dict(PLAN),cases=[],
        run_id=run_id,code_commit=code_commit,group_id='d92-margin-single-query-software-cost',
        additional_external_payload_bytes=0,
        caveats=['Scoring expansions are synthetic, not fitted models or accuracy evidence.',
                 'C_ONLY optional timing includes the necessary B score; only PAIR removes redundant B work.',
                 'Reference distance counters are subsets of raw distance work, not additive.',
                 'Peak working set is process-wide including import/warmup/all cases, never a per-case peak.',
                 'No measured satellite benefit or performance improvement is implied.'])
    startup=dict(run_id=run_id,code_commit=code_commit,configuration=dict(PLAN),synthetic_only=True,
                 pid=os.getpid(),python=sys.executable,cwd=os.getcwd(),argv=list(sys.argv),
                 source_file=str(Path(__file__).resolve()),started_at_utc=started_at_utc,
                 input_source='All support and single-sample input values constructed in memory; no external data files.',
                 thread_environment={key:os.environ.get(key) for key in THREAD_ENV},
                 scope='CPU software scoring cost; not a replacement for the independent real benchmark',
                 timing_scope='Sequential API calls plus identical local audit aggregation; no fitting or score equality check inside timing.')
    with (path/'startup.json').open('x',encoding='utf-8') as handle:json.dump(startup,handle,ensure_ascii=False,allow_nan=False,indent=2)
    with (path/'summary.json').open('x+',encoding='utf-8',newline='\n') as stream, \
         (path/'measurements.jsonl').open('x',encoding='utf-8',newline='\n') as compact, \
         (path/'measurements.csv').open('x',encoding='utf-8',newline='') as csvfile, \
         (path/'measurement.log').open('x',encoding='utf-8',newline='\n') as log:
        writer=None
        log.write('SYNTHETIC_ONLY\n'+json.dumps(startup,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
        def save():
            report['process_lifetime_peak_working_set']=process_peak_working_set()
            report['measurement_elapsed_seconds']=time.perf_counter()-run_start
            _write_json(stream,report)
        def log_sample(row,sample):
            nonlocal writer
            scalar=dict(row_id=row['row_id'],k=row['k'],case=row['case'],repeat=sample['repeat'],
                        order='/'.join(sample['order']),bitwise_equal=sample['bitwise_equal'])
            for arm in ('original','reused'):
                scalar[arm+'_seconds']=sample[arm]['seconds']
                scalar.update({arm+'_'+key:value for key,value in sample[arm]['work'].items()})
                for key,stats in sample[arm]['overhead_seconds'].items():
                    scalar.update({arm+'_'+key+'_'+name:value for name,value in stats.items()})
            compact.write(json.dumps(scalar,ensure_ascii=False,allow_nan=False)+'\n');compact.flush()
            if writer is None:writer=csv.DictWriter(csvfile,fieldnames=list(scalar));writer.writeheader()
            writer.writerow(scalar);csvfile.flush()
            log.write(json.dumps(scalar,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
        save()
        try:
            np,mj,pools=load_runtime();report['hardware']=hardware_metadata(pools)
            raw=make_features(np,mj,PLAN['query_count'],PLAN['evaluation_seed'])
            queries=[{key:value[i:i+1] for key,value in raw.items()} for i in range(PLAN['query_count'])]
            for k in PLAN['ks']:
                b,c=build_states(np,mj,k)
                for case in ('PAIR','C_ONLY'):
                    row=dict(k=k,case=case,old_support_count=6*k,registered_support_count=26*k,
                             row_id='k'+str(k)+'-'+case.lower(),classes=26,status='RUNNING',samples=[])
                    report['cases'].append(row);save()
                    rowdir=path/row['row_id'];rowdir.mkdir()
                    with (rowdir/'row.json').open('x+',encoding='utf-8',newline='\n') as rowfile:
                        def checkpoint():
                            log_sample(row,row['samples'][-1]);_write_json(rowfile,row);save()
                        _write_json(rowfile,row)
                        try:
                            measure_case(np,b,c,queries,case,warmups=PLAN['warmup_repetitions'],
                                         repetitions=PLAN['timed_repetitions'],record=row,checkpoint=checkpoint)
                        except Exception:
                            row['status']='TECHNICAL_FAILURE';_write_json(rowfile,row);raise
                        _write_json(rowfile,row)
                    save()
            report['status']='COMPLETE_SYNTHETIC_SOFTWARE_MEASUREMENT';save()
            with (path/'complete.json').open('x',encoding='utf-8') as handle:
                json.dump(dict(status=report['status'],run_id=run_id,code_commit=code_commit,
                               ended_at_utc=datetime.now(timezone.utc).isoformat(),
                               completed_row_count=len(report['cases'])),handle,allow_nan=False)
            log.write(report['status']+'\n');return 0
        except Exception as exc:
            report.update(status='TECHNICAL_FAILURE',error=dict(type=type(exc).__name__,message=str(exc)))
            save()
            with (path/'failed.json').open('x',encoding='utf-8') as handle:
                json.dump(dict(status=report['status'],error=report['error'],
                               completed_rows=[row['row_id'] for row in report['cases'] if row['status']=='COMPLETE']),handle,allow_nan=False)
            log.write(json.dumps(report['error'],allow_nan=False)+'\n');return 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--code-commit',required=True)
    args=parser.parse_args(argv)
    return run(args.output_root,run_id=args.run_id,code_commit=args.code_commit)


if __name__=='__main__':
    raise SystemExit(main())
