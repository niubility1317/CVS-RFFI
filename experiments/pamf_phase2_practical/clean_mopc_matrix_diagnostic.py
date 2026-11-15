"""Bounded support-only MoPC stability matrix; never produces predictions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .clean_scheduler import choose
from .io import atomic, read, require, write
from .runner import occupancy


def cases(config):
    result=[];seen=set()
    for source in config['sources']:
        require(source['group'] in ('native_cil','hybrid'), 'Unsupported diagnostic group')
        job=read(source['job']);profile=job['profile']
        require((source['group']=='native_cil' and profile=='mopc_hr') or
                (source['group']=='hybrid' and job.get('methods')==['mopc_hr']), 'Wrong diagnostic method')
        for k,n in config['points']:
            require([k,n] in job['points'],'Diagnostic point absent from original job')
            key=(source['group'],profile,k,n);require(key not in seen,'Duplicate diagnostic case');seen.add(key)
            result.append(dict(id=f"{source['group']}-{profile}-K{k}-N{n}",group=source['group'],
                               job=source['job'],runtime=source['runtime'],k=k,n=n))
    require(result,'Empty diagnostic matrix')
    return result


def diagnostic_result(path, code):
    require(code==0,'Diagnostic process failed')
    records=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    results=[r for r in records if r.get('event')=='DIAGNOSTIC_RESULT']
    require(len(results)==1 and results[0].get('status') in ('COMPLETE','NONFINITE') and
        results[0].get('query_read') is False and results[0].get('truth_read') is False and
        results[0].get('predictions_written') is False,'Wrong diagnostic result')
    return results[0]


def run(config, output):
    pending=cases(config);root=Path(output).resolve();require(not root.exists(),'Diagnostic root exists')
    root.mkdir(parents=True);(root/'logs').mkdir()
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),jobs=len(pending),query_read=False,truth_read=False))
    active={};finished=[];failed=[];fatal=None
    while pending or active:
        for key,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['stream'].close()
            try:
                result=diagnostic_result(item['path'],code)
                if result['status']!='COMPLETE':failed.append(dict(id=key,result=result))
                else:finished.append(dict(id=key,gpu=item['gpu'],result=result))
            except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
                failed.append(dict(id=key,error=str(exc)))
            del active[key]
        if not fatal:
            try:
                admitted=choose(pending,{k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in active.items()},
                    occupancy(),[5,6,7],6000,7,worker_reserve=3000,one_per_snapshot=True)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                fatal=str(exc);admitted=[];pending.clear()
            for case,gpu in admitted:
                module='clean_mopc_native_support_diagnostic' if case['group']=='native_cil' else 'clean_mopc_savedb_diagnostic'
                cmd=[sys.executable,'-u','-m',__package__+'.'+module,'--runtime',case['runtime'],'--job',case['job'],
                     '--k',str(case['k']),'--n',str(case['n']),'--device','cuda:0']
                if case['group']=='hybrid':cmd+=['--variant','clip1000_uninstrumented']
                path=root/'logs'/(case['id']+'.jsonl');stream=None
                try:
                    stream=path.open('x',encoding='utf-8')
                    process=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT)
                except OSError as exc:
                    if stream:stream.close()
                    fatal=str(exc);pending.clear();break
                active[case['id']]=dict(process=process,stream=stream,path=path,gpu=gpu);pending.remove(case)
        atomic(root/'state.json',dict(status='DRAINING' if fatal else 'RUNNING',pending=len(pending),
            finished=len(finished),failed=failed,fatal=fatal,active=[dict(id=k,pid=v['process'].pid,gpu=v['gpu'])
            for k,v in active.items()],query_read=False,truth_read=False))
        if pending or active:time.sleep(1)
    status='FAILED' if fatal or failed else 'COMPLETE'
    write(root/'complete.json',dict(status=status,finished=finished,failed=failed,fatal=fatal,
          query_read=False,truth_read=False,predictions_written=False))
    atomic(root/'state.json',dict(status=status,pending=0,active=[],finished=len(finished),failed=failed,fatal=fatal,
          query_read=False,truth_read=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read(a.config),a.output)


if __name__=='__main__':main()
