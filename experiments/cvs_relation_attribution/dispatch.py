"""One owner, immutable output paths; no intervention in source workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main(path):
    spec=json.loads(Path(path).read_text());release=Path(__file__).resolve().parent
    from run import validate_config
    expected={(v,s) for v in ('relation_frequency_energy','mirror_energy','mirror_subspace') for s in range(2026092701,2026092705)}
    if len(spec['rows'])!=12 or {(r['variant'],r['model_seed']) for r in spec['rows']}!=expected:raise ValueError('Incomplete diagnostic matrix')
    for row in spec['rows']:validate_config(row)
    project='/home/szu2070436088/2510044040/CV-SincNet'
    run='20261003-diagnostic-cvs-relation-attribution-source-manysig-m12-r01'
    if spec['runtime_root']!=project+'/runs/'+run or spec['log_root']!=project+'/logs/'+run:raise ValueError('Diagnostic roots differ')
    source_release=project+'/releases/cvs_mirror_subspace_identity_20261003_r01'
    sys.path[:0]=[source_release,source_release+'/code']
    from experiments.cvs_mirror_subspace_identity.prepare import PROJECT,RUN as SOURCE_RUN
    from experiments.cvs_clean_design.dispatch import choose_gpu
    source=Path(PROJECT)/'runs'/SOURCE_RUN
    state=json.loads((source/'pipeline_state.json').read_text())
    if not state['status'].startswith('SOURCE_RESEARCH_COMPLETE') or len(state['rows'])!=8 or any(r['status']!='SOURCE_TRAINED' for r in state['rows'].values()):raise ValueError('Wait for all frozen E200 source runs')
    if not (source/'source_selection.json').exists():raise ValueError('Source selection must already be frozen')
    root=Path(spec['runtime_root']);root.mkdir(parents=True,exist_ok=False)
    logs=Path(spec['log_root']);logs.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',pid=os.getpid(),cwd=str(release),launch_owner=spec['launch_owner'],rows={})
    def save(): (root/'pipeline_state.json').write_text(json.dumps(result,indent=2)+'\n')
    pending=list(spec['rows']);active={};save()
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid]
            entry=result['rows'][rid];done=Path(entry['output_root'])/'completion.json'
            entry.update(exit_code=code,status='COMPLETE' if code==0 and done.exists() else 'FAILED')
            save()
        if pending:
            gpu=choose_gpu(active)
            if gpu is not None:
                cfg=pending.pop(0);rid=cfg['variant']+'-s'+str(cfg['model_seed'])
                cfgpath=root/(rid+'.json');cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
                logfile=logs/(rid+'.log');stream=logfile.open('x')
                cmd=[sys.executable,'-u',str(release/'run.py'),'--config',str(cfgpath)]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                active[rid]=dict(process=child,gpu=gpu,log=stream)
                result['rows'][rid]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=cmd,cwd=str(release),log=str(logfile),output_root=cfg['output_root'])
                save()
        if pending or active:time.sleep(5)
    result['status']='COMPLETE' if all(r['status']=='COMPLETE' for r in result['rows'].values()) else 'FAILED_PARTIAL';save()
    return 0 if result['status']=='COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);raise SystemExit(main(p.parse_args().spec))
