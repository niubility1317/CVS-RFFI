"""Source-only frozen pair replay, truth-last score and independent recount."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback
from experiments.phase1_dura_posterior.resources import wait_for_cpu

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261103-phase1-iterative-geometry-fullsource-s2026092701-r01'
RELEASE=ROOT/'releases'/(RUN+'-release')
CONTROL=ROOT/'runs'/(RUN+'-controller')
PYTHON='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'

def status(state,**extra):
    tmp=CONTROL/'status.tmp';tmp.write_text(json.dumps(dict(state=state,**extra)),encoding='utf8');os.replace(tmp,CONTROL/'status.json')

def environment():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
      PYTHONPATH=':'.join(map(str,(RELEASE,ROOT/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release',
                                 ROOT/'releases/pamf_stage2_inference_20261021_r01',
                                 ROOT/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'))))
    return env

def execute(module,arguments,name):
    wait_for_cpu(lambda state,**extra:status(state,phase=name,**extra))
    status('RUNNING_'+name.upper())
    with (CONTROL/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run([PYTHON,'-X','utf8','-u','-m',module,*arguments],cwd=RELEASE,env=environment(),
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(name+' exited '+str(p.returncode))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);a=ap.parse_args()
    out=ROOT/'runs'/RUN;smoke=ROOT/'runs'/(RUN+'-smoke');score=ROOT/'runs'/(RUN+'-score')
    if not CONTROL.is_dir() or any(CONTROL.iterdir()) or any(p.exists() for p in (out,smoke,score)):
        raise ValueError('Unique source replay and score paths required')
    if (RELEASE/'release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Exact release required')
    try:
        common=['--root',str(ROOT),'--commit',a.commit]
        execute('experiments.phase1_iterative_geometry_fullsource.replay',[*common,'--out',str(smoke),'--smoke'],'smoke')
        check=json.loads((smoke/'smoke.json').read_text())
        if check.get('status')!='PASS' or check.get('count')!=64 or any(check.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')):
            raise ValueError('Actual source smoke must pass')
        execute('experiments.phase1_iterative_geometry_fullsource.replay',[*common,'--out',str(out)],'replay')
        fixed=json.loads((out/'prediction_complete.json').read_text())
        if fixed.get('status')!='SOURCE_PREDICTIONS_FIXED' or fixed.get('count')!=11761 or fixed.get('pilot_replayed')!=672:
            raise ValueError('All source predictions fixed before truth required')
        execute('experiments.phase1_iterative_geometry_fullsource.score',[*common,'--source',str(out),'--out',str(score)],'score')
        execute('experiments.phase1_iterative_geometry_fullsource.audit',[*common,'--source',str(out),'--score',str(score),
                '--out',str(score/'independent_audit.json')],'audit')
        audit=json.loads((score/'independent_audit.json').read_text())
        if audit.get('status')!='PASS' or audit.get('comparisons')!=96:raise ValueError('Independent recount required')
        status('SOURCE_SCORE_AND_INDEPENDENT_AUDIT_COMPLETE',count=11761,L_s_scored=1047)
    except Exception as exc:
        status('FAILED',error=str(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
