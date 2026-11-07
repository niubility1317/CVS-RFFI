"""One-owner source training followed by frozen seven-view truth-last scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.phase1_dura_orthogonal.controller import training_environments


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--ground',required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    train_env,cpu_env=training_environments('cuda:0',os.environ)
    source=args.root/'runs'/args.source
    output=args.root/'runs'/args.ground
    control=args.root/'runs'/(args.source+'-controller')
    if any(x.exists() for x in (source,output,control)):
        raise FileExistsError('Unique source, ground and controller roots required')
    control.mkdir(parents=True)
    def status(value,**extra):
        (control/'status.json').write_text(json.dumps(dict(state=value,**extra))+'\n',encoding='utf8')
    def execute(label,cmd,env):
        with (control/(label+'.log')).open('x',encoding='utf8') as log:
            subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    try:
        status('RUNNING_SOURCE_TRAINING')
        execute('source_train',[sys.executable,'-X','utf8','-u','-m',
            'experiments.phase1_dura_orthogonal.signal_gate_oof_train','--root',str(args.root),
            '--out',str(source),'--commit',args.commit],train_env)
        summary=json.loads((source/'source_summary.json').read_text(encoding='utf8'))
        if (summary.get('status')!='SOURCE_TRAINED' or summary.get('commit')!=args.commit
                or summary.get('epochs')!=20 or any(summary.get(k)!=0 for k in ('TX_labels_read','F_weights_read','F_outputs_read','V_reads','target_reads'))):
            raise ValueError('Fixed source-only final epoch required')
        fixed=['--root',str(args.root),'--out',str(output),'--source',args.source,'--commit',args.commit]
        for stage in ('smoke','predict','score'):
            status('RUNNING_GROUND_'+stage.upper())
            execute('ground_'+stage,[sys.executable,'-X','utf8','-u','-m',
                'experiments.phase1_dura_orthogonal.signal_gate_oof_ground',stage,*fixed],cpu_env)
        report=json.loads((output/'flow/test_summary.json').read_text(encoding='utf8'))
        paired=json.loads((output/'flow/paired_dura.json').read_text(encoding='utf8'))
        if (report.get('status')!='ANALYZED' or report.get('all_predictions_before_truth') is not True
                or report.get('target_feedback') is not False or paired.get('status')!='PAIRED_COMPLETE'):
            raise ValueError('Complete independent paired truth-last score required')
        status('GROUND_SCORE_COMPLETE')
    except Exception as exc:
        status('FAILED',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
