"""One-owner Phase1 source training, fixed selection, truth-last test chain."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    scripts=Path(__file__).resolve().parent
    marker=scripts.parent.parent/'release_commit.txt'
    if marker.read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Exact immutable release required')
    if args.run_root.exists():raise FileExistsError('Immutable experiment root already exists')
    args.run_root.mkdir(parents=True)
    base=[sys.executable,'-X','utf8','-u']
    def execute(script,*extra,environment=None):
        command=base+[str(scripts/script),*map(str,extra)]
        print('PIPELINE_EXEC',json.dumps(command),flush=True)
        subprocess.run(command,check=True,env=environment)
    source=scripts/'phase1_complex_multires_source.py'
    execute(source,'--root',args.root,'--out',args.run_root/'source_smoke','--commit',args.commit,'--device',args.device,'--smoke')
    execute(source,'--root',args.root,'--out',args.run_root/'source','--commit',args.commit,'--device',args.device)
    with (args.run_root/'source'/'source_selection.json').open(encoding='utf8') as f:selection=json.load(f)
    if selection['selected']['kind']=='dura':
        result=dict(status='SOURCE_SELECTION_RETAINS_DURA',source_selection=selection['selected'],target_predictions=False)
    else:
        predictor=scripts/'phase1_complex_multires_predict.py';out=args.run_root/'source'
        execute(predictor,'smoke','--root',args.root,'--out',out,'--commit',args.commit,'--device',args.device)
        execute(predictor,'predict','--root',args.root,'--out',out,'--commit',args.commit,'--device',args.device)
        cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        execute(predictor,'score','--root',args.root,'--out',out,'--commit',args.commit,'--device','cpu',environment=cpu_env)
        execute(scripts/'phase1_complex_multires_compare.py','--root',args.root,'--out',out,'--commit',args.commit,environment=cpu_env)
        result=dict(status='PHASE1_PREDICTIONS_AND_SCORE_COMPLETE',source_selection=selection['selected'],target_predictions=True)
    with (args.run_root/'completion.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('PIPELINE_COMPLETION',json.dumps(result),flush=True)


if __name__=='__main__':main()
