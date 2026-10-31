"""Sequential fixed GPU work: train, predict all panels, then profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.pamf_identity_flow import source_train as candidate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    root=args.root;commit=args.commit;out=root/'runs'/candidate.RUN
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=commit or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Pinned release and one assigned physical GPU required')
    reference=root/'releases/20261031-phase3-fresh-dura-s2026092701-r01-release/runtime.json'
    commands=[('train',[sys.executable,'-u','-m','experiments.pamf_identity_flow.source_train','train','--root',str(root),'--reference-runtime',str(reference),'--code-commit',commit])]
    for role in ('source','target'):
        commands.append(('predict-'+role,[sys.executable,'-u','-m','experiments.pamf_identity_flow.historical_eval','predict','--root',str(root),'--commit',commit,'--role',role]))
    commands.append(('predict-phase3',[sys.executable,'-u','-m','experiments.pamf_identity_flow.phase3_eval','predict','--config',str(out/'phase3_config.json'),'--device','cuda:0']))
    commands.append(('profile',[sys.executable,'-u','-m','experiments.pamf_identity_flow.historical_eval','profile','--root',str(root),'--commit',commit]))
    for name,command in commands:
        print(json.dumps(dict(event='START',stage=name,command=command)),flush=True)
        with (out/'logs'/(name+'.log')).open('xb') as log:
            code=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        print(json.dumps(dict(event='FINISH',stage=name,exit_code=code)),flush=True)
        if code and name!='profile':raise RuntimeError(name+' failed; later scoring remains closed')


if __name__=='__main__':main()
