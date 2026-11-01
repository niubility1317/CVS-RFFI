"""Run source-only V smoke, sealed predictions and truth-last scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    scripts=Path(__file__).resolve().parent
    if (scripts.parent.parent/'release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Exact immutable release required')
    if a.run_root.exists():raise FileExistsError('Immutable run root exists')
    a.run_root.mkdir(parents=True)
    script=scripts/'phase1_source_v_restoration_diagnostic.py'
    for mode in ('smoke','predict','score'):
        cmd=[sys.executable,'-X','utf8','-u',str(script),mode,'--root',str(a.root),'--out',str(a.run_root),'--commit',a.commit]
        print('SOURCE_V_PIPELINE_EXEC',json.dumps(cmd),flush=True)
        subprocess.run(cmd,check=True)
    with (a.run_root/'completion.json').open('x',encoding='utf8') as f:json.dump(dict(status='SOURCE_V_DIAGNOSTIC_COMPLETE',target_reads=0),f);f.write('\n')
    print('SOURCE_V_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':main()
