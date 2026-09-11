"""Stream an immutable read-only remote audit into local evidence files."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent
argv=['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607',
      '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -']
audit={}
with (ROOT/'collection_stderr.log').open('wb') as errors:
    proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
    proc.stdin.write((ROOT/'audit_remote.py').read_bytes()); proc.stdin.close()
    for raw in proc.stdout:
        record=json.loads(raw)
        v=record['variant']; out=ROOT/v; out.mkdir(exist_ok=True)
        for name,text in record['files'].items(): (out/name).write_text(text,encoding='utf-8')
        audit[v]=record['audit']
        (ROOT/'prediction_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        print(v,'VERIFIED',audit[v]['prediction_rows'],'predictions',audit[v]['compared_score_groups'],'groups',flush=True)
    code=proc.wait()
assert code==0, (code,(ROOT/'collection_stderr.log').read_text(encoding='utf-8'))
assert len(audit)==8
print('ALL_EIGHT_AUDITED',flush=True)
