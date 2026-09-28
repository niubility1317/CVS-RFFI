"""Collect completed fit diagnostics and provide compact CSV without query arrays."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

from read_d92_run import FLAGS


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_text(encoding='utf-8'));remote=spec['execution']['remote_run_root']
    probe='import json\nfrom pathlib import Path\nr=Path('+repr(remote)+')\nassert json.loads((r/"complete.json").read_text())["status"]=="SCORED"\n'
    subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=probe.encode(),check=True)
    out=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'results/fit_logs'
    out.mkdir(parents=True,exist_ok=False)
    for row in spec['rows']:
        rid=row['row_id'];folder=out/rid;folder.mkdir()
        for name,rel in [('compact.jsonl','scv/compact.jsonl'),('d92.jsonl','d92.log')]:
            subprocess.run(['scp',*FLAGS,'N607:'+row['output_root']+'/'+rel,str(folder/name)],check=True)
        entries=[json.loads(line) for line in (folder/'compact.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(entries)!=spec['confirmation']['splits_per_model']:raise ValueError('Incomplete fit diagnostics')
        small=[]
        for r in entries:
            selected=r.pop('selected');r.update({'selected_'+k:v for k,v in selected.items()});small.append(r)
        with (folder/'compact.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(small[0]));writer.writeheader();writer.writerows(small)
    print(out)


if __name__=='__main__':main()
