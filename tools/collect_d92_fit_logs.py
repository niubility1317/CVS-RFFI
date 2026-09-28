"""Collect completed fit diagnostics and provide compact CSV without query arrays."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

from read_d92_run import FLAGS
from run_d92_confirmation import candidate_definition


def fit_log_paths(confirmation):
    candidate=candidate_definition(confirmation)['candidate_folder']
    return [('compact.jsonl',candidate+'/compact.jsonl'),('d92.jsonl','d92.log')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_text(encoding='utf-8'));remote=spec['execution']['remote_run_root']
    paths=fit_log_paths(spec['confirmation'])
    probe='import json\nfrom pathlib import Path\nr=Path('+repr(remote)+')\nassert json.loads((r/"complete.json").read_text())["status"]=="SCORED"\n'
    subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=probe.encode(),check=True)
    out=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'results/fit_logs'
    out.mkdir(parents=True,exist_ok=False)
    for row in spec['rows']:
        rid=row['row_id'];folder=out/rid;folder.mkdir()
        for name,rel in paths:
            origin=row.get('reuse_row_root',row['output_root']) if name=='d92.jsonl' else row['output_root']
            subprocess.run(['scp',*FLAGS,'N607:'+origin+'/'+rel,str(folder/name)],check=True)
        entries=[json.loads(line) for line in (folder/'compact.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(entries)!=spec['confirmation']['splits_per_model']:raise ValueError('Incomplete fit diagnostics')
        small=[]
        for r in entries:
            if isinstance(r.get('selected'),dict):
                selected=r.pop('selected');r.update({'selected_'+k:v for k,v in selected.items()})
            small.append(r)
        with (folder/'compact.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(small[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in small)
    print(out)


if __name__=='__main__':main()
