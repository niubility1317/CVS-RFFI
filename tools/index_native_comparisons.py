"""Mirror the two authoritative new index entries without dropping history."""
import argparse
import csv
import json
from pathlib import Path

RUNS={'20260930-phase1-native-baselines-practical-manysig-m5-r01','20260930-phase12-native-baselines-practical-m5-r01'}
METHODS=('protonet','feature_separation','dadda','mrior','twostage','csil','mopc_hr','orthogonal','radionet_ada')

def update(authority, destination):
    authority,destination=Path(authority),Path(destination)
    fresh=[]
    with (authority/'catalog.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['id'] in RUNS:fresh.append(row)
    if {row['id'] for row in fresh}!=RUNS:raise ValueError('Both new registrations must be indexed first')
    path=destination/'catalog.jsonl'
    existing=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    old_count=sum(row['id'] in RUNS for row in existing)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in existing if row['id'] not in RUNS)+
                    ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in fresh),encoding='utf-8')
    with (authority/'catalog.csv').open(encoding='utf-8',newline='') as f:
        new_csv=[row for row in csv.DictReader(f) if row['id'] in RUNS]
    path=destination/'catalog.csv'
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;existing_csv=list(reader)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(row for row in existing_csv if row['id'] not in RUNS);writer.writerows(new_csv)
    heading='## Native comparison registrations 2026-09-30'
    links='\n'.join('- ['+run+'](../automation_reports/CV-SincNet/'+run+'/report.md)' for run in sorted(RUNS))
    path=destination/'README.md';text=path.read_text(encoding='utf-8')
    if heading not in text:path.write_text(text+'\n'+heading+'\n\n'+links+'\n',encoding='utf-8')
    for method in METHODS:
        path=destination/'by_method'/(method+'.md')
        text=path.read_text(encoding='utf-8') if path.exists() else '# '+method+'\n'
        if heading not in text:path.write_text(text+'\n'+heading+'\n\n'+links.replace('(../','(../../')+'\n',encoding='utf-8')
    path=destination/'coverage.json';obj=json.loads(path.read_text(encoding='utf-8'))
    obj['record_counts']['managed_run']+=len(fresh)-old_count
    obj['native_comparison_registration_refs']=sorted(RUNS)
    obj['native_comparison_authority']=str(authority)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--authority',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();update(a.authority,a.destination)
