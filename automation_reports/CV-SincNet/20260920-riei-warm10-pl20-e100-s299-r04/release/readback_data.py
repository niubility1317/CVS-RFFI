import json
from pathlib import Path
P=Path(__file__).parent
s=json.loads((P/'state.json').read_text());out=[]
for r in s['rows']:
    p=Path(r['output_root']);d={k:r[k] for k in ['run_id','pid','gpu','status']}
    for name in ['data_summary','resolved_config']:
        f=p/(name+'.json');d[name]=json.loads(f.read_text()) if f.exists() else None
    f=p/'batches.jsonl';lines=f.read_text().splitlines() if f.exists() else []
    rows=[json.loads(line) for line in lines]
    d['batches']=len(rows);d['latest']=rows[-1] if rows else None
    d['transitions']=[x for x in rows if x['batch']==0 and x['epoch'] in [1,10,11,19,20]]
    out.append(d)
print(json.dumps(out))
