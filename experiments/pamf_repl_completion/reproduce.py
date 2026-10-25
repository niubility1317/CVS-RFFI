"""Pure-stdlib complete score/cost/timing replay; never opens IQ or weights."""
import argparse,csv,gzip,json,math,sqlite3,tempfile
from contextlib import closing
from pathlib import Path
from .analyze import aggregate
from .results import records,supplement,csvrows

def compare(reference,replayed,names=None):
    numeric=0;maximum=0.;rows=0
    names=names or ['per_seed.csv.gz','net_corrections.csv.gz','summary.csv.gz','restoration_gain.csv.gz','summary_overall.csv','restoration_gain_overall.csv','core_effects.csv','m3_crossed_cells.csv','m3_crossed_diagonal.csv']
    for name in names:
        # Seed rows include set-derived S6 ordering; join by all nonnumeric fields
        # plus explicit numeric seed/batch/round identifiers, not line position.
        with tempfile.TemporaryDirectory() as tmp,closing(sqlite3.connect(':memory:')) as db:
            db.execute('create table data(k text primary key,v text)')
            def split(r):
                texts={};nums={}
                for k,v in r.items():
                    if k=='seed_values':
                        nums.update({k+'['+str(i)+']':float(x) for i,x in enumerate(v.split(';'))});continue
                    try:x=float(v)
                    except (ValueError,TypeError):texts[k]=v;continue
                    if k in ('seed','F_seed','G_seed','batch','round','stratum'):texts[k]=v
                    else:nums[k]=x
                return json.dumps(texts,sort_keys=True),nums
            for r in csvrows(Path(reference)/name):
                key,n=split(r);db.execute('insert into data values(?,?)',(key,json.dumps(n)))
            db.commit();count=db.execute('select count(*) from data').fetchone()[0];seen=0
            for r in csvrows(Path(replayed)/name):
                key,n=split(r);saved=db.execute('select v from data where k=?',(key,)).fetchone()
                if not saved:raise ValueError(('Missing CSV key',name,key))
                old=json.loads(saved[0])
                if old.keys()!=n.keys():raise ValueError('Numeric columns differ')
                for k,v in n.items():
                    diff=abs(v-old[k])
                    if not math.isfinite(diff) or diff>1e-10:raise ValueError((name,k,diff))
                    maximum=max(maximum,diff);numeric+=1
                seen+=1
            if seen!=count:raise ValueError('Row coverage differs')
            rows+=seen
    return dict(status='PASS',CSV_files=len(names),rows=rows,numeric_values=numeric,maximum_absolute_difference=maximum,tolerance=1e-10,byte_identical_not_claimed=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--external-seed',type=Path);p.add_argument('--external-summary',type=Path);p.add_argument('--compare',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    v=aggregate(records(a.root/'analysis/score_inputs.jsonl.gz'),a.output)
    if v['logical_rows']!=228:raise ValueError('All228 rows required')
    s=supplement(a.root,a.output,a.external_seed,a.external_summary)
    result=dict(CM=v,supplement=s)
    (a.output/'reproduction_verification.json').write_text(json.dumps(dict(result,status='CM_REPLAY_COMPLETE'),indent=2)+'\n',encoding='utf-8')
    if a.compare:result['comparison']=compare(a.root/'analysis',a.output)
    (a.output/'reproduction_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
