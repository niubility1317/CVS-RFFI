"""Reproduce the final private/package aggregate inputs with Python stdlib only."""
import argparse,csv,gzip,json,math
from pathlib import Path
from .analyze_full import run


def verify(expected,actual):
    expected,actual=Path(expected),Path(actual);numeric=0;maximum=0.;files=[]
    for path in sorted([*expected.glob('*.csv'),*expected.glob('*.csv.gz')]):
        other=actual/path.name
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8',newline='') as one,opener(other,'rt',encoding='utf-8',newline='') as two:
            a,b=csv.reader(one),csv.reader(two)
            if next(a)!=next(b):raise ValueError('CSV schema differs: '+path.name)
            import itertools
            count=0
            for left,right in itertools.zip_longest(a,b):
                if left is None or right is None or len(left)!=len(right):raise ValueError('Row coverage differs')
                count+=1
                for x,y in zip(left,right):
                    try:f,g=float(x),float(y)
                    except ValueError:
                        if x!=y:raise ValueError('Text differs: '+path.name)
                        continue
                    if not math.isfinite(f) or not math.isfinite(g):raise ValueError('Nonfinite result')
                    delta=abs(f-g);maximum=max(maximum,delta);numeric+=1
                    if delta>1e-10:raise ValueError('Numeric difference: '+path.name)
        files.append(dict(file=path.name,rows=count))
    proof=dict(status='REPRODUCED',CSV_files=files,numeric_values=numeric,maximum_absolute_difference=maximum,tolerance=1e-10)
    (actual/'reproduction_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--expected',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output);verify(a.expected,a.output)
