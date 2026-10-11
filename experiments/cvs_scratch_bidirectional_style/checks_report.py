"""Bounded source-diagnostic report aggregation from actual synthetic readouts."""
import argparse,json
from pathlib import Path
from . import design as d
from .analyze_source import _style_add,_style_finish,_style_search,_add,_finish
from .style_report import lines

def run(path):
    evidence=d.read(path);rows=[]
    for pool,value in evidence['pools'].items():
        c=d.config(next(r for r in d.rows() if r['arm']=='DELAY_POS'))
        c['multilevel']['style_pool']=pool
        c['batch_size']=32
        # Controlled branch fixtures contain native forward/backward diagnostics.
        stats={};layer={}
        records=value['records']
        for record in records:
            style=record['details']
            metrics=_style_search(style,c,1333,1.)
            for k,v in metrics.items():_add(stats,k,v)
            for k,v in style['levels'].items():_style_add(layer.setdefault(k,{}),v,style['batch_occurrences'])
        rows.append(dict(row_id=pool,style_search_summary=_finish(stats),L_style={k:_style_finish(v) for k,v in layer.items()}))
    report='\n'.join(lines(rows))
    assert 'POS' in report and 'BIDIR' in report and 'log-scale' in report
    return dict(status='PASS',rows=len(rows),source_only_synthetic=True,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=run(a.input);d.write(a.output,v);print(json.dumps(v))
