"""Availability-aware comparison of existing source physics records;no IQ access."""
import argparse,json
from itertools import combinations
from pathlib import Path
from . import design as d

def compare_saved(first,second):
 if first is None or second is None:
  return 'MISSING_BOTH' if first is None and second is None else ('MISSING_FIRST' if first is None else 'MISSING_SECOND')
 return 'AVAILABLE_EQUAL' if first==second else 'AVAILABLE_UNEQUAL'

def audit(inp,out):
 inp,out=Path(inp),Path(out)
 if out.exists():raise FileExistsError('Preserve existing physics comparison')
 sources={r['key']:{v:d.read(inp/r['key']/'source_readouts'/(v+'.json')).get('physics') for v in d.VIEWS[1:]} for r in d.sources()}
 coverage={k:{v:len(p) if isinstance(p,list) else 0 for v,p in views.items()} for k,views in sources.items()}
 pairs=[]
 for a,b in combinations(sources,2):
  pairs.append(dict(first=a,second=b,views={v:compare_saved(sources[a][v],sources[b][v]) for v in d.VIEWS[1:]}))
 statuses=[s for p in pairs for s in p['views'].values()]
 result=dict(status='ANALYZED',source_only=True,target_access=False,new_NN=0,physics_records_per_control_view=coverage,pairs=pairs,comparison_counts={s:statuses.count(s) for s in sorted(set(statuses))},limits='Missing records are unknown,not demonstrated channel differences. Equal configs or stored physics do not prove exact IQ waveform equality.')
 out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','physics_records_per_control_view')}));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.input,a.output)
