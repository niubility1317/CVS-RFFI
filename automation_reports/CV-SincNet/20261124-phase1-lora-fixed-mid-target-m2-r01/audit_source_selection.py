from pathlib import Path
import json
root=Path('automation_reports/CV-SincNet')
rows=[]
for f in root.glob('*lora*/full-source-scores.json'):
 p=f.parent
 if p.name == '20261124-phase1-lora-polar-shape-source-m2-r01': continue
 rs=p/'residual-source-scores.json'
 if not rs.exists(): continue
 a,b=[json.loads(x.read_text(encoding='utf-8')) for x in (f,rs)]
 if not all(x.get('n')==900 and x.get('six') and x.get('modes') for x in (a,b)): continue
 if a['modes']!=b['modes'] or a['six'][0]!=3028 or b['six'][0]!=2848: continue
 c=p/'clean-complete-scores.json'
 c=json.loads(c.read_text(encoding='utf-8')) if c.exists() else None
 for i,name in enumerate(a['modes']):
  if name in ('direct','reference'): continue
  low=[x['metrics']['practical_low_urban']['correct'][i] for x in (a,b)]
  scores=[x['six'][i] for x in (a,b)]
  diffs=[q-x['six'][1] for q,x in zip(scores,(a,b))]
  rows.append(dict(name=name,run=p.name,scores=scores,diffs=diffs,low=low,
      clean=c['summary']['correct'][i] if c and c['modes']==a['modes'] else None))

feasible = [r for r in rows if r['clean'] is not None and r['clean'] >= 8986
            and min(r['diffs']) > 0 and r['low'][0] >= 374 and r['low'][1] >= 403]
print(json.dumps({'compared': len(rows), 'feasible': feasible}, ensure_ascii=False))
