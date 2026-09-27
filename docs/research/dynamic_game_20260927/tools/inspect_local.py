from pathlib import Path
import json, csv, gzip, subprocess
ROOT=Path('E:/type10-7')
OUT=ROOT/'analysis/dynamic_game_audit_20260927'
OUT.mkdir(parents=True,exist_ok=True)
for rel in ['automation_reports/CV-SincNet/response_matrix_eval_20260917/inventory.json','automation_reports/CV-SincNet/response_matrix_eval_20260917/experiment.json','automation_reports/CV-SincNet/phase1_daot_rc4_pure_game_m3_20260917_r2/experiment.json']:
 d=json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
 print(rel, 'keys',list(d)[:25])
 if 'rows' in d: print('row0',str(d['rows'][0])[:1800])
for rel in ['automation_reports/CV-SincNet/response_matrix_eval_20260917/results/summary.csv','automation_reports/CV-SincNet/response_matrix_eval_20260917/results/source_curves.csv']:
 with (ROOT/rel).open(encoding='utf-8-sig',newline='') as f:
  r=list(csv.DictReader(f))
 print(rel,'count',len(r),'row0',r[0])
remote="""from pathlib import Path
import json,datetime
r=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
names=[p.name for p in r.iterdir() if p.is_dir() and any(t in p.name.lower() for t in ['game','response','xuc','native_dr','core90_eg'])]
out={'read_at':datetime.datetime.now().astimezone().isoformat(),'runs':names,'details':{}}
for n in ['phase1_native_response_matrix_20260914_r1','phase1_daot_rc4_pure_game_m3_20260917_r2','phase1_adv3b02_xuc_full_s392005_20260913_r1']:
 p=r/n
 out['details'][n]=[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size} for f in p.rglob('*') if f.is_file() and f.suffix in ['.json','.jsonl','.csv','.log']]
print(json.dumps(out))
"""
p=subprocess.run(['ssh','-F',str(ROOT/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=remote.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace'))
d=json.loads(p.stdout)
(OUT/'remote_inventory.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print('remote runs',d['runs'])
for k,v in d['details'].items():print(k,'files',len(v),'first',v[:12],'bytes',sum(x['bytes'] for x in v))
