"""Read-only source synthesis; writes a new, self-contained research report bundle."""
import csv,json,gzip,shutil,statistics,math,datetime,hashlib
from pathlib import Path
from collections import Counter
B=Path('E:/type10-7'); A=B/'automation_reports/CV-SincNet'; O=A/'daot_fasttrust_game_comprehensive_20260914'; P=O/'report_bundle'; P.mkdir(exist_ok=True)
X=B/'github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913/experiments/adv3b02_xuc'
G=B/'code/snapshots/core90_game_20260911_wt'
manifest=[]
def js(p):
 s=gzip.decompress(p.read_bytes()).decode('utf-8') if p.suffix=='.gz' else p.read_text(encoding='utf-8-sig');return json.loads(s)
def writej(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def wc(p,rs):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rs:return
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def copy(src,dst):
 dst=P/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 manifest.append({'source':str(src),'bundle_path':str(dst.relative_to(P)).replace('\\','/'),'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
families={'A1':A/'all_exploration_20260913','GAME_V2':A/'core90_game_v2_target_all21_20260913','EG_HIGH':A/'core90_eg_high_target6_20260914_r1','XUC_FULL':A/'phase1_adv3b02_xuc_full6_eval_s392005_20260914_r1','XUC_DR7':A/'phase1_adv3b02_xuc_dr_s392005_20260913_r1','BASELINE':B/'docs/CORE90_REPRODUCTION_INDEX_20260911'}
for name,root in families.items():
 for f in root.iterdir():
  if f.is_file() and f.suffix in ['.csv','.md'] and 'lora' not in f.name:copy(f,Path('sources')/name/f.name)
for f in ['full_live_audit.json.gz','full_mechanics.json.gz']:copy(O/f,Path('evidence')/f)
for name,fn in [('A1','evidence.json.gz'),('XUC_DR7','full_run_audit.json'),('XUC_DR7','grl_semantics_check.json'),('XUC_FULL','mechanism_evidence.json'),('XUC_FULL','detailed_recount.json'),('BASELINE','config_comparison.csv')]:
 src=families[name]/fn
 if src.suffix=='.json' and src.stat().st_size>2_000_000:
  dst=P/'evidence'/f'{name}_{fn}.gz';dst.write_bytes(gzip.compress(src.read_bytes()));manifest.append({'source':str(src),'bundle_path':str(dst.relative_to(P)).replace('\\','/'),'bytes':dst.stat().st_size,'transformation':'gzip of source bytes'})
 else:copy(src,Path('evidence')/f'{name}_{fn}')
for f in ['CORE90_OPTIMIZATION_REPORT_392005_20260911.md','CORE90_GAME_V2_IMPLEMENTATION_ACCEPTANCE_20260911.md']:
 if (B/'docs'/f).exists():copy(B/'docs'/f,Path('sources/GAME_V1')/f)
for f in (B/'docs/evidence').glob('core90*'):
 if f.is_file() and f.suffix in ['.csv','.json'] and f.stat().st_size<12_000_000:copy(f,Path('sources/GAME_V1')/f.name)
for f in (X/'configs').glob('*.json'):copy(f,Path('configs/XUC')/f.name)
for prefix in ['code/configs','configs','local_artifacts/core90_reaudit_392005/matrix']:
 for f in (G/prefix).rglob('matrix.json'):
  if 'core90' in str(f):copy(f,Path('configs/GAME')/f.relative_to(G))
for f in (G/'local_artifacts/core90_reaudit_392005/c2_s4_deep_20260911').glob('*/resolved_config.json'):copy(f,Path('configs/GAME/resolved_V1')/f.parent.name/f.name)
for mod in ['xuc_fusion','game_tracking','cross_response']:
 for f in (X/'code/cvsrffi'/mod).rglob('*.py'):copy(f,Path('code_snapshot/cvsrffi')/mod/f.relative_to(X/'code/cvsrffi'/mod))
for pat in ['*daot*.py','*fasttrust*.py','a1_ecrs_cross_rx.py','*rc4*.py']:
 for f in (X/'code/cvsrffi').glob(pat):copy(f,Path('code_snapshot/cvsrffi')/f.name)
for f in (X/'code/cvsrffi').glob('*.py'):
 if not (P/'code_snapshot/cvsrffi'/f.name).exists():copy(f,Path('code_snapshot/cvsrffi')/f.name)
for f in (X/'code/SSDG').glob('*.py'):copy(f,Path('code_snapshot/SSDG')/f.name)
live=js(O/'full_live_audit.json.gz'); mech=js(O/'full_mechanics.json.gz'); old=js(families['XUC_DR7']/'full_run_audit.json'); a1=js(families['A1']/'evidence.json.gz')
inv=[];configflat=[];summaries=[];epochrows=[];probes=[];auditrows=[];activation=[]
for rid,r in mech['rows'].items():
 writej(P/'configs/FULL_RESOLVED'/f'{rid}.json',{'resolved_config':r['config'],'resolved_dr_config':r['dr_config']})
 for namespace in ['config','dr_config']:
  for k,v in (r[namespace] or {}).items():configflat.append({'row':rid,'namespace':namespace,'key':k,'value':json.dumps(v,ensure_ascii=False)})
 for ep,e in r['epochs'].items():epochrows.append({'row':rid,'origin_epoch':int(ep),'actions':json.dumps(e['actions']),'scenes':json.dumps(e['scenes']),'satellite_selected_by_scene':json.dumps(e['applied']),**e['means']})
 for e in r['probes']:
  probes.append({'row':rid,'step':e['step'],'origin_epoch':e['epoch'],**{k:v for k,v in (e.get('fusion') or {}).items() if isinstance(v,(int,float,bool))},**{k:v for k,v in e['dr'].items() if isinstance(v,(int,float,bool))},**{'weighted_'+k:v for k,v in e['dr'].get('weighted_components',{}).items()}})
for family,data in [('INITIAL15',old['old']),('DR7',old['dr']),('FULL9',live['full'])]:
 for rid,r in data['rows'].items():
  logs=r.get('logs.jsonl',{}).get('records',[]);aa=r.get('action_audit',{});cnt=aa.get('counts',{});comp=r.get('completion') or {};lr=logs[-1] if logs else {}
  inv.append({'family':family,'row':rid,'training_status':r.get('state',{}).get('status'),'last_epoch':lr.get('epoch'),'action_records':cnt.get('records'),'accepted':cnt.get('accepted'),'has_completion':bool(comp),'target_score_in_pipeline':bool(r.get('score')),'score_note':'FULL6 early evaluator scores live outside the parent pipeline' if family=='FULL9' else '', 'read_at':live['read_at'] if family=='FULL9' else old.get('read_at')})
  if family!='FULL9':continue
  audits=r.get('legacy_audit.jsonl',{}).get('records',[]);ca=r.get('cstar_audit.jsonl',{}).get('records',[])
  for a in audits:auditrows.append({'row':rid,'kind':'legacy','step':a.get('step'),'valid':a.get('valid'),'online_ce':a.get('online',{}).get('ce'),'recovered_ce':a.get('recovered',{}).get('ce'),'gap':a.get('ce_gap_raw'),'G_lag':a.get('G_lag')})
  for a in ca:auditrows.append({'row':rid,'kind':'cstar','step':a.get('step'),'valid':a.get('recovery',{}).get('valid'),'reason':a.get('recovery',{}).get('reason'),'online_ce':a.get('online_ce'),'recovered_ce':a.get('recovered_ce'),'G_lag':a.get('G_lag'),'readability':a.get('readability'),'direction_imbalance':a.get('direction_imbalance'),'monitor_delta_upper95':a.get('recovery',{}).get('monitor_delta_upper95')})
  for k,v in (r.get('activation') or {}).get('components',{}).items():activation.append({'row':rid,'component':k,**v})
  epochs=mech['rows'][rid]['epochs'];applied=Counter();actions=aa.get('actions',{});allprobe=[p for p in probes if p['row']==rid]
  for ep,e in epochs.items():
   if int(ep)>=80:applied.update(e['applied'])
  summaries.append({'row':rid,'epoch':lr.get('epoch'),'steps':cnt.get('records'),'L_unique':aa.get('unique_L'),'U_unique':aa.get('unique_U'),**cnt,'actions':json.dumps(actions),'legacy_audits':len(audits),'legacy_recovery_worse':sum(a.get('ce_gap_raw',0)<0 for a in audits),'legacy_recovery_better':sum(a.get('ce_gap_raw',0)>0 for a in audits),'cstar_audits':len(ca),'cstar_valid':sum(bool(a.get('recovery',{}).get('valid')) for a in ca),'E80plus_satellite_selected':sum(applied.values()),'E80plus_satellite_by_scene':json.dumps(applied),'source_val_pct':100*lr.get('source_validation',{}).get('accuracy',float('nan')),'hours':comp.get('elapsed_seconds',0)/3600 if comp else None,'gaps':len(aa.get('gaps',[])),'bad_actions':len(aa.get('bad',[])),'xu_cosine_mean':statistics.mean([p['xu_gradient_cosine'] for p in allprobe if 'xu_gradient_cosine' in p]) if any('xu_gradient_cosine' in p for p in allprobe) else None})
for run,rr in a1['runs'].items():
 for rid,r in rr.get('rows',{}).items():
  writej(P/'configs/A1_CAPTURED'/f'{run}__{rid}.json',{k:r[k] for k in ['matrix','base_options','live_argv','state'] if k in r})
wc(P/'tables/xuc_31_coverage.csv',inv);wc(P/'tables/full_config_all_keys.csv',configflat);wc(P/'tables/full_epoch_mechanics.csv',epochrows);wc(P/'tables/full_gradient_probes.csv',probes);wc(P/'tables/full_audits.csv',auditrows);wc(P/'tables/full_component_activation.csv',activation);wc(P/'tables/full_mechanism_summary.csv',summaries)
result=[]
for family,src in [('A1',families['A1']/'all_final_results.csv'),('GAME_V1',B/'docs/evidence/core90_optimization_summary_392005.csv'),('XUC',families['XUC_FULL']/'all_28_scored_comparison.csv'),('GAME_V2',families['GAME_V2']/'target_summary.csv'),('EG_HIGH',families['EG_HIGH']/'target_summary.csv')]:
 for r in rows(src):
  z={'family':family,'run':r.get('run',r.get('run_id','')),'row':r.get('row',r.get('method','')),'seed':r.get('seed','392005'),'status':r.get('status','scored')}
  for k in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:z[k]=float(r.get(k,r.get(k+'_accuracy_pct')))
  z['leo_mean']=float(r.get('leo_mean',r.get('leo_mean_accuracy_pct')));z['source_table']=str(src)
  z['comparison_warning']='row records, not unique independent models; compare only matched conditions';z['inherited_target_contact']=r.get('inherited_target_contact','');result.append(z)
wc(P/'tables/all_scored_row_records.csv',result)
scores={r['row']:r for r in rows(families['XUC_FULL']/'all_28_scored_comparison.csv')}
contrast=[]
for a,b,meaning in [('M09','M10','native A1 plus X'),('M01','M02','X on grid'),('M01','M03','Ux on grid'),('M05','M08','C2 package on X+Ux'),('M00','M11','ordinary C2 package'),('F-A1','F-M11','full extension plus C2 vs native-loss carrier; confounded'),('F-M05','F-M08','C2 package conditional on full DR+X+Ux'),('F-M14','F-M12','Cstar enabled vs passive; both zero correction, not active-control efficacy')]:
 contrast.append({'baseline':a,'variant':b,'meaning':meaning,**{k+'_delta_pp':float(scores[b][k])-float(scores[a][k]) for k in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak','leo_mean']}})
wc(P/'tables/controlled_and_confounded_contrasts.csv',contrast)
writej(P/'tables/summary.json',{'scored_row_records':len(result),'family_counts':dict(Counter(r['family'] for r in result)),'xuc_coverage':len(inv),'full_mechanics_read_at':mech['read_at'],'full_live_read_at':live['read_at'],'full_summary':summaries,'contrasts':contrast,'A1_snapshot':a1['captured_at']})
wc(P/'source_manifest.csv',manifest)
print(json.dumps({'row_records':len(result),'families':dict(Counter(r['family'] for r in result)),'sources':len(manifest),'full_summary':summaries},ensure_ascii=False))
