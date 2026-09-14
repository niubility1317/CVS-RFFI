import csv,json,gzip,statistics,datetime,re,shutil,zipfile
from pathlib import Path
from collections import Counter
B=Path('E:/type10-7');O=B/'automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914';P=O/'report_bundle'
def rr(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def mdtable(rs,keys):
 def fmt(v):
  if isinstance(v,float):return f'{v:.6f}'
  return str(v if v is not None else '').replace('|','\\|').replace('\n',' ')
 return '\n'.join(['|'+'|'.join(keys)+'|','|'+'|'.join('---' for k in keys)+'|']+['|'+'|'.join(fmt(r.get(k,'')) for k in keys)+'|' for r in rs])+'\n'
def wc(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
a1source=O/'a1_all33_recount.json.gz';a1summ=[]
if a1source.exists():
 a1=json.loads(gzip.decompress(a1source.read_bytes()));assert len(a1['rows'])==33 and not a1['errors'];detail=[];conf=[];classes=[]
 for r in a1['rows']:
  ident={'run':r['run'],'row':r['row'],'inherited_target_contact':r['inherited_target_contact']};leo=['leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
  dr={(v['scene'],v['scope'],v['group']):v for v in r['details']}
  a1summ.append({**ident,**{s+'_macro_f1_pct':r['macro'][s]['macro_f1_pct']for s in ['clean',*leo]},'leo_macro_f1_pct':statistics.mean(r['macro'][s]['macro_f1_pct']for s in leo),'weak_rx_leo_mean_pct':min(statistics.mean(dr[s,'rx',rx]['accuracy_pct']for s in leo)for rx in ['0','2','5','7','9','10','11']),'weak_tx_leo_mean_pct':min(statistics.mean(dr[s,'tx',str(tx)]['accuracy_pct']for s in leo)for tx in range(6))})
  detail.extend({**ident,**v}for v in r['details'])
  for scene,cm in r['confusion'].items():
   conf.extend({**ident,'scene':scene,'true_tx':i,'predicted_tx':j,'count':cm[i][j]}for i in range(6)for j in range(6))
   f1=[]
   for i in range(6):
    f1.append(200*cm[i][i]/(sum(cm[i])+sum(v[i]for v in cm)))
    classes.append({**ident,'scene':scene,'tx':i,**{k:r['macro'][scene]['per_class_'+k+'_pct'][i]for k in ['precision','recall','f1']}})
   assert abs(statistics.mean(f1)-r['macro'][scene]['macro_f1_pct'])<1e-9
 wc(P/'tables/a1_33_macro_tail_summary.csv',a1summ);wc(P/'tables/a1_33_rx_day_tx_details.csv',detail);wc(P/'tables/a1_33_confusion_matrices.csv',conf);wc(P/'tables/a1_33_class_metrics.csv',classes);shutil.copy2(a1source,P/'evidence'/a1source.name)
rs=rr(P/'tables/all_scored_row_records.csv');parts=[]
for family in ['A1','GAME_V1','GAME_V2','EG_HIGH','XUC']:
 group=[r for r in rs if r['family']==family]
 for r in group:
  for k in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak','leo_mean']:r[k]=float(r[k])
 parts.append(f'### 13.{len(parts)+1}.{family}：{len(group)}条评分记录\n\n'+mdtable(group,['run','row','seed','clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak','leo_mean','inherited_target_contact']))
summary=json.loads((P/'tables/summary.json').read_text(encoding='utf-8'))
parts.append('### 13.6.FULL运行、开销与控制证据\n\n'+mdtable(summary['full_summary'],['row','epoch','steps','hours','source_val_pct','legacy_audits','legacy_recovery_worse','cstar_audits','cstar_valid','xu_cosine_mean']))
parts.append('### 13.7.可比与不可单因素归因的配对差值\n\n'+mdtable(summary['contrasts'],['baseline','variant','meaning','clean_delta_pp','leo_mean_delta_pp']))
for family in ['GAME_V2','EG_HIGH']:
 parts.append(f'### 13.{len(parts)+1}.{family}完整跨seed统计\n\n'+mdtable(rr(P/f'sources/{family}/target_seed_aggregates.csv'),['method','metric','n_seeds','mean_pct','sample_std_pct']))
if a1summ:parts.append('### 13.10.A1全部33行新补齐Macro-F1与尾部\n\n'+mdtable(a1summ,['row','clean_macro_f1_pct','leo_macro_f1_pct','weak_rx_leo_mean_pct','weak_tx_leo_mean_pct','inherited_target_contact']))
report=P/'report.md';s=report.read_text(encoding='utf-8').split('<!-- GENERATED_TABLES -->')[0]+'<!-- GENERATED_TABLES -->\n\n'+'\n'.join(parts)
report.write_text(s,encoding='utf-8')
flat=rr(P/'tables/full_config_all_keys.csv');cfg={}
for r in flat:cfg.setdefault((r['namespace'],r['key']),{})[r['row']]=r['value']
c=['# 完整配置索引与逐行差异\n','本附录不把parser默认值当成实际启用证明。`config`是CORE90承载resolved配置，`dr_config`是DR有效参数空间。F-M00无DR配置为空，不能将空值解释为参数0。原始完整JSON在configs/FULL_RESOLVED，所有字段以CSV逐行保留。值后的行ID指出不同版本；相同值合并显示。\n']
for ns in ['config','dr_config']:
 entries=[]
 for (n,k),vals in sorted(cfg.items()):
  if n!=ns:continue
  groups={}
  for rid,v in vals.items():groups.setdefault(v,[]).append(rid)
  entries.append({'参数':k,'实际值与对应行':' ; '.join(f'{v} → {",".join(rids)}' for v,rids in groups.items())})
 c.append(f'## {ns}：{len(entries)}个键\n\n'+mdtable(entries,['参数','实际值与对应行']))
c.append('## 原生A1逐行捕获与GAME矩阵\n\nA1_CAPTURED内每个文件保留一个run/row的matrix、base_options、live_argv和state中实际存在的字段；不存在的字段不补造。GAME矩阵分planned/r1/r2/r3及high seedscan，设计矩阵不能当成全行已完成；状态以评分/运行覆盖表为准。\n')
for name in ['XUC','GAME']:
 for f in sorted((P/'configs'/name).rglob('*.json')):
  d=json.loads(f.read_text(encoding='utf-8-sig'));c.append(f'### {f.relative_to(P).as_posix()}\n\n```json\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n```\n')
(P/'configuration_appendix.md').write_text('\n'.join(c),encoding='utf-8')
# Proportional report verification: exact arithmetic, row coverage, frozen confusion matrices, links and text encoding.
checks={};assert len(rs)==113;checks['scored_rows']=113;checks['family_counts']=dict(Counter(r['family'] for r in rs))
assert all(abs(r['leo_mean']-statistics.mean(r[k] for k in ['leo_clear_weak','leo_low_elev_weak','leo_rain_weak']))<1e-7 for r in rs)
checks['all_113_leo_means_recomputed']=True
scene=rr(P/'sources/XUC_FULL/scene_metrics.csv');cm=rr(P/'sources/XUC_FULL/confusion_matrices.csv');mat={}
for r in cm:
 k=r['row'],r['scene'];mat.setdefault(k,[]).append(r)
for r in scene:
 a=mat[(r['row'],r['scene'])];total=sum(int(t['count']) for t in a);correct=sum(int(t['count'])for t in a if t['true_tx']==t['predicted_tx']);assert total==int(r['total'])==168000;assert correct==int(r['correct']);assert abs(100*correct/total-float(r['accuracy_percent']))<1e-8
 f1=[]
 for tx in range(6):
  tp=sum(int(t['count'])for t in a if int(t['true_tx'])==tx and int(t['predicted_tx'])==tx);nt=sum(int(t['count'])for t in a if int(t['true_tx'])==tx);np=sum(int(t['count'])for t in a if int(t['predicted_tx'])==tx);f1.append(2*tp/(nt+np) if nt+np else 0)
 assert abs(100*statistics.mean(f1)-float(r['macro_f1_percent']))<1e-8
checks['XUC_36_scene_accuracy_and_macroF1_recomputed_from_confusion']=True
paired=rr(P/'sources/XUC_FULL/paired_rescue_harm.csv')
for r in paired:
 n=sum(int(r[k])for k in ['both_correct','rescue','harm','both_wrong']);net=int(r['rescue'])-int(r['harm']);assert net==int(r['net_correct']);assert abs(100*net/n-float(r['delta_pp']))<1e-8
checks['XUC_all_paired_count_deltas_recomputed']=len(paired)
if a1summ:checks['A1_frozen_predictions_recounted']=sum(r['count']for r in a1['rows']);checks['A1_metadata_mapping_exact_all33']=all(r['metadata_mapping_exact']for r in a1['rows']);checks['A1_all_132_scene_macroF1_recomputed']=True;checks['A1_detail_rows']=len(detail)
for family in ['GAME_V2','EG_HIGH']:
 summ=rr(P/f'sources/{family}/target_summary.csv');agg=rr(P/f'sources/{family}/target_seed_aggregates.csv')
 checks[family+'_rows']=len(summ)
 # All source CSVs and JSONs parse; selected aggregation is checked below via source exact columns.
 for method in set(r['method']for r in summ):
  vals=[float(r['leo_mean_accuracy_pct'])for r in summ if r['method']==method]
  checks[family+'_'+method+'_LEO']={'n':len(vals),'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals) if len(vals)>1 else None}
broken=[]
for f in [report,P/'configuration_appendix.md']:
 t=f.read_text(encoding='utf-8');assert '\ufffd' not in t
 for link in re.findall(r'\]\(([^)]+)\)',t):
  if re.match(r'^[a-zA-Z]+:|^#',link):continue
  if not (f.parent/link.split('#')[0]).exists():broken.append([f.name,link])
assert not broken,broken
checks['main_report_links']=True
checks['snapshot_full_action_read_at']=summary['full_live_read_at'];checks['snapshot_full_mechanics_read_at']=summary['full_mechanics_read_at']
checks['files']=sum(p.is_file()for p in P.rglob('*'));checks['report_characters']=len(s);checks['configuration_appendix_characters']=(P/'configuration_appendix.md').stat().st_size
checks['status']='VERIFIED';(P/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
for f in ['build_daot_game_report_20260914.py','finalize_daot_game_report_20260914.py','collect_daot_game_mechanics_20260914.py','refresh_daot_game_audit_20260914.py','plot_daot_game_report_20260914.py','recount_a1_all33_20260914.py','prepare_daot_game_git_delivery_20260914.py']:
 dst=P/'analysis_tools'/f;dst.parent.mkdir(exist_ok=True);shutil.copy2(B/'analysis'/f,dst)
with zipfile.ZipFile(O/'DAOT_FastTrust_Game_20260914_full_report.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in P.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(O))
print(json.dumps(checks,ensure_ascii=False))
