"""Assemble an auditable publication bundle without changing training code."""
from pathlib import Path
import shutil,json,csv,gzip,hashlib,subprocess,statistics,collections
R=Path('E:/type10-7'); A=Path(__file__).resolve().parent
W=R/'code/snapshots/daot_game_analysis_20260914_wt'
D=W/'docs/research/dynamic_game_20260927'; D.mkdir(parents=True,exist_ok=True)
OLD=R/'automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/report_bundle'
OLDDEST=W/'docs/research/daot_fasttrust_game_20260914'
manifest=[];excluded=[]
ALLOW={'.py','.json','.jsonl','.md','.csv','.gz','.xml','.log','.sh','.txt','.yaml','.yml','.toml','.png','.pdf','.xlsx'}
def copy(src,dst,kind):
 if any(x in src.parts for x in ['__pycache__','.pytest_cache','.git']):return
 if src.suffix not in ALLOW and src.name not in ['.gitattributes','LICENSE']:
  excluded.append({'source':str(src),'reason':'generated/binary or duplicate archive outside publication selection'});return
 if src.stat().st_size>50_000_000:raise ValueError(('unexpected large file',str(src)))
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not (kind=='historical_bundle' and dst.exists()):shutil.copyfile(src,dst)
 b=dst.read_bytes();manifest.append({'kind':kind,'source':str(src),'destination':dst.relative_to(W).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
def tree(src,dst,kind):
 for f in sorted(src.rglob('*')):
  if f.is_file():copy(f,dst/f.relative_to(src),kind)
tree(OLD,OLDDEST,'historical_bundle')
N=R/'code/snapshots/native_dr_eg_prepare_20260914_wt'
tree(N/'experiments/adv3b02_xuc',D/'implementation/response','native_response_snapshot')
for f in ['analysis/response_games_plan_20260914.md','analysis/native_dr_eg_traceability_20260914.md']:
 copy(N/f,D/'implementation'/Path(f).name,'design_traceability')
for run in ['response_matrix_eval_20260917','response_matrix_release_20260914','response_games_prepare_20260914','native_dr_eg_prepare_20260914','phase1_daot_rc4_pure_game_m3_20260917_r1','phase1_daot_rc4_pure_game_m3_20260917_r2','all_exploration_20260913','g1_user_stop_20260914']:
 p=R/'automation_reports/CV-SincNet'/run
 if p.exists():tree(p,D/'records'/run,'experiment_record')
for run,filename in [('20260920-phase1-daot-rc4-residual-ratios-manysig-s392005-r01','final_analysis_20260921.md'),('20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01','report.md')]:
 p=R/'automation_reports/CV-SincNet'/run/filename
 if p.exists():copy(p,D/'adjacent'/run/filename,'adjacent_not_game_result')
for f in A.glob('*'):
 if f.is_file() and f.name not in ['build_dossier.py']:
  folder='chapters' if f.name.endswith('_findings.md') else 'tools' if f.suffix=='.py' else 'evidence'
  copy(f,D/folder/f.name,'current_audit')
copy(A/'build_dossier.py',D/'tools/build_dossier.py','current_audit')
copy(R/'项目.md',D/'protocol_snapshot.md','applicable_protocol')
def write(name,txt):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt,encoding='utf-8',newline='\n')
def csvwrite(name,rows):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
remote=json.loads(gzip.decompress((A/'remote_evidence_20260927.json.gz').read_bytes()))
rs=json.loads((A/'remote_summary.json').read_text(encoding='utf-8'));stat=json.loads((A/'response_recomputed.json').read_text(encoding='utf-8'))
curves=[];states=[];native=[]
for run,rd in remote['runs'].items():
 for rel,x in rd['files'].items():
  if rel.endswith(('logs.jsonl','metrics_epoch.jsonl')):
   row=rel.split('/')[0]
   for m in x['records']:
    v=m.get('source_validation',{});curves.append(dict(run=run,row=row,epoch=m.get('epoch'),accepted=m.get('accepted'),total_step=m.get('total_step'),mean_loss=m.get('mean_loss',m.get('train_loss')),source_V_accuracy=v.get('accuracy',m.get('val_acc')),source_V_CE=v.get('ce'),elapsed_seconds=m.get('elapsed_seconds'),skip_grad=m.get('train_skipped_nonfinite_grad'),skip_loss=m.get('train_skipped_nonfinite_loss')))
   if rel.endswith('metrics_epoch.jsonl'):
    native.append(dict(row=row,epochs=len(x['records']),skipped_nonfinite_grad=sum(float(m.get('train_skipped_nonfinite_grad',0)) for m in x['records']),skipped_nonfinite_loss=sum(float(m.get('train_skipped_nonfinite_loss',0)) for m in x['records'])))
 for row,x in rs['runs'][run]['rows'].items():
  c=x.get('completion',{});states.append(dict(run=run,row=row,epochs=x.get('epoch_records'),verified_training_complete=x['verified_training_complete'],accepted_steps=c.get('steps'),final_checkpoint_exists=any(a['path'].endswith('final_ssdg.pth') and a['bytes']>0 for a in x['artifacts']),score_paths='|'.join(x.get('score_paths',[])),hours=c.get('elapsed_seconds',0)/3600 if c else None))
csvwrite('tables/current_run_state.csv',states);csvwrite('tables/all_available_epoch_records.csv',curves);csvwrite('tables/native_amp_skips.csv',native)
full=[];fullclass=[]
for rel,x in remote['runs']['phase1_adv3b02_xuc_full_s392005_20260913_r1']['files'].items():
 if rel.endswith('score.json'):
  row=rel.split('/')[0];s=x['data'];m=s['metrics'];item={'row':row}
  for sc in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:
   assert m[sc]['total']==168000 and abs(m[sc]['correct']/168000-m[sc]['accuracy'])<1e-12
   item[sc]=m[sc]['accuracy']*100
   for tx,val in m[sc]['per_class_accuracy'].items():fullclass.append(dict(row=row,scene=sc,tx=tx,recall=val))
  item['leo_mean']=statistics.mean(item[k] for k in item if k.startswith('leo_'));full.append(item)
full.sort(key=lambda x:x['row']);csvwrite('tables/full9_final_scores.csv',full);csvwrite('tables/full9_per_class.csv',fullclass)
csvwrite('publication_manifest.csv',manifest)
write('excluded_files.json',json.dumps(excluded,ensure_ascii=False,indent=2))
ver={'assembled_files':len(manifest),'assembled_bytes':sum(x['bytes'] for x in manifest),'all_epoch_records':len(curves),'remote_read_at':remote['read_at'],'current_source_commit':subprocess.check_output(['git','-C',str(N),'rev-parse','HEAD'],text=True).strip(),'base_report_commit':'358a087a0e6a1e226c4257afbdd4cad51d706347','native_amp_skips':native,'note':'This is archival publication and descriptive analysis; no training code modified or experiments launched.'}
write('assembly_summary.json',json.dumps(ver,ensure_ascii=False,indent=2))
print(json.dumps(ver,ensure_ascii=False));print('FULL9',full)
