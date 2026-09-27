from pathlib import Path
import gzip,json,csv,collections,math
OUT=Path(__file__).resolve().parent
d=json.loads(gzip.decompress((OUT/'remote_evidence_20260927.json.gz').read_bytes()))
summary={'read_at':d['read_at'],'runs':{},'process_note':'Matched argv include five unrelated 20260927 runs referencing the old source contract. None uses an audited run as --output.'}
for n,rd in d['runs'].items():
 files=rd['files']; rows=collections.defaultdict(dict); count=collections.Counter()
 for rel,item in files.items():
  if '/' not in rel:continue
  row,name=rel.split('/',1); records=item.get('records',[])
  if item.get('parse_errors'):raise ValueError((n,rel,item['parse_errors']))
  if name in ['logs.jsonl','metrics_epoch.jsonl']:
   count[name]+=len(records)
   rows[row]['epoch_records']=len(records);rows[row]['log_path']=rel
   rows[row]['epoch_range']=[records[0].get('epoch'),records[-1].get('epoch')] if records else []
   rows[row]['last_epoch_record']=records[-1] if records else {}
   rows[row]['nonfinite_mean_loss']=sum(not math.isfinite(float(x['mean_loss'])) for x in records if 'mean_loss' in x)
  elif name=='completion.json':rows[row]['completion']=item.get('data')
  elif name=='resolved_config.json':rows[row]['config']=item.get('data')
  elif 'score' in name:rows[row]['score_paths']=rows[row].get('score_paths',[])+[rel]
 for row,x in rows.items():
  arts=[a for a in rd['artifacts'] if a['path'].startswith(row+'/')]
  x['artifacts']=arts
  x['verified_training_complete']=bool(x.get('completion',{}).get('status')=='TRAINING_COMPLETE' and x.get('epoch_range',[None,None])[-1]==200 and any(a['path'].endswith('final_ssdg.pth') and a['bytes']>0 for a in arts))
 sr={'rows':rows,'record_counts':dict(count),'stdout_files':len(rd['stdout_scans']),'stdout_lines':sum(s['lines'] for s in rd['stdout_scans']),'stdout_match_count':sum(len(s['matches']) for s in rd['stdout_scans']),'completed_verified':sum(x['verified_training_complete'] for x in rows.values())}
 summary['runs'][n]=sr
 print(n,'counts',sr['record_counts'],'verified',sr['completed_verified'],'stdout',sr['stdout_lines'],'matches',sr['stdout_match_count'])
 print('artifacts other than ckpt',[a for a in rd['artifacts'] if not a['path'].endswith(('.pth','.pt'))][:20])
 for row,x in sorted(rows.items()):
  print(row,'epochs',x.get('epoch_records'),'complete',x['verified_training_complete'],'score',x.get('score_paths'),'lastkeys',list(x.get('last_epoch_record',{}))[:8])
(OUT/'remote_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
