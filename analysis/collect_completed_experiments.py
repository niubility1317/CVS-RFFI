import json,subprocess
from pathlib import Path
REMOTE=r'''
import json,csv,io,re,gc,datetime,sys,statistics
from pathlib import Path
import torch
torch.set_num_threads(2)
p=Path('/home/szu2070436088/2510044040/CV-SincNet');run='a1_mechanism_periodic_s392005_20260910_r1'
root=p/'runs'/run;state=json.loads((root/'pipeline_state.json').read_text());release=Path(state['release'])
sys.path[:0]=[str(release/'code'),str(release)]
result={'captured_at':datetime.datetime.now().isoformat(),'run':run,'state':state,'rows':{}}
for name,item in state['rows'].items():
 if item['status']!='EXPLORATORY_SCORED_PENDING_ANALYSIS':continue
 folder=root/name;log=p/'logs'/run/(name+'.train.log')
 records=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text().splitlines() if x.strip()]
 csvrows=list(csv.DictReader(io.StringIO((folder/'metrics_epoch.csv').read_text())))
 assert len(records)==len(csvrows)==200 and not any(None in x for x in csvrows)
 assert [int(x['epoch']) for x in records]==list(range(1,201))
 stdout=log.read_text(errors='strict');lines=stdout.splitlines()
 checkpoint=folder/'final_ssdg.pth';snap=folder/'epoch_200_ssdg.pth'
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);fixed=torch.load(snap,map_location='cpu',weights_only=False)
 args=saved['args']
 assert saved['epoch']==fixed['epoch']==200 and args['from_scratch'] and args['a1_scratch_only']
 assert not args.get('baseline_ckpt') and not args.get('teacher_ckpt')
 exact=set(saved['model'])==set(fixed['model']) and all(torch.equal(v,fixed['model'][k]) for k,v in saved['model'].items())
 assert exact
 selected=['from_scratch','a1_scratch_only','baseline_ckpt','teacher_ckpt','epochs','seed','wisig_train_rxs',
  'wisig_train_days','labeled_ratio','unlabeled_ratio','source_val_ratio','best_metric','a1_source_screen_only']
 row={'state':item,'checkpoint':str(checkpoint),'checkpoint_epoch':saved['epoch'],'e200_snapshot_model_equal_final':exact,
  'checkpoint_args':{k:args.get(k) for k in selected},'completed_at':datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat(),
  'checkpoint_bytes':checkpoint.stat().st_size,'jsonl_records':len(records),'csv_records':len(csvrows),
  'stdout_lines':len(lines),'errors':[x for x in lines if re.search(r'Traceback|RuntimeError:|CUDA out of memory|RC4_SYSTEMIC_NONFINITE',x)],
  'warning_count':sum('Warning:' in x or 'WARNING' in x for x in lines),
  'source_init':[x for x in lines if 'init=scratch' in x], 'scores':[]}
 del saved,fixed;gc.collect()
 for score in sorted((folder/'target_epochs').glob('E*/score.json')):
  scope=score.with_name('evaluation_scope.json');pred=score.with_name('predictions.json')
  assert scope.is_file() and pred.is_file()
  row['scores'].append({'epoch':int(score.parent.name[1:]),'score':json.loads(score.read_text()),
   'scope':json.loads(scope.read_text()),'prediction_bytes':pred.stat().st_size})
 assert [x['epoch'] for x in row['scores']]==list(range(80,201,10))
 fields=['epoch','epoch_time_s','train_time_periodic_target_seconds','train_loss','train_loss_tx_labeled','val_tx_acc',
 'stage_source_val_sat_mean_tx','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss',
 'train_muse_peak_cuda_memory_mb','train_muse_time_train_batches_s','train_a1_ema_successful_updates','lr']
 row['curve']=[{k:x.get(k) for k in fields} for x in records]
 row['epoch_hours']=sum(x['epoch_time_s'] for x in records)/3600
 row['target_test_hours']=sum(x.get('train_time_periodic_target_seconds',0) for x in records)/3600
 row['peak_mb']=max(x.get('train_muse_peak_cuda_memory_mb',0) for x in records)
 row['final_source_clean']=records[-1].get('val_tx_acc');row['final_source_leo']=records[-1].get('stage_source_val_sat_mean_tx')
 row['final_loss']=records[-1].get('train_loss')
 result['rows'][name]=row
print(json.dumps(result))
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607',
 '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=90)
if result.returncode:raise RuntimeError(result.stderr)
d=json.loads(result.stdout);out=Path(__file__).with_name('completed_experiments_20260910_2122.json')
out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'captured_at':d['captured_at'],'completed_rows':list(d['rows']),'epochs':sum(x['jsonl_records'] for x in d['rows'].values()),'scores':sum(len(x['scores']) for x in d['rows'].values())}))
