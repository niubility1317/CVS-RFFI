"""Read-only, full-log audit of the eight completed rows already discussed."""
import gzip, json, subprocess
from pathlib import Path

BASE = Path(__file__).parent
REMOTE = r'''
import json,csv,io,re,datetime
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='a1_mechanism_periodic_s392005_20260910_r1'
names=['B0_FIXED','B1_TAIL_LR','B2_RC4_WEIGHT','D0_NO_ORBIT','D1_THREE_VIEW','D2_PHYSICAL_ORBIT','D3_TANGENT','F3_R3_SWAP']
out={'captured_at':datetime.datetime.now().isoformat(),'run':run,'rows':{}}
pattern=re.compile(r'epoch$|^lr|loss_sat|daot|orbit|tangent|pseudo|accept|cross_rx|nonfinite|sat_applied|sat_count|sat_cls|train_loss$|val_tx_acc$|stage_source_val_sat|time_train_batches|lambda_sat|loss_group|group_ce|rc4|r3_')
for name in names:
 folder=p/'runs'/run/name
 records=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text().splitlines() if x.strip()]
 csvrows=list(csv.DictReader(io.StringIO((folder/'metrics_epoch.csv').read_text())))
 assert len(records)==len(csvrows)==200 and [int(x['epoch']) for x in records]==list(range(1,201))
 text=(p/'logs'/run/(name+'.train.log')).read_text(encoding='utf-8')
 lines=text.splitlines()
 out['rows'][name]={'jsonl_records':len(records),'csv_records':len(csvrows),'stdout_lines':len(lines),
  'errors':[x for x in lines if re.search(r'Traceback|RuntimeError:|CUDA out of memory|RC4_SYSTEMIC_NONFINITE',x)],
  'warning_count':sum('WARNING' in x or 'Warning:' in x for x in lines),
  'config_lines':[x for x in lines if x.startswith('[CONFIG-') or 'init=scratch' in x or '[BATCH-GEOM]' in x],
  'all_field_names':sorted(set().union(*(x.keys() for x in records))),
  'records':[{k:v for k,v in x.items() if pattern.search(k)} for x in records]}
print(json.dumps(out,separators=(',',':')))
'''
compile(REMOTE, '<remote-read-only>', 'exec')
r = subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=60)
if r.returncode:
    raise RuntimeError(r.stderr)
d=json.loads(r.stdout)
dest=BASE/'method_class_analysis_20260910'
dest.mkdir(exist_ok=True)
with gzip.open(dest/'full_log_audit.json.gz','wt',encoding='utf-8') as stream:
    json.dump(d,stream,ensure_ascii=False,separators=(',',':'))
print(json.dumps({n:{k:v for k,v in x.items() if k not in ('records','all_field_names','config_lines')} for n,x in d['rows'].items()}))
print('B1 fields',list(d['rows']['B1_TAIL_LR']['records'][-1]))
