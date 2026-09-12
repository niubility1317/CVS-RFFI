"""Refresh all conversation A1 roots, retaining historical/empty rows and side results."""
import ast,gzip,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'analysis/audit_conversation_mechanisms.py').read_text(encoding='utf-8'))
remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets))
remote=remote.replace("names=['a1_ecrs", "names=['a1_e600_repair_s392005_20260911_r1','a1_ecrs")
remote=remote.replace("'final_checkpoint_exists':(folder/'final_ssdg.pth').exists()", """'final_checkpoint_exists':(folder/'final_ssdg.pth').exists(),
   'resources':{'epoch_hours':sum(x.get('epoch_time_s',0) or 0 for x in records)/3600,
    'target_hours':sum(x.get('train_time_periodic_target_seconds',0) or 0 for x in records)/3600,
    'peak_mb':max([x.get('train_muse_peak_cuda_memory_mb',0) or 0 for x in records] or [0]),
    'nonfinite_grad_epochs':sum(bool(x.get('train_skipped_nonfinite_grad',0)) for x in records),
    'nonfinite_loss_epochs':sum(bool(x.get('train_skipped_nonfinite_loss',0)) for x in records)},
   'curve':[{k:x.get(k) for k in ['epoch','epoch_time_s','train_time_periodic_target_seconds','train_loss','val_tx_acc','stage_source_val_sat_mean_tx','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss','lr']} for x in records]""")
remote=remote.replace("'scope':json.loads(sp.with_name", "'prediction_bytes':sp.with_name('predictions.json').stat().st_size if sp.with_name('predictions.json').exists() else 0,'scope':json.loads(sp.with_name")
remote=remote.replace("'stop_files':", "'dispatcher_log':(Path(state['release'])/'dispatcher.log').read_text(errors='replace') if state.get('release') and (Path(state['release'])/'dispatcher.log').exists() else None,'stop_files':")
remote=remote.replace("print(json.dumps(out,separators=(',',':')))",r'''
out['side_results']={}
for name in ['tweak_config2_portability_20260910_v7']:
 root=p/'runs'/name
 files=list(root.glob('**/results.json')) if root.exists() else []
 out['side_results'][name]={str(f.relative_to(root)):json.loads(f.read_text()) for f in files}
base=p/'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'
out['baseline_candidates']={str(f.relative_to(base)):json.loads(f.read_text()) for f in base.glob('**/score.json') if 'baseline' in str(f)}
print(json.dumps(out,separators=(',',':')))
''')
compile(remote,'<read-only-full-conversation-audit>','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=remote,text=True,encoding='utf-8',capture_output=True,check=True,timeout=60)
d=json.loads(r.stdout);out=ROOT/'analysis/all_exploration_20260913';out.mkdir(exist_ok=False)
with gzip.open(out/'evidence.json.gz','wt',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False)
print(d['captured_at'])
for run,v in d['runs'].items():
 print(run)
 for row,x in v['rows'].items():
  scores=x['scores'];s=scores[-1]['score'] if scores else None
  print(row,x['jsonl_records'],x['state'].get('status'),bool(x['live_argv']),len(scores),
   None if not s else {k:round(v['accuracy']*100,4) for k,v in s['metrics'].items()},x['resources'])
print('SIDES',[(k,list(v)) for k,v in d['side_results'].items()],'BASELINE',list(d['baseline_candidates']))
