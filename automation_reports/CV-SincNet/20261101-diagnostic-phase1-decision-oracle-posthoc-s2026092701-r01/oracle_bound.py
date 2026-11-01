"""Post-hoc label oracle: upper bound only for selecting fixed Phase1 decisions."""
import json,subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import json,numpy as np
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
runs={'dura':'20261101-phase1-dura-trisigtime-baseline-s2026092701-r01',
      'boundary':'20261101-phase1-boundary-student-E30-s2026092701-r02',
      'feature':'20261101-phase1-feature-conditioned-E30-s2026092701-r02'}
views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
rows={name:root/'runs'/run/'flow' for name,run in runs.items()};predictions={}
expected_f='20261027-phase1-pamf-repl-shuffled-views-m16-r02';expected_row='main_split-repl-shuffled_time-3view-s2026092701'
ids=None
for name,row in rows.items():
 complete=json.loads((row/'test_prediction/complete.json').read_text());summary=json.loads((row/'test_summary.json').read_text())
 assert complete['status']=='PREDICTIONS_COMPLETE' and complete['count']==168000 and complete['views']==views
 assert complete['TX_truth_reads']==0 and complete['model_updates']==0 and summary['status']=='ANALYZED'
 assert summary['all_predictions_before_truth'] and not summary['target_feedback']
 assert complete['F_run']==summary['F_run']==expected_f and complete['F_row']==summary['F_row']==expected_row
 current=np.load(row/'test_prediction/physical_ids.npy',allow_pickle=False)
 assert current.shape==(168000,) and len(np.unique(current))==len(current)
 if ids is None:ids=current
 else:assert np.array_equal(ids,current)
 predictions[name]={}
 for view in views:
  raw=np.load(row/'test_prediction'/(view+'.raw.npy'),allow_pickle=False)
  restored=np.load(row/'test_prediction'/(view+'.restored.npy'),allow_pickle=False)
  assert raw.shape==restored.shape==(168000,) and raw.dtype==restored.dtype==np.int64
  assert np.all((raw>=0)&(raw<6)) and np.all((restored>=0)&(restored<6))
  predictions[name][view]=(raw,restored)
for view in views:
 assert all(np.array_equal(predictions['dura'][view][0],predictions[name][view][0]) for name in runs)
# Load ground truth only after all 42 fixed decision arrays and metadata have been checked.
truth=json.loads((root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text())
labels=np.asarray([truth[str(pid)]['label'] for pid in ids]);clean=float(np.mean(predictions['dura']['clean'][0]==labels)*100)
relative_threshold=clean*.97;percentage_point_threshold=clean-3.;results=[]
for view in views:
 raw=predictions['dura'][view][0];dura=predictions['dura'][view][1]
 boundary=predictions['boundary'][view][1];feature=predictions['feature'][view][1]
 accuracy={name:float(np.mean(values==labels)*100) for name,values in [('raw',raw),('dura',dura),('boundary',boundary),('feature',feature)]}
 correct={name:values==labels for name,values in [('raw',raw),('dura',dura),('boundary',boundary),('feature',feature)]}
 oracle={}
 for name,keys in [('dura_boundary',('dura','boundary')),('dura_feature',('dura','feature')),
                   ('dura_boundary_feature',('dura','boundary','feature')),
                   ('raw_dura_boundary_feature',('raw','dura','boundary','feature'))]:
  union=np.logical_or.reduce([correct[key] for key in keys]);count=int(union.sum())
  oracle[name]=dict(correct=count,accuracy=count/len(ids)*100,gain_pp=(count-int(correct['dura'].sum()))/len(ids)*100,
                    below_relative_three_percent=count/len(ids)*100<relative_threshold,
                    below_three_percentage_points=count/len(ids)*100<percentage_point_threshold)
 results.append(dict(view=view,count=len(ids),accuracy=accuracy,oracle_selector=oracle))
print('ORACLE_BOUND '+json.dumps(dict(status='POSTHOC_FIXED_DECISION_SELECTOR_BOUND',count=len(ids),F_run=expected_f,F_row=expected_row,
 clean_raw_accuracy=clean,relative_three_percent_threshold=relative_threshold,three_percentage_point_threshold=percentage_point_threshold,
 predictions_fixed_before_truth=True,models_updated=0,
 runs=runs,results=results,caveat='Ground truth chooses among four already-fixed predicted labels, so this is not deployable; bound applies only to selectors restricted to those labels, not new networks, ensembles or waveforms. Previously observed single-seed Phase1 development test.'),ensure_ascii=False),flush=True)
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=120)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(x[len('ORACLE_BOUND '):] for x in stdout.splitlines() if x.startswith('ORACLE_BOUND '))
 path=Path(__file__).parent/'result.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated bound differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(p.returncode)
