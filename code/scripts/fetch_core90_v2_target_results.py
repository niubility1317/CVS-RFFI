"""Read-only collection and full prediction verification of a completed target batch."""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json,sys,math,time
from array import array
from collections import defaultdict
from pathlib import Path
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
run=sys.argv[1];out=root/'runs'/run
sys.path.insert(0,str(root/'releases'/run/'code/scripts'))
sys.dont_write_bytecode=True
from collect_core90_v2_completed import metric
m=json.loads((out/'frozen_manifest.json').read_text());c=json.loads((out/'complete.json').read_text());p=json.loads((out/'predictions_complete.json').read_text())
assert c['complete'] and p['complete'] and c['rows']==m['rows']==p['rows']
assert all(v==m['samples_per_scene'] for v in p['counts'].values()) and p['truth_accessed_for_scoring'] is False
truth={}
for line in (out/'target_truth.jsonl').open():
 t=json.loads(line);assert t['sample_id'] not in truth;truth[t['sample_id']]=t
assert len(truth)==m['samples_per_scene']
truth_index={sample:i for i,sample in enumerate(truth)};truth_values=list(truth.values());n=len(truth)
scene_index={s:i for i,s in enumerate(m['scenes'])}
scores={};source={};proof={};confusions={};prediction_maps={}
for row in m['rows']:
 k=m['num_classes'];mat={s:[[0]*k for _ in range(k)] for s in m['scenes']};seen=set();count=0
 pm=(array('b',[-1])*(n*4),array('f',[0.])*(n*4))
 for line in (out/(row+'_predictions.jsonl')).open():
  r=json.loads(line);key=(r['scene'],r['sample_id']);assert key not in seen;seen.add(key)
  assert not {'truth','y','capture_group'}.intersection(r)
  t=truth[r['sample_id']];assert r['rx_i']==t['rx_i'] and r['day_i']==t['day_i']
  assert 0<=r['prediction']<k and math.isfinite(r['confidence']) and 0<=r['confidence']<=1
  mat[r['scene']][t['truth']][r['prediction']]+=1;count+=1
  index=scene_index[r['scene']]*n+truth_index[r['sample_id']];pm[0][index]=r['prediction'];pm[1][index]=r['confidence']
 assert count==m['samples_per_scene']*4
 s=json.loads((out/(row+'_target_scores.json')).read_text());assert s['complete'] and s['target_evaluated'] and not s['source_only']
 maxerror=0.
 for scene,cm in mat.items():
  independent=metric(cm);saved=s['scenes'][scene]
  assert independent['count']==m['samples_per_scene']
  for key in ('accuracy','macro_recall','macro_f1','worst_tx_accuracy'):maxerror=max(maxerror,abs(independent[key]-saved[key]))
  for group in ('per_rx','per_day'):
   assert sum(v['count'] for v in saved[group].values())==saved['count']
   assert math.isclose(sum(v['accuracy']*v['count'] for v in saved[group].values())/saved['count'],saved['accuracy'],abs_tol=1e-12)
 assert maxerror<1e-12
 scores[row]=s;confusions[row]=mat;proof[row]=dict(predictions=count,max_metric_error=maxerror)
 src=Path(m['checkpoints'][row]).parent/'source_final_eval/source_scores.json'
 source[row]=json.loads(src.read_text())
 prediction_maps[row]=pm
pairs=[]
for a,b in [('V2_A','V2_B'),('V2_A','V2_C'),('V2_B','V2_D')]:
 for seed in (392005,392006,392007):
  aa=f'{a}_seed{seed}';bb=f'{b}_seed{seed}'
  if aa not in prediction_maps or bb not in prediction_maps:continue
  pair={s:dict(disagreement=0,correct_to_wrong=0,wrong_to_correct=0,max_confidence_delta=0.) for s in m['scenes']}
  for index,pred in enumerate(prediction_maps[aa][0]):
   confidence=prediction_maps[aa][1][index];other=prediction_maps[bb][0][index];oc=prediction_maps[bb][1][index]
   v=pair[m['scenes'][index//n]];y=truth_values[index%n]['truth']
   v['disagreement']+=pred!=other;v['correct_to_wrong']+=pred==y and other!=y;v['wrong_to_correct']+=pred!=y and other==y
   v['max_confidence_delta']=max(v['max_confidence_delta'],abs(confidence-oc))
  pairs.append(dict(baseline=aa,variant=bb,scenes=pair))
print(json.dumps(dict(manifest=m,complete=c,predictions_complete=p,scores=scores,source_scores=source,verification=proof,
 confusion_matrices=confusions,paired_predictions=pairs,stdout=(root/'logs'/run/'eval.stdout.log').read_text(),collected_unix=time.time())))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='core90_game_v2_target10_20260912_r1');p.add_argument('--output',required=True)
    args=p.parse_args()
    r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-',args.run],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=240)
    if r.returncode:raise RuntimeError(r.stderr)
    data=json.loads(r.stdout);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(models=len(data['scores']),predictions=sum(v['predictions'] for v in data['verification'].values()),
        max_metric_error=max(v['max_metric_error'] for v in data['verification'].values()),seconds=data['complete']['total_seconds'])))

if __name__=='__main__':main()
