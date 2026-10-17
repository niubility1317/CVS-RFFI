"""Recount every frozen source physical pair without IQ/model/truth access."""
import gzip,json,math
from collections import Counter,defaultdict
from pathlib import Path

def recount(base):
 base=Path(base);done=json.loads((base/'completion.json').read_text());r=json.loads((base/'result.json').read_text())
 if done['status']!='ANALYZED' or done['physical_pairs']!=18900 or done['NN'] or done['checkpoint_loads'] or done['target_access']:raise ValueError('Source physical run not closed')
 pairs=set();ids=Counter();cell=defaultdict(set);n=effective=locks=0;by_rx=Counter();rx_changes=Counter();by_scene=Counter();scene_changes=Counter();per_TX=Counter();TX_changes=Counter()
 with gzip.open(base/'physical_pairs.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   if not line.endswith('\n'):raise ValueError('Truncated source observations')
   x=json.loads(line);key=(x['physical_id'],x['scene'])
   if key in pairs or not x['same_input_latent']:raise ValueError('Duplicate physical pair or inconsistent latent')
   pairs.add(key);ids[x['physical_id']]+=1;cell[(x['TX'],x['RX'],x['day'])].add(x['physical_id'])
   a,b=x['white_scale'],x['waveform_scale'];flag=x['white_lock']!=x['waveform_lock'] or max(a/b,b/a)>=1.10
   if flag!=x['effective_change'] or x['lock_changed']!=(x['white_lock']!=x['waveform_lock']) or abs(x['delta_quality_db']-(x['waveform_quality_db']-x['white_quality_db']))>1e-10:raise ValueError('Saved paired calculation inconsistent')
   if not all(math.isfinite(x[k]) for k in ('white_quality_db','waveform_quality_db','white_scale','waveform_scale','relative_output_difference')):raise ValueError('Nonfinite observed values')
   n+=1;effective+=flag;locks+=x['lock_changed'];by_rx[x['RX']]+=1;rx_changes[x['RX']]+=flag;by_scene[x['scene']]+=1;scene_changes[x['scene']]+=flag;per_TX[x['TX']]+=1;TX_changes[x['TX']]+=flag
 if n!=18900 or len(ids)!=6300 or set(ids.values())!={3} or len(cell)!=90 or set(map(len,cell.values()))!={70}:raise ValueError('Complete original source coverage required')
 if set(by_rx)!={1,3,4,6,8} or set(by_rx.values())!={3780} or set(by_scene.values())!={6300}:raise ValueError('RX/scene coverage differs')
 qualified=effective/n>=.05 and all(rx_changes[k]/by_rx[k]>=.03 for k in by_rx)
 expected='PASS_PHYSICAL_MISMATCH_ONLY' if qualified else 'FAIL_FIXED_QUALITY_COUPLING_CLOSED'
 if r['decision']!=expected or r['overall']['effective_changes']!=effective or r['overall']['lock_changes']!=locks:raise ValueError('Fixed source conclusion does not match full observations')
 return dict(status='VERIFIED',physical_pairs=n,unique_L=6300,all90_source_cells_70_L_each=True,effective=effective,effective_fraction=effective/n,locks_changed=locks,decision=expected,by_RX={str(k):dict(n=by_rx[k],changed=rx_changes[k],fraction=rx_changes[k]/by_rx[k]) for k in sorted(by_rx)},by_scene={k:dict(n=by_scene[k],changed=scene_changes[k],fraction=scene_changes[k]/by_scene[k]) for k in sorted(by_scene)},by_TX={str(k):dict(n=per_TX[k],changed=TX_changes[k],fraction=TX_changes[k]/per_TX[k]) for k in sorted(per_TX)},new_IQ_reads=0,NN=0,CP=0,updates=0,raw_truth_reads=0,joint_goal_complete=False)
if __name__=='__main__':
 import sys
 print(json.dumps(recount(sys.argv[1])))
