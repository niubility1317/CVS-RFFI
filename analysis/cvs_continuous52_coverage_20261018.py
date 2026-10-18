"""Three baseline As seeds: replay labeled-source channel states without IQ/NN.

Uses the exact prior metadata-only replay; changes only explicit row binding.
Never decodes U labels and never generates a channel waveform.
"""
import argparse,csv,importlib.util,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import cvs_crossviewSupCon_state_coverage_20261018 as old
import cvs_continuous52_results_20261018 as analysis
PROJECT=analysis.PROJECT;B=PROJECT/'runs'/analysis.REF
TRAIN=PROJECT/'releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
METADATA=PROJECT/'releases/cvs_FcrossRX_crossviewSupCon_state_coverage_20261018_r02/source_container_shapes.json'
read,write=old.read,old.write
def replay(row,d,ch,opaque,rows,allowed,configs):
 arm=row['key'];root=row['root']
 seeds=old.channel_seeds(d.b);total=Counter();epochs=Counter();seen=set();as_seen=set();perID=defaultdict(set);checks=0;stages=[]
 for stage,offset,expected in [('source',0,6660)]:
  result=read(root/stage/'result.json');args=read(root/stage/'actual_native_args.json');audit=read(root/stage/'actual_channel.json')
  if str(args['wisig_equalized'])!='1' or int(args['wisig_out_len'])!=256 or int(args['num_classes'])!=6 or 0<int(args['wisig_max_day123_per_combo'])<1000:raise ValueError('Actual source index changed')
  if audit['configurations']!=read(B/'B01-s2026092701/stage20/actual_channel.json')['configurations']:raise ValueError('Actual all-row source channel differs')
  refs=iter(old.records(root/stage/'reference_steps.jsonl.gz'));count=Counter();local=0
  for record in old.records(root/stage/'training_sequence.jsonl.gz'):
   if record['role']=='U':continue
   if record['role']!='L' or len(record['ids'])!=128:raise ValueError('Only original labeled128 packets')
   local+=1;step=offset+local;epoch=(step-1)//222+1;ref=next(refs);scene_counts=Counter();state_counts=Counter()
   if ref['step']!=local or ref['cumulative_step']!=step or ref['As_packets']!=64 or ref['L_forwards']!=6 or not ref['all_satellite_L_supervised'] or ref['crossview_mode']!='ORD128':raise ValueError('Actual sixforward/As step differs')
   labels=[];receivers=[];keys=[]
   for pid in record['ids']:
    tx,key=old.decode_L(pid,'L',rows,allowed,opaque);labels.append(tx);receivers.append(pid[1]);keys.append(key);seen.add(key)
   y=np.asarray(labels);rx=np.asarray(receivers);positive=(y[:,None]==y[None,:])&(rx[:,None]!=rx[None,:]);negative=y[:,None]!=y[None,:];valid=positive.any(1)&negative.any(1)
   if epoch>5 and any(q['total_anchors']!=128 or q['valid_anchors']!=int(valid.sum()) or q['true_satellite_anchors'] for q in ref['SupCon_details']):raise ValueError('Only original ordinary SupCon graph')
   selected=np.flatnonzero((np.arange(128)+local)%2!=0);namespace='FcrossRX/train/e'+str(epoch)+'/step'+str(step)+'/L_S'
   for i in selected:
    i=int(i);pid=tuple(record['ids'][i]);scene=str(np.random.default_rng(ch.stable_seed(seeds['train'],namespace,'scene',pid)).choice(d.b.TRAIN_SCENES));seed=ch.stable_seed(seeds['train'],namespace,scene,pid);state=old.state_from_seed(ch,configs[scene],seed);key=(int(y[i]),int(rx[i]),int(pid[2]),scene,state);total[key]+=1;count[key]+=1;epochs[(epoch,)+key]+=1;scene_counts[scene]+=1;state_counts[state]+=1;as_seen.add(keys[i]);perID[keys[i]].add((scene,state))
   if dict(scene_counts)!=ref['scene_counts'] or dict(state_counts)!=ref['state_counts']:raise ValueError('Per-batch actual scene/state replay mismatch')
   checks+=1
  if local!=expected or next(refs,None)is not None:raise ValueError('Complete source L/reference streams required')
  saved=Counter()
  for q in result['L_channel']['strata']:saved[(q['RX'],q['day'],q['scene'],q['state'])]+=q['count']
  if old.marginal(count)!=saved or sum(count.values())!=result['L_As']:raise ValueError('Native complete state marginal mismatch')
  stages.append(dict(stage=stage,L_batches=local,As=sum(count.values()),native_aggregate_match=True))
 if checks!=6660 or sum(total.values())!=426240 or seen!=allowed or as_seen!=allowed:raise ValueError('Complete L6300 and actual As coverage required')
 grid=[(tx,rx,day,scene,state) for tx in range(6) for rx in old.RX for day in old.DAYS for scene in d.b.TRAIN_SCENES for state in old.STATES];missing=[key for key in grid if not total[key]]
 summary=dict(status='VERIFIED_METADATA_REPLAY',actual_As_observations=sum(total.values()),actual_As_unique_L=len(as_seen),L_batches=checks,per_batch_native_scene_state_checks=checks,TX_RX_day_scene_state_cells=len(grid),nonempty_cells=len(grid)-len(missing),missing_cells=missing,minimum_cell_count=min(total[key] for key in grid),stages=stages,per_physical_state_scene_count_histogram=dict(Counter(len(v) for v in perID.values())),source_labels='originalL6300 only',U_TX_truth_read=False,identity_NN=0,target_access=False)
 return dict(row_id=row['key'],model_seed=row['seed'],arm=row['arm'],counts=total,epochs=epochs,summary=summary)
def run(out,source):
 proof=read(source/'complete.json')
 if proof['status']!='SOURCE_ANALYZED' or proof['formal_scores_read'] or proof['model_seed_count']!=1:raise ValueError('Completed source-only fourrow single-seed analysis required')
 for row in analysis.rows():analysis.source_complete(row)
 if out.exists():raise FileExistsError('Immutable coverage output')
 if (TRAIN/'release_commit.txt').read_text().strip()!='19aedc3604fed9c193237dd92366164178af0983':raise ValueError('Exact original channel metadata runtime')
 spec=importlib.util.spec_from_file_location('coverage49_runtime',TRAIN/'experiments/cvs_noise_shape_factorial43/runner.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.bootstrap()
 from experiments.cvs_noise_shape_factorial43 import design as d
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 import leo_practical.channel as ch
 from cvsrffi.xuc_fusion.native import opaque_id
 audit=read(B/'B01-s2026092701/stage20/actual_channel.json');configs={k:ch.Config(**v) for k,v in audit['configurations'].items()}
 if any(c.processing_route!='residual' or c.fs_hz!=25e6 or c.fc_hz!=2.462e9 or c.equalization_enabled or c.boundary!='reflect' or c.reference_snr_db!=30 for c in configs.values()):raise ValueError('Original full residual/noEQ actual configuration')
 # Reuse existing validated layout and role IDs; no dataset/IQ revalidation.
 layout=old.layout(read(METADATA));allowed=set(read(d.SOURCE)['role_ids']['L_s']);out.mkdir();started=time.perf_counter();results=[];off=[]
 reference=read(analysis.CACHED_SOURCE.parent/'coverage/complete.json')
 if reference['status']!='VERIFIED_SOURCE_TRAINING_STATE_COVERAGE' or reference['model_seed_count']!=3:raise ValueError('Prior verified full staged coverage required')
 for row in analysis.rows():
  if row['reused']:continue
  if row['arm']=='B01':results.append(replay(row,d,ch,opaque_id,layout,allowed,configs));print(row['key']+' state coverage complete0NN',flush=True)
  else:
   counts=[read(row['root']/'source/result.json')['L_As']]
   if counts!=[0]:raise ValueError('Actual As-off row unexpectedly generated channels')
   off.append(dict(row_id=row['key'],model_seed=row['seed'],actual_As=0))
 with (out/'TX_RX_day_scene_state.csv').open('x',newline='',encoding='utf8') as f:
  fields=['row_id','model_seed','TX','RX','day','scene','state','count'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in results:
   for key,n in sorted(r['counts'].items()):w.writerow(dict(row_id=r['row_id'],model_seed=r['model_seed'],**dict(zip(['TX','RX','day','scene','state'],key)),count=n))
 with (out/'epoch_TX_RX_state.csv').open('x',newline='',encoding='utf8') as f:
  fields=['row_id','model_seed','epoch','TX','RX','day','scene','state','count'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in results:
   for key,n in sorted(r['epochs'].items()):w.writerow(dict(row_id=r['row_id'],model_seed=r['model_seed'],**dict(zip(['epoch','TX','RX','day','scene','state'],key)),count=n))
 write(out/'complete.json',dict(status='VERIFIED_SOURCE_TRAINING_STATE_COVERAGE',model_seed_count=1,reference_staged_As_coverage=reference['As_rows']['B01-s2026092701'],As_rows={r['row_id']:r['summary'] for r in results},As_off_rows=off,all_cells_nonempty=all(not r['summary']['missing_cells'] for r in results),actual_As_observations=sum(r['summary']['actual_As_observations'] for r in results),analysis_NN=0,IQ_reads=0,checkpoint_reads=0,U_TX_truth_read=False,target_access=False,wall_seconds=time.perf_counter()-started,scope='Actual named propagation state only;not receiverlock/AGC/noise waveform replay or formal targetstates',goal_complete=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path,required=True);a=p.parse_args();run(a.output,a.source_analysis)
if __name__=='__main__':main()
