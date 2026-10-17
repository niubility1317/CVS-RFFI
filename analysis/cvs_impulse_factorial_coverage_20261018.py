"""P43 labeled-source metadata replay;no IQ,checkpoint,hidden-U label or NN."""
import argparse,csv,importlib.util,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import cvs_crossviewSupCon_state_coverage_20261018 as old
PROJECT=old.PROJECT;RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01';B=PROJECT/'runs'/RUN
TRAIN=PROJECT/'releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01';ARMS=('B01','B11_WHITE','B11_BG');read,write=old.read,old.write
METADATA=PROJECT/'releases/cvs_FcrossRX_crossviewSupCon_state_coverage_20261018_r02/source_container_shapes.json'
def replay(arm,d,ch,opaque,rows,allowed,configs):
 seeds=old.channel_seeds(d.b);root=B/(arm+'-s2026092701');total=Counter();epochs=Counter();seen=set();as_seen=set();perID=defaultdict(set);checks=0;stages=[]
 for stage,offset,expected in [('stage20',0,4440),('source',4440,2220)]:
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
 return dict(arm=arm,counts=total,epochs=epochs,summary=summary)
def run(out,source):
 proof=read(source/'complete.json')
 if proof['status']!='SOURCE_ANALYZED' or proof['formal_scores_read']:raise ValueError('Completed full source recount required first')
 if (TRAIN/'release_commit.txt').read_text().strip()!='19aedc3604fed9c193237dd92366164178af0983':raise ValueError('Use exact actual P43 training release')
 s=importlib.util.spec_from_file_location('coverage43_runtime',TRAIN/'experiments/cvs_noise_shape_factorial43/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 from experiments.cvs_noise_shape_factorial43 import design as d
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 import leo_practical.channel as ch
 from cvsrffi.xuc_fusion.native import opaque_id
 audit=read(B/'B01-s2026092701/stage20/actual_channel.json');configs={k:ch.Config(**v) for k,v in audit['configurations'].items()}
 if any(c.processing_route!='residual' or c.fs_hz!=25e6 or c.fc_hz!=2.462e9 or c.equalization_enabled or c.boundary!='reflect' or c.reference_snr_db!=30 for c in configs.values()):raise ValueError('Unchanged actual residual/G1 required')
 # Reuse validated source container shapes and native index layout;no data revalidation.
 rows=old.layout(read(METADATA));allowed=set(read(d.SOURCE)['role_ids']['L_s']);out.mkdir(exist_ok=False);tic=time.perf_counter();results=[]
 for arm in ARMS:results.append(replay(arm,d,ch,opaque_id,rows,allowed,configs));print(arm+' coverage complete0NN',flush=True)
 with (out/'TX_RX_day_scene_state.csv').open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['arm','TX','RX','day','scene','state','count']);w.writeheader()
  for r in results:
   for key,n in sorted(r['counts'].items()):w.writerow(dict(arm=r['arm'],**dict(zip(['TX','RX','day','scene','state'],key)),count=n))
 with (out/'epoch_TX_RX_state.csv').open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['arm','epoch','TX','RX','day','scene','state','count']);w.writeheader()
  for r in results:
   for key,n in sorted(r['epochs'].items()):w.writerow(dict(arm=r['arm'],**dict(zip(['epoch','TX','RX','day','scene','state'],key)),count=n))
 write(out/'complete.json',dict(status='VERIFIED_SOURCE_TRAINING_STATE_COVERAGE',rows={r['arm']:r['summary'] for r in results},all_cells_nonempty=all(not r['summary']['missing_cells'] for r in results),As_off_rows={a:0 for a in ('B00','B10_WHITE','B10_BG')},source_proof=str(source),analysis_NN=0,IQ_reads=0,checkpoint_reads=0,U_TX_truth_read=False,target_access=False,wall_seconds=time.perf_counter()-tic,scope='Frozen propagation state;not receiverlock/AGC/noise replay or formal state accuracy',goal_complete=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve existing output')
 run(a.output,a.source_analysis)
if __name__=='__main__':main()
