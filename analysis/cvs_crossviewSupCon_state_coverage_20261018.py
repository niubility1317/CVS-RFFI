"""Metadata-only replay of the existing frozen-state residual channel;no identity NN."""
import argparse,bisect,csv,gzip,importlib.util,json,math,os,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet');RUN='20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01';B=PROJECT/'runs'/RUN;TRAIN=PROJECT/'releases/cvs_FcrossRX_crossviewSupCon_E30_20261018_r01';ARMS=('ORD128','DUP192','SAT192');RX=(1,3,4,6,8);DAYS=(1,2,3);STATES=('LOS','PARTIAL','BLOCKED')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def records(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def layout(metadata):
 shapes={(r['TX'],r['RX'],r['day'],r['eq_index']):r['shape'] for r in metadata['metadata_records']}
 if metadata['source_RX']!=list(RX) or metadata['source_days']!=list(DAYS) or metadata['source_TX_count']!=6 or metadata['equalized_list']!=[0,1]:raise ValueError('Existing original source index metadata only')
 rows=[];start=0
 for tx in range(6):
  for rx in RX:
   for day in DAYS:
    shape=shapes[(tx,rx,day,1)]
    if shape!=[1000,256,2]:raise ValueError('Original untruncated per-combo index length differs')
    rows.append((start,start+shape[0],tx,rx,day,1));start+=shape[0]
 return rows

def decode_L(pid,role,rows,allowed,opaque):
 if role!='L':raise ValueError('Never decode a hidden U packet label')
 if len(pid)!=5 or any(type(x)is not int for x in pid):raise ValueError('Original five-integer native metadata tuple required')
 eq,rx,day,sig,base=pid;starts=[r[0] for r in rows];i=bisect.bisect_right(starts,base)-1
 if i<0 or base>=rows[i][1]:raise ValueError('Source base index outside original source pool')
 start,end,tx,rr,dd,ee=rows[i]
 if (rx,day,eq,sig)!=(rr,dd,ee,base-start):raise ValueError('Physical source metadata disagrees with exact native index layout')
 key=opaque(SimpleNamespace(tx_i=tx,rx_i=rx,day_i=day,eq_i=eq,sig_i=sig))
 if key not in allowed:raise ValueError('Only existing L6300 role IDs may be labeled')
 return tx,key

def state_from_seed(ch,cfg,seed):
 geometry=ch.sample_geometry(cfg,np.random.default_rng(ch.stable_seed(seed,'geometry')));probs=ch.state_probabilities(geometry.elevation_deg,ch.SCENARIOS[cfg.scenario][0]);s=int(np.random.default_rng(ch.stable_seed(seed,'states')).choice(3,p=probs));return STATES[s]
def marginal(c):
 out=Counter()
 for (tx,rx,day,scene,state),n in c.items():out[(rx,day,scene,state)]+=n
 return out

def setup():
 if (TRAIN/'release_commit.txt').read_text().strip()!='ec4c5d7f46aedb5aaece77fdd72dbe6ed4c157f2':raise ValueError('Pinned actual training code differs')
 s=importlib.util.spec_from_file_location('coverage41_pinned',TRAIN/'experiments/cvs_crossview_supcon41/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 from experiments.cvs_crossview_supcon41 import design as d
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 import leo_practical.channel as ch,leo_practical.residual as residual
 from cvsrffi.xuc_fusion.native import opaque_id
 return d,ch,residual,opaque_id

def synthetic_smoke(ch,residual,configs,receiver_seed):
 hits=0
 for i in range(64):
  scene=list(configs)[i%len(configs)];cfg=configs[scene];seed=ch.stable_seed(20261018501,'state-coverage/synthetic',i);hw=ch.receiver_for_session(cfg,receiver_seed,'synthetic/rx'+str(RX[i%5])+':day'+str(DAYS[i%3])+':eq1');_,meta=residual.ResidualChannel(cfg,seed,hw).process(np.ones(256,complex))
  if meta['state_end']!=state_from_seed(ch,cfg,seed):raise ValueError('Analytic frozen-state replay differs from actual residual engine')
  hits+=1
 return dict(status='PASS',native_residual_synthetic_calls=hits,synthetic_IQ_packets=64,real_IQ_packets=0,identity_NN=0,optimizer_updates=0)

def replay(arm,d,ch,opaque,rows,allowed,configs):
 root=B/(arm+'-s2026092701');total=Counter();active=Counter();epochs=Counter();seen=set();as_seen=set();perID=defaultdict(set);stage_results=[];steps=0;valid_sat=0;batch_checks=0;prior_draws=0
 for stage,offset in [('stage20',0),('source',4440)]:
  result=read(root/stage/'result.json');args=read(root/stage/'actual_native_args.json')
  if str(args['wisig_equalized'])!='1' or int(args['wisig_out_len'])!=256 or int(args['num_classes'])!=6 or 0<int(args['wisig_max_day123_per_combo'])<1000:raise ValueError('Actual original full source index or waveform length differs')
  audit=read(root/stage/'actual_channel.json')
  if audit['configurations']!=read(B/'ORD128-s2026092701/stage20/actual_channel.json')['configurations']:raise ValueError('Across-row/stage channel configs differ')
  refs=iter(records(root/stage/'reference_steps.jsonl.gz'));local=0;stage_count=Counter()
  for r in records(root/stage/'training_sequence.jsonl.gz'):
   if r['role']=='U':continue
   if r['role']!='L' or len(r['ids'])!=128:raise ValueError('Only original true L128 may supply coverage labels')
   local+=1;step=offset+local;epoch=(step-1)//222+1;ref=next(refs);steps+=1
   if ref['step']!=local or ref['cumulative_step']!=step or ref['As_packets']!=64 or not ref['all_satellite_L_supervised']:raise ValueError('Actual channel/CE step differs')
   labels=[];rr=[];keys=[]
   for pid in r['ids']:
    tx,key=decode_L(pid,'L',rows,allowed,opaque);labels.append(tx);rr.append(pid[1]);keys.append(key);seen.add(key)
   yy=np.asarray(labels);rx=np.asarray(rr);mask=(np.arange(128)+local)%2!=0;selected=np.flatnonzero(mask);counts_scene=Counter();counts_state=Counter();namespace='FcrossRX/train/e'+str(epoch)+'/step'+str(step)+'/L_S'
   all_y=np.concatenate((yy,yy[mask]));all_rx=np.concatenate((rx,rx[mask]));positive=(all_y[:,None]==all_y[None,:])&(all_rx[:,None]!=all_rx[None,:]);negative=all_y[:,None]!=all_y[None,:];valid=positive.any(1)&negative.any(1)
   if epoch>5:
    n=128 if arm=='ORD128' else 192
    if n==128:
     pos=(yy[:,None]==yy[None,:])&(rx[:,None]!=rx[None,:]);neg=yy[:,None]!=yy[None,:];valid_log=pos.any(1)&neg.any(1)
    else:valid_log=valid
    if any(q['valid_anchors']!=int(valid_log.sum()) or q['total_anchors']!=n for q in ref['SupCon_details']):raise ValueError('Actual relationship graph exposure disagrees with physical L sequence')
   for j,i in enumerate(selected):
    pid=tuple(r['ids'][int(i)]);scene=str(np.random.default_rng(ch.stable_seed(d.old.CHANNEL_SEEDS['train'],namespace,'scene',pid)).choice(d.TRAIN_SCENES));seed=ch.stable_seed(d.old.CHANNEL_SEEDS['train'],namespace,scene,pid);state=state_from_seed(ch,configs[scene],seed);prior_draws+=1;key=(int(yy[i]),int(rx[i]),int(pid[2]),scene,state);total[key]+=1;stage_count[key]+=1;epochs[(epoch,)+key]+=1;counts_scene[scene]+=1;counts_state[state]+=1;as_seen.add(keys[i]);perID[keys[i]].add((scene,state))
    if epoch>5:active[key]+=1;valid_sat+=int(valid[128+j])
   if dict(counts_scene)!=ref['scene_counts'] or dict(counts_state)!=ref['state_counts']:raise ValueError('Per-step native state/scene counts disagree with replay')
   batch_checks+=1
  if local!=(4440 if stage=='stage20' else 2220) or next(refs,None)is not None:raise ValueError('Complete actual stage L sequence/ref records required')
  saved=Counter()
  for q in result['L_channel']['strata']:saved[(q['RX'],q['day'],q['scene'],q['state'])]+=q['count']
  if marginal(stage_count)!=saved or sum(stage_count.values())!=result['L_As']:raise ValueError('All saved native RX/day/scene/state marginals must agree')
  stage_results.append(dict(stage=stage,records=local,As=sum(stage_count.values()),native_aggregate_match=True))
 if steps!=6660 or sum(total.values())!=426240 or sum(active.values())!=355200 or seen!=allowed or as_seen!=allowed:raise ValueError('Full actual L/satellite exposure differs')
 grid=[(tx,rx,day,scene,state) for tx in range(6) for rx in RX for day in DAYS for scene in d.TRAIN_SCENES for state in STATES];missing=[k for k in grid if not total[k]]
 return dict(arm=arm,counts=total,active=active,epochs=epochs,summary=dict(status='VERIFIED_METADATA_REPLAY',L_batches=steps,per_batch_native_scene_state_checks=batch_checks,actual_As_observations=426240,actual_As_unique_L=len(as_seen),known_L6300_only=True,active_SupCon_As_observations=355200,active_dense_satellite_eligible=valid_sat,TX_RX_day_scene_state_cells=len(grid),nonempty_cells=len(grid)-len(missing),missing_cells=missing,minimum_cell_count=min(total[k] for k in grid),minimum_active_cell_count=min(active[k] for k in grid),prior_state_draws=prior_draws,native_waveform_calls=0,identity_NN=0,real_IQ_values_consumed=0,model_checkpoint_reads=0,U_TX_truth_read=False,target_access=False,stages=stage_results,per_physical_state_scene_count_histogram=dict(Counter(len(v) for v in perID.values()))))

def run(out,metadata,source_analysis,wait_pids):
 out.mkdir(exist_ok=False);tic=time.perf_counter();d,ch,residual,opaque=setup();cfg=read(B/'ORD128-s2026092701/stage20/actual_channel.json');configs={s:ch.Config(**cfg['configurations'][s]) for s in d.TRAIN_SCENES}
 if any(c.processing_route!='residual' or c.fs_hz!=25e6 or c.fc_hz!=2.462e9 or c.equalization_enabled or c.boundary!='reflect' or c.reference_snr_db!=30 for c in configs.values()):raise ValueError('Use unchanged actual training residual/G1 configuration')
 rows=layout(read(metadata));allowed=set(d.read(d.SOURCE)['role_ids']['L_s'])
 with gzip.open(B/'ORD128-s2026092701/stage20/training_sequence.jsonl.gz','rt',encoding='utf8') as f:sample=json.loads(next(f))
 if sample['role']!='L' or len(sample['ids'])!=128:raise ValueError('Actual L metadata smoke requires128 legal packets')
 decoded=[decode_L(pid,'L',rows,allowed,opaque)[0] for pid in sample['ids']]
 write(out/'L_metadata_smoke.json',dict(status='PASS',L_packets=128,unambiguous_TX_matches=128,TX_labels_seen=sorted(set(decoded)),source_IQ_values_consumed=0,U_truth_read=False,target_access=False,NN=0))
 sm=synthetic_smoke(ch,residual,configs,d.old.CHANNEL_SEEDS['receiver']);write(out/'synthetic_smoke.json',sm);print('STATE_SMOKE PASS64synthetic/128actualLmetadata/0NN;waiting actual frozen source logs',flush=True)
 wait_started=time.perf_counter()
 while not (source_analysis/'complete.json').exists():
  if not any(Path('/proc',str(pid)).exists() for pid in wait_pids):raise RuntimeError('All observed owned handles gone before source analysis')
  time.sleep(20)
 proof=read(source_analysis/'complete.json')
 if proof['status']!='SOURCE_ANALYZED' or proof['formal_scores_read']:raise ValueError('Only completed source analysis;no formal feedback')
 waiting_seconds=time.perf_counter()-wait_started;replay_started=time.perf_counter();results=[]
 for arm in ARMS:results.append(replay(arm,d,ch,opaque,rows,allowed,configs));print(arm,'actual state coverage replay complete',flush=True)
 with (out/'TX_RX_day_scene_state.csv').open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['arm','TX','RX','day','scene','state','count','SupCon_active_count']);w.writeheader()
  for r in results:
   for key,n in sorted(r['counts'].items()):w.writerow(dict(arm=r['arm'],**dict(zip(['TX','RX','day','scene','state'],key)),count=n,SupCon_active_count=r['active'][key]))
 with (out/'epoch_TX_RX_state.csv').open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=['arm','epoch','TX','RX','day','scene','state','count']);w.writeheader()
  for r in results:
   for key,n in sorted(r['epochs'].items()):w.writerow(dict(arm=r['arm'],**dict(zip(['epoch','TX','RX','day','scene','state'],key)),count=n))
 write(out/'complete.json',dict(status='VERIFIED_SOURCE_TRAINING_STATE_COVERAGE',rows={r['arm']:r['summary'] for r in results},synthetic_physics_smoke=sm,all_cells_nonempty=all(not r['summary']['missing_cells'] for r in results),analysis_NN=0,model_checkpoint_reads=0,real_IQ_values_consumed=0,source_carrier_reads=0,source_labels='Only original L physicalIDs;nativebase_index->TX validated by exact metadata and rolehandle',target_access=False,training_release_changed=False,scope='Frozen propagation state only,not replayed receiverlock/AGC/noise or identity accuracy',wall_seconds=time.perf_counter()-tic,waiting_seconds=waiting_seconds,replay_seconds=time.perf_counter()-replay_started,waiting_included_in_wall=True,code_commit=(Path(__file__).resolve().parents[1]/'release_commit.txt').read_text().strip(),goal_complete=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--source-analysis',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Reconcile own existing coverage;no overwrite/retry')
 try:run(a.output,a.metadata,a.source_analysis,a.wait_pids)
 except BaseException as e:
  if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),preserve_outputs=True,no_auto_retry=True))
  raise
if __name__=='__main__':main()
