"""Retrospective metadata only; all original E20 L batches, no IQ/model/U truth."""
import csv,gzip,hashlib,json,math,statistics,time
from collections import Counter
from pathlib import Path
from .reorder import BLOCK,BATCH,SEED,RX,metrics,reorder,labeled_records
ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'local_artifacts/F_physical_scene_cycle_E20_20261017/final/B00-RX1-s2026092701/source'
LABELS=ROOT/'local_artifacts/F_shared_direction_20261017/final'
OUT=ROOT/'local_artifacts/F_L_batch_order_20261017'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def records(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def distribution(values):
 s=sorted(values)
 def quant(q):
  x=(len(s)-1)*q;i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)
 return dict(min=s[0],q10=quant(.1),median=statistics.median(s),q90=quant(.9),max=s[-1],mean=statistics.mean(s))
def label_map():
 ids=read(LABELS/'fit_L_ids.json');labels={}
 for p in sorted((LABELS/'predictions').glob('RX*_day*.json')):
  z=read(p)
  if z['RX'] not in RX or z['day'] not in (1,2,3):raise ValueError('Fit source labels only')
  for key,y in zip(z['physical_ids'],z['labels']):
   if key in labels or type(y)is not int or y not in range(6):raise ValueError('Unique visible L labels required')
   labels[key]=y
 if set(ids)!=set(labels) or len(labels)!=5040 or set(ids)!=set(read(CONTROL/'source_contract.json')['role_ids']['L_s']):raise ValueError('Existing L identity contracts differ')
 return labels

def resolve(meta,labels,cache):
 if not isinstance(meta,list) or len(meta)!=5 or any(type(x)is not int for x in meta):raise ValueError('Original L physical metadata required')
 eq,rx,day,sig,base=meta;k=tuple(meta)
 if eq!=1 or rx not in RX or day not in (1,2,3) or min(sig,base)<0:raise ValueError('No held RX/other role')
 if k not in cache:
  # Original ManySig source index is TX/RX/day/eq/sig nested;1000 per cell.
  # Resolve only already-open L, then require its saved opaque ID and label.
  # No matching by sig alone; no U record is passed to this function.
  rxs=(1,3,4,6,8);tx=base//15000
  expected=tx*15000+rxs.index(rx)*3000+(day-1)*1000+sig
  if tx not in range(6) or sig>=1000 or base!=expected:raise ValueError('L base index does not match frozen source construction')
  opaque=hashlib.sha256(('%d/%d/%d/%d/%d'%(tx,rx,day,eq,sig)).encode()).hexdigest()[:24]
  if opaque not in labels or labels[opaque]!=tx:raise ValueError('L metadata and saved visible label disagree')
  cache[k]=(labels[opaque],eq,rx,day,sig,base,tx)
 return cache[k]

def run():
 started=time.perf_counter();labels=label_map();cache={};raw=[]
 args=read(CONTROL/'actual_native_args.json')
 if args['use_tx_rx_balanced_sampler'] or args['batch_size']!=128 or args['epochs']!=20:raise ValueError('Expected original unbalanced E20 L128')
 for batch in labeled_records(records(CONTROL/'training_sequence.jsonl.gz')):
  if len(batch)!=128:raise ValueError('Original L128 only')
  raw.append([resolve(v,labels,cache) for v in batch])
 if len(raw)!=4440 or len(cache)!=5040:raise ValueError('Complete original L4440/5040 required')
 fixed=[];block_count=0
 for begin in range(0,len(raw),BLOCK):
  original=raw[begin:begin+BLOCK];ordered=reorder(original,block_count)
  if Counter(x for b in original for x in b)!=Counter(x for b in ordered for x in b):raise ValueError('Changed original block observations')
  fixed.extend(ordered);block_count+=1
 stats={k:[metrics(b) for b in rows] for k,rows in (('original',raw),('reordered',fixed))}
 saved=list(records(CONTROL/'satmix_steps.jsonl.gz'));checked=0
 if len(saved)!=4440:raise ValueError('Full original saved steps required')
 for step,z in enumerate(saved):
  if z['step']!=step+1 or z['epoch']!=step//222+1:raise ValueError('Original step order')
  details=z['supcon_details']
  if step<1110:
   if details:raise ValueError('Original SupCon warmup altered')
  else:
   if len(details)!=4:raise ValueError('Four existing layer/forward terms required')
   for d in details:
    for key in ('valid_anchors','total_anchors','positive_min','positive_max'):
     if d[key]!=stats['original'][step][key]:raise ValueError('Derived original masks differ from runtime')
    checked+=1
 summary={name:{k:distribution([r[k] for r in rows]) for k in rows[0]} for name,rows in stats.items()}
 original_count=Counter(x for b in raw for x in b);new_count=Counter(x for b in fixed for x in b)
 if original_count!=new_count:raise ValueError('Global physical observations changed')
 changed=sum(a!=b for old,new in zip(raw,fixed) for a,b in zip(old,new))
 result=dict(status='ANALYZED_METADATA_ONLY',source_run='20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01',source_row='B00-RX1-s2026092701',source_execution_commit='c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9',model_seed=2026092701,split_seed=392005,steps=4440,epochs=20,L_batch_size=128,L_appearances=568320,unique_L=5040,all_original_masks_match_saved_terms=checked,block_updates=BLOCK,blocks=block_count,private_metadata_seed=SEED,all_block_physical_multisets_identical=True,global_physical_frequency_identical=True,physical_visit_distribution=distribution(list(original_count.values())),reordered_positions=changed,summary=summary,starved_original_updates=sum(z['valid_anchors']<128 for z in stats['original']),starved_original_active_updates=sum(z['valid_anchors']<128 for z in stats['original'][1110:]),low_positive_min_updates={name:{str(cut):sum(z['positive_min']<cut for z in rows) for cut in (8,12,15)} for name,rows in stats.items()},labels_from='Only saved fit L5040 labels and original L records; U records skipped before accessing IDs',U_truth_resolutions=0,IQ_reads=0,checkpoint_loads=0,NN=0,optimizer_updates=0,new_Ag=False,new_As=False,training_launched=False,target_access=False,performance_claim=False,whole_goal_complete=False,elapsed_seconds=time.perf_counter()-started)
 OUT.mkdir(exist_ok=True);p=OUT/'analysis_summary.json'
 if p.exists():raise FileExistsError('Preserve existing metadata result')
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 with (OUT/'all_4440_batch_metrics.csv').open('x',encoding='utf8',newline='') as f:
  keys=('order','step','epoch',*stats['original'][0]);writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for name,rows in stats.items():
   for i,row in enumerate(rows,1):writer.writerow(dict(order=name,step=i,epoch=(i-1)//222+1,**row))
 print(json.dumps({k:result[k] for k in ('status','steps','unique_L','all_original_masks_match_saved_terms','all_block_physical_multisets_identical','starved_original_active_updates','low_positive_min_updates','elapsed_seconds')}))
 print(json.dumps({name:{k:row[k] for k in ('positive_min','positive_variance','RX_count_range','batch_TX_RX_mutual_information_nats')} for name,row in summary.items()}))
 return result
if __name__=='__main__':run()
