"""Independent full actual-input audit. No candidate reordering code/model/IQ/U labels."""
import gzip,hashlib,json,math
from collections import Counter,defaultdict,deque
from pathlib import Path
RX=(3,4,6,8)
def require(ok,message):
 if not ok:raise ValueError(message)
def read(path):return json.loads(path.read_text(encoding='utf8'))
def records(path):
 with gzip.open(path,'rt',encoding='utf8') as stream:
  return [json.loads(line) for line in stream if line.strip()]
def pid(p):
 require(isinstance(p,list) and len(p)==5 and all(type(x)is int for x in p),'Five integer physical metadata fields')
 eq,rx,day,sig,base=p;require(eq==1 and rx in RX and day in (1,2,3) and sig>=0 and base>=0,'Only fixed fit-source metadata')
 return tuple(p)
def items(ids,labels):
 require(len(ids)==len(labels)==128 and all(type(y)is int and y in range(6) for y in labels),'Only original visible L128 labels')
 return [(y,*pid(p)) for p,y in zip(ids,labels)]
def batch_metrics(values):
 require(len(values)==128,'L128 statistics only');cells=Counter((p[0],p[2]) for p in values);classes=Counter(p[0] for p in values);receivers=Counter(p[2] for p in values)
 positives=[classes[y]-cells[(y,r)] for y,e,r,d,s,b in values];good=[v for p,v in zip(values,positives) if v>0 and classes[p[0]]<128];mu=math.fsum(positives)/128
 days={(y,r,d) for y,e,r,d,s,b in values};information=math.fsum((n/128)*math.log(n*128/(classes[y]*receivers[r])) for (y,r),n in cells.items())
 return dict(valid_anchors=len(good),total_anchors=128,positive_min=min(good,default=0),positive_max=max(good,default=0),positive_mean=mu,positive_variance=math.fsum((v-mu)**2 for v in positives)/128,TX_count_range=max(classes.get(y,0) for y in range(6))-min(classes.get(y,0) for y in range(6)),RX_count_range=max(receivers.get(r,0) for r in RX)-min(receivers.get(r,0) for r in RX),missing_TX_RX_cells=24-len(cells),missing_TX_RX_day_cells=72-len(days),batch_TX_RX_mutual_information_nats=information)
def same_stats(a,b):
 require(set(a)==set(b),'Statistics keys differ')
 for k,v in a.items():require(math.isclose(v,b[k],rel_tol=1e-10,abs_tol=1e-10),'Saved statistic differs: '+k)
def block(rows,arm):
 require(arm in ('O0','O1') and len(rows)==6,'Registered six-update block')
 before=[items(v['original_ids'],v['original_labels']) for v in rows];after=[items(v['actual_ids'],v['actual_labels']) for v in rows]
 flat=[p for row in before for p in row];actual=[p for row in after for p in row]
 require(Counter(flat)==Counter(actual),'Physical packet/label multiplicity changed in block')
 if arm=='O0':require(before==after,'Control input order changed')
 else:
  universe={(p[0],p[2]) for p in flat}
  for key in universe:
   n=[sum((p[0],p[2])==key for p in row) for row in after];require(max(n)-min(n)<=1,'Actual TX/RX quota not balanced within block')
 queues=defaultdict(deque)
 for i,p in enumerate(flat):queues[p].append(i)
 positions=[queues[p].popleft() for p in actual]
 require(sorted(positions)==list(range(768)),'A physical observation was duplicated/deleted')
 return before,after,sum(i!=j for i,j in enumerate(positions))

def audit(src,arm,mix):
 source=read(src/'source_contract.json');expected_L=set(source['role_ids']['L_s']);a=read(src/'L_batch_order_audit.json');cfg=read(src/'L_batch_order_config.json');trace=records(src/'L_batch_order_steps.jsonl.gz');sequence=records(src/'training_sequence.jsonl.gz')
 require(a['config']==cfg['config'] and a['arm']==arm and a['epochs']==20 and a['steps']==4440 and a['blocks']==740,'Full final20 actual L audit required')
 require(a['config']['block_updates']==6 and a['config']['policy']==('identity' if arm=='O0' else 'balance_TX_RX'),'Fixed L policy')
 require(a['new_IQ_transform'] is False and a['U_buffered'] is False and a['hidden_U_truth_read'] is False and a['target_access'] is False and a['new_NN_forward_packets']==0,'L-only no IQ/NN/U transform scope')
 require(a['actual_L_generators']==cfg['actual_L_generators'],'Actual common L generators')
 g=a['actual_L_generators'];require(g['L_sampler_seed']==202610175201 and g['L_loader_seed']==202610175202 and g['model_seed']==2026092701 and g['private_L_generators'] and g['original_U_loader_unmodified'],'Frozen common L generators;original U untouched')
 require(len(trace)==len(mix)==len(a['original_stats'])==len(a['actual_stats'])==4440,'All4440 actual input records required')
 original_L=[];U=[]
 for v in sequence:
  require(v['role'] in ('L','U'),'Unexpected input role')
  values=[pid(x) for x in v['ids']]
  if v['role']=='L':require(len(values)==128,'Original L128');original_L.append(values)
  else:U.append(values) # Never resolve a hidden U transmitter or derive its label.
 require(len(original_L)==4440 and sum(map(len,U))==396368 and len({x for row in U for x in row})==45360,'Original full L/U observations')
 original_all=Counter();actual_all=Counter();seen_L=set();base_map={};moved=0;stats_original=[];stats_actual=[]
 for begin in range(0,4440,6):
  rows=trace[begin:begin+6]
  for offset,v in enumerate(rows):
   step=begin+offset+1;require(v['step']==step and v['epoch']==(step-1)//222+1 and v['block']==(step-1)//6+1 and v['slot']==(step-1)%6 and v['arm']==arm,'Actual order/epoch/block sequence')
   require(v['IQ_changed'] is False and v['U_used'] is False,'No IQ alteration/U buffer')
   require([tuple(p) for p in v['original_ids']]==original_L[step-1],'Prefetch L stream and saved original order differ')
  before,after,changed=block(rows,arm);moved+=changed
  for offset,(raw,ordered) in enumerate(zip(before,after)):
   i=begin+offset;original_all.update(raw);actual_all.update(ordered)
   for value in raw:
    y,eq,r,day,sig,base=value;opaque=hashlib.sha256(('%d/%d/%d/%d/%d'%(y,r,day,eq,sig)).encode()).hexdigest()[:24]
    require(opaque in expected_L,'Actual visible L escaped existing role IDs or label map');seen_L.add(opaque)
    require(base_map.setdefault(base,(y,eq,r,day,sig))==(y,eq,r,day,sig),'L base index label/physical binding changed')
   raw_stats=batch_metrics(raw);actual_stats=batch_metrics(ordered);same_stats(raw_stats,a['original_stats'][i]);same_stats(actual_stats,a['actual_stats'][i]);stats_original.append(raw_stats);stats_actual.append(actual_stats)
   details=mix[i]['supcon_details'];epoch=i//222+1;require(len(details)==(4 if epoch>5 else 0),'Original SupCon terms/schedule')
   if details:
    require({(v['phase'],v['level']) for v in details}=={(p,l) for p in ('first','second') for l in ('time_down','t2.pool')},'Original shallow/middle/ordinary two-pass scope')
    for v in details:
     for k in ('valid_anchors','total_anchors','positive_min','positive_max'):require(v[k]==actual_stats[k],'SupCon used wrong actual L labels/order')
    require(math.isclose(math.fsum(v['loss'] for v in details)/4,mix[i]['supcon'],rel_tol=1e-5,abs_tol=1e-6),'Saved SupCon layer mean differs')
 require(original_all==actual_all and seen_L==expected_L and len(actual_all)==5040 and sum(actual_all.values())==568320,'Full unchanged physical L exposure')
 require(moved==a['moved_positions'] and (moved==0 if arm=='O0' else moved>0),'Actual moved-position count')
 require(a['maximum_buffer_L_packets']==768 and a['buffer_CPU_seconds']>=0,'Actual buffer CPU cost required')
 def means(rows):return {k:math.fsum(r[k] for r in rows)/len(rows) for k in rows[0]}
 by_epoch=[]
 for e in range(20):by_epoch.append(dict(epoch=e+1,original=means(stats_original[e*222:(e+1)*222]),actual=means(stats_actual[e*222:(e+1)*222])))
 summary=dict(status='VERIFIED',all4440_actual_inputs_checked=True,existing_SupCon_mask_terms_checked=13320,all740_physical_block_multisets_identical=True,global_L_frequency_identical=True,unique_physical_L=5040,L_observations=568320,U_observations=396368,unique_U_without_TX_resolution=45360,U_TX_resolutions=0,moved_positions=moved,original_stats_mean=means(stats_original),actual_stats_mean=means(stats_actual),buffer_CPU_seconds=a['buffer_CPU_seconds'],maximum_logical_buffer_packets=768,private_L_generators=g,metadata_only=True,new_NN=0,source_IQ_reads=0,checkpoint_loads=0,target_access=False)
 return summary,by_epoch,original_L,U
