"""Source-only actual native realizations,not file hashes or a release gate."""
import argparse,gzip,json,time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
RUN='20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01'
REFERENCE='20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01'
def compare(actual,reference,expected_steps=6660):
 count=pairs=equal=0;fields=Counter();examples=[]
 with gzip.open(actual,'rt',encoding='utf8') as aa,gzip.open(reference,'rt',encoding='utf8') as bb:
  for i,(left,right) in enumerate(zip_longest(aa,bb),1):
   if left is None or right is None:raise ValueError('Both complete native physics logs required')
   a=json.loads(left);b=json.loads(right);positions=[j for j in range(128) if (j+i)%2==1]
   if a['step']!=i or b['step']!=i or len(a['records'])!=64 or len(b['records'])!=128 or a['positions']!=positions:raise ValueError('Ordered64/128 physical correspondence required')
   count+=1
   if a['scene']!=b['scene']:fields['scene']+=64
   for j,x in zip(positions,a['records']):
    y=b['records'][j];pairs+=1;diff=[k for k in sorted(set(x)|set(y)) if x.get(k)!=y.get(k)]
    if not diff and a['scene']==b['scene']:equal+=1
    fields.update(diff)
    if diff and len(examples)<10:examples.append(dict(step=i,position=j,fields=diff,sample_id_R25=x.get('sample_id'),sample_id_AS50=y.get('sample_id')))
 if count!=expected_steps or pairs!=expected_steps*64:raise ValueError('Complete fixed training realization budget required')
 return dict(status='SOURCE_REALIZATION_PAIR_RECOUNT_COMPLETE',steps=count,pairs=pairs,identical_records=equal,record_mismatches=pairs-equal,field_mismatches=dict(fields),first_mismatch_examples=examples,all_selected_realizations_identical=equal==pairs,NN=0,IQ_reads=0,CP_reads=0,truth_reads=0,new_updates=0,scientific_comparison_not_a_release_gate=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-worker',type=int,required=True);p.add_argument('--commit',required=True);a=p.parse_args();B=a.project/'runs'/RUN;source=B/'R25-s2026092701/source';tic=time.perf_counter()
 while not (source/'result.json').is_file():
  if (B/'failure.json').is_file():raise RuntimeError('Owned run failure;preserve all outputs')
  proc=Path('/proc')/str(a.wait_worker)
  if not proc.is_dir() or 'cvs_cleanretain37/runner.py' not in (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode():raise RuntimeError('Original owned worker missing;reconcile,no restart')
  time.sleep(15)
 started=time.perf_counter();r=compare(source/'training_physics.jsonl.gz',a.project/'runs'/REFERENCE/'B01-s2026092701/source/training_physics.jsonl.gz');r.update(code_commit=a.commit,wall_seconds_including_wait=time.perf_counter()-tic,analysis_wall_seconds=time.perf_counter()-started)
 if a.output.exists():raise FileExistsError('No output overwrite')
 a.output.mkdir();(a.output/'complete.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(r))
if __name__=='__main__':main()
