"""Predeclared source-only accuracy audit; separate from primary IQ-guarded selection."""
import argparse,json,math
from pathlib import Path
from self_reference_selection import REQUIRED

def choose_accuracy(results):
 best=dict(kind='none',validation_net=0)
 for epoch,combos in results.items():
  for key,cells in combos.items():
   if set(cells)!=REQUIRED:raise ValueError('All source receiver/scene cells required')
   alpha,threshold=map(float,key.split(':'));val=[(k,v) for k,v in cells.items() if k.startswith('1:')]
   if any(v['count']<=0 or v['valid_count']<=0 or not math.isfinite(v['baseline_iq']) or not math.isfinite(v['candidate_iq']) for _,v in val):raise ValueError('Finite valid source cells required')
   net=sum(v['candidate_correct']-v['baseline_correct'] for k,v in val if not k.endswith(':clean'))
   cls=all(v['candidate_correct']>=v['baseline_correct']-max(1,math.ceil(.005*v['valid_count'])) for _,v in val)
   clean=all(v['candidate_iq']<=v['baseline_iq']+.001 for k,v in val if k.endswith(':clean'))
   ratios=[v['candidate_iq']/max(v['baseline_iq'],1e-12) for k,v in val if not k.endswith(':clean')]
   if cls and clean and net>=10 and net>best['validation_net']:best=dict(kind='self_reference_net',epoch=int(epoch),alpha=alpha,threshold=threshold,validation_net=net,max_impaired_iq_ratio=max(ratios))
 return best

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.source_selection.read_text(encoding='utf8'));result=dict(primary_selected=d['selected'],accuracy_audit=choose_accuracy(d['validation']),source_only=True,validation_receivers=[4,6],heldout_used=False,IQ_impaired_limit=False,caveat='Source pseudo-target agreement is not truth; impaired waveform quality is reported without a hard limit; no test accessed.')
 with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
