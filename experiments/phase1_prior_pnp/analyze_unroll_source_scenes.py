"""Post-hoc source cell breakdown; input contains aggregates, not TX labels."""
import argparse,json
from pathlib import Path

VIEWS=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')

def analyze(summary):
 if summary['status']!='SOURCE_VALIDATION_COMPLETE' or not summary['source_only'] or summary['DURA_in_candidate'] or any(summary[k]!=0 for k in ('TX_labels_read','U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source-only validation required')
 cells=summary['validation']['1.0']['0.8'];out={}
 for name,part,receivers in (('RX4_6',1,(4,6)),('RX8',2,(8,))):
  scenarios={}
  for view in VIEWS:
   rows=[cells[f'{part}:{rx}:{view}'] for rx in receivers];count=sum(x['count'] for x in rows)
   if not count or any(not x['valid_count'] for x in rows):raise ValueError('Complete source cells required')
   mean=lambda field:sum(x['count']*x[field] for x in rows)/count
   par=sum(x['parent_correct'] for x in rows);candidate=sum(x['candidate_correct'] for x in rows);base=mean('parent_iq');corrected=mean('candidate_iq')
   if base<=0:raise ValueError('Positive parent IQ error required')
   scenarios[view]=dict(count=count,trusted=sum(x['valid_count'] for x in rows),parent=par,candidate=candidate,dura=sum(x['baseline_correct'] for x in rows),gain_vs_parent=candidate-par,parent_iq=base,candidate_iq=corrected,iq_change_percent=(corrected/base-1)*100)
  out[name]=scenarios
 return dict(status='SOURCE_SCENE_BREAKDOWN',source_only=True,pseudo_target_not_truth=True,DURA_comparator_only=True,scale=1.0,gate=.8,iq_metric='count-weighted mean of per-cell aligned IQ normalized error, not a pooled waveform estimate',groups=out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(json.loads(a.summary.read_text(encoding='utf8'))),allow_nan=False,indent=2))
if __name__=='__main__':main()
