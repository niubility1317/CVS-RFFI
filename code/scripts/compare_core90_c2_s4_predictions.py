"""Read all frozen paired C2/S4 predictions, then summarize with existing truth."""
import argparse,json
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.input)
 complete=json.loads((root/'complete.json').read_text(encoding='utf-8'))
 assert complete['complete'] and {'C2','S4'}<=set(complete['rows'])
 truth={r['sample_id']:r['truth'] for r in (json.loads(l) for l in (root/'target_truth.jsonl').open(encoding='utf-8'))}
 paired=defaultdict(Counter);confusion={r:defaultdict(lambda:[[0]*6 for _ in range(6)]) for r in ['C2','S4']}
 with (root/'C2_predictions.jsonl').open(encoding='utf-8') as ca,(root/'S4_predictions.jsonl').open(encoding='utf-8') as sa:
  for lc,ls in zip_longest(ca,sa):
   assert lc is not None and ls is not None
   c,s=json.loads(lc),json.loads(ls)
   assert all(c[k]==s[k] for k in ['sample_id','scene','rx_i','day_i'])
   y=truth[c['sample_id']];cp,sp=c['prediction'],s['prediction'];scene=c['scene']
   key='both_correct' if cp==sp==y else 'C2_only_correct' if cp==y else 'S4_only_correct' if sp==y else 'both_wrong'
   paired[scene][key]+=1;paired[scene]['total']+=1;paired[scene]['prediction_disagreed']+=int(cp!=sp)
   confusion['C2'][scene][y][cp]+=1;confusion['S4'][scene][y][sp]+=1
 assert len(paired)==4 and all(v['total']==len(truth) for v in paired.values())
 Path(a.output).write_text(json.dumps(dict(paired=dict(paired),confusion=confusion),indent=2),encoding='utf-8')

if __name__=='__main__':main()
