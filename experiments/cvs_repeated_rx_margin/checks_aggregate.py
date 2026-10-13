"""Full aggregate regression using synthetic already-scored records only."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d
from .score import finish
from experiments.cvs_report_ssl.evaluate import metric_rows,HELDOUT_RX,TARGET_DAYS
def run():
 cells=[(tx,rx,day) for tx in range(6) for rx in HELDOUT_RX for day in TARGET_DAYS];y=np.array([a for a,b,c in cells]);rx=np.array([b for a,b,c in cells]);day=np.array([c for a,b,c in cells]);ids=np.array([f'synthetic-{i}' for i in range(len(y))])
 with tempfile.TemporaryDirectory() as folder:
  base=Path(folder);truth=base/'fake_truth.json';d.write(truth,{})
  with patch.object(d,'BASE',base),patch.object(d,'TRUTH',truth):
   for row in d.rows():
    c=d.config(row);p=base/row['row_id'];(p/'prediction').mkdir(parents=True);pred=y.copy();wrong=range(20) if row['arm']=='MATCHED_BASE' else range(10,25);pred[list(wrong)]=(pred[list(wrong)]+1)%6;np.savez(p/'prediction/predictions.npz',ids=ids,FINAL=pred);rr,dd=metric_rows(c,pred,ids,y,rx,day)
    for r in rr+dd:r.update(view='clean',readout='FINAL')
    d.write(p/'scores/metrics.json',dict(records=rr+dd,truth_last=True,independent_recount='VERIFIED'));d.write(p/'completion.json',dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],readouts=['FINAL'],count_per_view=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED'))
   from . import predict
   with patch.object(predict,'preflight',lambda row:(d.config(next(r for r in d.rows() if r['row_id']==row)),ids)),patch('experiments.cvs_report_ssl.evaluate.truth_arrays',lambda truth,ids:(y,rx,day)):
    a=finish()
   if len(a['records'])!=216 or len(a['summary'])!=4 or len(a['paired'])!=6 or len(a['net_reclassification'])!=162:raise AssertionError('Full synthetic aggregate matrix incomplete')
   overall=[r for r in a['net_reclassification'] if r['dimension']=='overall']
   if any(r['rescued']!=10 or r['regressed']!=5 or abs(r['net']-5/len(y))>1e-12 for r in overall) or not any(r['value']=='1-1' for r in a['net_reclassification']) or not any(r['value']=='2021_03_01' for r in a['net_reclassification']):raise AssertionError('String RX/day or net counts differ')
 return dict(status='PASS',synthetic_only=True,groups=12,metric_records=216,paired_comparisons=6,string_RX_day_preserved=True,net_rescued_regressed_verified=True,query_IQ_or_truth_read=False)
if __name__=='__main__':print(json.dumps(run()))
