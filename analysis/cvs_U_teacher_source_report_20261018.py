"""Independent source prediction recount; never loads models, U or target data."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def softmax(x):
 z=x.astype(np.float64);z-=z.max(1,keepdims=True);v=np.exp(z);return v/v.sum(1,keepdims=True)
def cm(y,p):return np.bincount(6*y+p,minlength=36).reshape(6,6).tolist()
def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(exist_ok=False);terminal=read(root/'complete.json')
 if terminal['status']!='ANALYZED' or terminal['training_updates'] or terminal['target_access'] or terminal['U_truth_read'] or terminal['U_records_read'] or terminal['actual_new_NN']!=28808:raise ValueError('Only completed source-only diagnostic')
 rows=[];matrices={};ids0=None;labels0=None;passed=[]
 for endpoint in ['E20','E30']:
  z=np.load(root/endpoint/'predictions.npz',allow_pickle=False);y=z['TX'];rx=z['RX'];t=z['teacher_clean'];s=z['student_clean'];a=z['student_satellite'];ids=z['ids'];proof=read(root/endpoint/'complete.json')
  if len(set(ids.tolist()))!=1800 or y.shape!=(1800,) or t.shape!=(1800,6) or s.shape!=t.shape or a.shape!=(2,3,1800,6) or not all(np.isfinite(v).all() for v in [t,s,a]):raise ValueError('Invalid source prediction dimensions')
  if ids0 is not None and (not np.array_equal(ids,ids0) or not np.array_equal(y,labels0)):raise ValueError('Source panels differ between fixed checkpoints')
  ids0=ids;labels0=y;p=softmax(t);conf=p.max(1);q=np.maximum(0,np.minimum(1,(conf-.95)/.05));mask=q>0;correct=p.argmax(1)==y
  if not np.allclose(q,z['q'],atol=1e-12,rtol=1e-12):raise ValueError('Confidence weights differ')
  coverage=float(mask.mean());precision=float(correct[mask].mean());perRX={str(r):float(correct[mask&(rx==r)].mean()) if (mask&(rx==r)).any() else None for r in [1,3,4,6,8]};opp=[]
  matrices[endpoint]={'teacher_clean':cm(y,p.argmax(1)),'student_clean':cm(y,s.argmax(1))}
  for draw in range(2):
   dd=[]
   for si,scene in enumerate(['practical_low_suburban','practical_mid_urban','practical_low_urban']):
    pred=a[draw,si].argmax(1);op=float((q*correct*(pred!=y)).sum()/q.sum());dd.append(op);matrices[endpoint][str(draw)+'/'+scene]=cm(y,pred)
    rows.append(dict(endpoint=endpoint,draw=draw,scene=scene,teacher_coverage=coverage,teacher_selected=int(mask.sum()),teacher_errors=int((mask&~correct).sum()),teacher_selected_precision=precision,teacher_q_weighted_precision=float((q*correct).sum()/q.sum()),student_accuracy=float((pred==y).mean()),q_weighted_correct_teacher_wrong_student=op,q_weighted_wrong_teacher_correct_student=float((q*~correct*(pred==y)).sum()/q.sum())))
   opp.append(dd)
  checks=[precision>=.99,coverage>=.75,all(v is not None and v>=.98 for v in perRX.values()),bool((np.asarray(opp)[:,1:]>=.10).all())]
  if not np.allclose(opp,proof['transfer_opportunity'],atol=1e-12,rtol=1e-12) or proof['source_qualified']!=all(checks) or not np.isclose(proof['confidence_precision'],precision) or not np.isclose(proof['coverage'],coverage):raise ValueError('Independent source decision recount mismatch')
  passed.append(all(checks))
 with (out/'source_qualification.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(out/'confusion_matrices.json',matrices)
 result=dict(status='INDEPENDENT_RECOUNT_PASS',qualified=all(passed),endpoints=2,model_seeds=1,confusion_matrices=16,source_samples_each=1800,records=len(rows),NN=0,training_updates=0,target_access=False,U_access=False,plots_generated=0)
 write(out/'complete.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.root,a.output)))
