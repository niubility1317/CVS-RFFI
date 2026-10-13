"""Independent CPU error-pair counts after all registered predictions are fixed."""
import argparse,json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
REMOTE=r'''
import sys,json,time
from pathlib import Path
import numpy as np
c=CONFIG;root=Path(c['project'])/'releases'/c['release'];sys.path[:0]=[str(root),str(root/'code')]
from importlib import import_module
d=import_module(c['package']+'.design');preflight=import_module(c['package']+'.predict').preflight
predictions={};ids=None
for row in d.rows():
 cfg,current=preflight(row['row_id'])
 if ids is None:ids=current
 if not np.array_equal(ids,current):raise ValueError('Paired physical IDs differ')
 with np.load(d.BASE/row['row_id']/'prediction/predictions.npz',allow_pickle=False) as p:predictions[row['arm'],row['model_seed']]=p['NATIVE_HEAD'].copy()
# No truth connection occurs before the complete prediction loop above.
from experiments.cvs_report_ssl.evaluate import truth_arrays
y,rx,days=truth_arrays(d.read(d.TRUTH),ids)
records=[]
for row in d.rows():
 arm,seed=row['arm'],row['model_seed'];base=predictions['BASE_REF',seed];new=predictions[arm,seed];bc=base==y;nc=new==y
 groups=[('overall','ALL',np.ones(len(y),dtype=bool))]
 groups+=[('receiver',str(v),rx==v) for v in np.unique(rx)]+[('day',str(v),days==v) for v in np.unique(days)]+[('transmitter',str(v),y==v) for v in np.unique(y)]
 for dim,stratum,mask in groups:
  corrected=int((mask&~bc&nc).sum());introduced=int((mask&bc&~nc).sum());count=int(mask.sum());delta=float(nc[mask].mean()-bc[mask].mean())
  if abs((corrected-introduced)/count-delta)>1e-12:raise ValueError('Paired error net differs from accuracy delta')
  records.append(dict(row_id=row['row_id'],arm=arm,model_seed=seed,control='BASE_REF',readout='NATIVE_HEAD',view='clean',dimension=dim,stratum=stratum,count=count,corrected=corrected,introduced=introduced,net_corrected=corrected-introduced,changed_decisions=int((mask&(base!=new)).sum()),accuracy_difference=delta))
print(json.dumps(dict(status='VERIFIED',run_id=d.RUN,records=records,registered_prediction_count=len(d.rows())*168000,all_predictions_fixed_before_truth=True,independent_CPU_score=True,training_updates=0,feedback_to_training=False,read_at=time.time())))
'''
def score(package,output):
 from importlib import import_module
 d=import_module(package+'.design');cfg=dict(project=d.PROJECT,release=d.RELEASE,package=package)
 payload=REMOTE.replace('CONFIG',repr(cfg));wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES=''))\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
 v=json.loads(ssh(wrapper));path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=v['status'],records=len(v['records']),registered_prediction_count=v['registered_prediction_count'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--package',required=True,choices=['experiments.cvs_feature_sequence50','experiments.cvs_time_factorial50']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();score(a.package,a.output)
