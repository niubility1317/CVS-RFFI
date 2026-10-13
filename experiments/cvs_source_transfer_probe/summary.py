import statistics as st
from . import design as d
def finish():
 result=[];details=[]
 for r in d.rows():
  done=d.read(d.BASE/r['row_id']/'completion.json')
  if done['config']!=d.config(r) or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['target_access'] or done['feature_updates'] or not done['old_class_weights_absent'] or not done['feature_parameters_buffers_unchanged'] or len(done['records'])!=33:raise ValueError('Incomplete feature probe')
  details.append(done)
 for arm in d.ARMS:
  models=[v for v in details if v['config']['arm']==arm]
  if len(models)!=3:raise ValueError('Incomplete three-seed source probe')
  for readout in ('ORIGINAL_HEAD','FRESH_COSINE_ALL_L','COSINE_PROTO_ALL_L','FRESH_COSINE_CROSS_RX_K'):
   for k in d.K_VALUES if readout=='FRESH_COSINE_CROSS_RX_K' else (None,):
    a=[st.mean(r['accuracy'] for r in m['records'] if r['readout']==readout and r.get('k')==k) for m in models]
    result.append(dict(arm=arm,readout=readout,k=k,accuracy_mean=st.mean(a),accuracy_sd=st.stdev(a),seed_accuracy={str(m['config']['model_seed']):v for m,v in zip(models,a)}))
 value=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,models=9,feature_updates=0,target_access=False,readouts=297,head_fits=279,summary=result,records=details,claim='Source rehead evidence only; no Phase2/new-TX result')
 d.write(d.BASE/'analysis.json',value);d.write(d.BASE/'completion.json',{k:v for k,v in value.items() if k not in ('summary','records')});return value
