import copy,json,torch
from . import design as d
from .probe import objects,select,dropout_replay,descriptors,statistics,smoke
def checks():
 torch.set_num_threads(2);torch.manual_seed(319);negatives=0
 from experiments.cvs_moment_area50.model import build
 model=build(d.source_config(),'cpu').eval();s=smoke(model);_,block,pool=objects(model)
 h=torch.randn(720,96,32);clean=descriptors(h,pool);samples={k:[] for k in clean}
 for seed in d.MASK_SEEDS:
  noise=dropout_replay(h,seed);assert torch.equal(noise,dropout_replay(h,seed))
  for k,v in descriptors(noise,pool).items():samples[k].append(v)
 rx=torch.tensor([v for v in d.RX for _ in range(144)]);day=torch.tensor([v for v in d.DAYS for _ in range(240)]);samples={k:torch.stack(v) for k,v in samples.items()}
 r=statistics(clean,samples,rx,day);assert len(r)==63 and all(x['noise_RMS']>0 for x in r)
 exact=statistics(clean,{k:v[None].expand(8,*v.shape) for k,v in clean.items()},rx,day);assert all(x['noise_RMS']==0 and x['mask_mean_bias_RMS']==0 for x in exact)
 for role in ('U_s','target','V'):
  try:select({},role)
  except ValueError:negatives+=1
  else:raise AssertionError('Unauthorized source role accepted')
 for q in [h[:,:95],h*float('nan')]:
  try:dropout_replay(q,d.MASK_SEEDS[0])
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid intermediate accepted')
 try:dropout_replay(h,d.MASK_SEEDS[0],.2)
 except ValueError:negatives+=1
 else:raise AssertionError('Unregistered probability accepted')
 try:objects(model.train())
 except ValueError:negatives+=1
 else:raise AssertionError('Train-mode observer accepted')
 bad=copy.deepcopy(d.config(d.rows()[0]));bad['mask_samples']=9
 try:d.validate(bad)
 except ValueError:negatives+=1
 else:raise AssertionError('Changed fixed config accepted')
 from .publish import configure,INSPECT
 from experiments.cvs_report_ssl import publish as transport
 configure();compile(transport.REMOTE.replace('CONFIG',repr({'project':d.PROJECT,'release':d.RELEASE,'run':d.RUN,'archive':'x','sha256':'x','commit':'x'})),'REMOTE','exec');compile(INSPECT.replace('CONFIG',repr({'project':d.PROJECT,'release':d.RELEASE,'run':d.RUN})),'INSPECT','exec')
 assert 'experiments/cvs_moment_area50' in transport.PATHS
 return dict(status='PASS',synthetic_only=True,protocol_negative_checks=negatives,native_zeroIQ_smoke=s,fixed_descriptor_counts=len(r),replay_deterministic=True,exact_replay_zero_error=True,remote_scripts_compile=True,no_source_or_target_IQ_read=True)
if __name__=='__main__':print(json.dumps(checks()))
