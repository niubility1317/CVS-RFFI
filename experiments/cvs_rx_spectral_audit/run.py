import os,sys,time
from pathlib import Path
import torch
from . import design as d
from .spectral import observations,grid,audit

def load_l(c):
 from experiments.cvs_phase1_stack.recover import verify_contract
 preparation=Path(c['preparation']);done=d.read(preparation/'preparation_complete.json')
 if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['target_access'] or done['identity_updates']!=0:raise ValueError('Preparation provenance differs')
 packed=torch.load(preparation/'source_cache.pt',map_location='cpu',weights_only=False)
 verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(done['source_contract'],packed['contract'])
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')}
 if len(L['ids'])!=6300 or len(set(L['ids']))!=6300 or sorted(L['ids'])!=sorted(packed['contract']['role_ids']['L_s']):raise ValueError('Wrong L physical IDs')
 # Packed legacy file also contains V and old features. None are used or returned.
 return L,dict(cache=(preparation/'source_cache.pt').as_posix(),L_ids=L['ids'],source_contract=packed['contract'],checkpoint_loaded=False,legacy_other_roles_deserialized_but_not_used=True)

def run(row_id):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 started=time.perf_counter();torch.set_num_threads(2);L,provenance=load_l(c)
 d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),device='CPU',target_loaders_constructed=False))
 d.write(out/'provenance.json',provenance);result={}
 for window in c['observation']['windows']:
  for mode,values in observations(L['x'],window).items():
   means,halves,labels=grid(values,L['y'],L['rx'],L['day']);report=audit(means,halves);report['labels']=labels;result[window+'/'+mode]=report
   print(window,mode,'relative_MSE',report['relative_MSE'],'positive_folds',report['folds_better_than_zero'],flush=True)
 d.write(out/'spectral_crossfit.json',dict(observations=result,primary=c['observation']['primary'],V_used=False,U_used=False,target_access=False))
 done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,identity_updates=0,target_access=False,checkpoint_loaded=False,V_used=False,U_used=False,L_count=len(L['ids']),elapsed_seconds=time.perf_counter()-started)
 d.write(out/'completion.json',done);print(done,flush=True)
