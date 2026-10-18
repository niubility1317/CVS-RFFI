"""Full-size synthetic integration,plotting mocked only;no real IQ."""
import json,tempfile,types,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .checks import Checks
from .core import RX,load_observations,evaluate
from .recount import main as recount

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);obs={};L,V,_,_=Checks().fixture()
  for pre,count,values in [('L',70,L),('V',300,V)]:
   f=[];y=[];rx=[];day=[]
   for t in range(6):
    for ri,r in enumerate(RX):
     for di,d in enumerate([1,2,3]):
      a=np.empty((count,4,127));a[:,0]=10**(values[t,ri,di]/20);a[:,1]=10**(-values[t,ri,di]/20);a[:,2]=1;a[:,3]=0;f.append(a);y.extend([t]*count);rx.extend([r]*count);day.extend([d]*count)
   f=np.concatenate(f);n=len(f);obs.update({pre+'_features':f,pre+'_y':np.asarray(y),pre+'_rx':np.asarray(rx),pre+'_day':np.asarray(day),pre+'_ids':np.asarray([pre+str(i) for i in range(n)])})
  source=root/'observations.npz';np.savez_compressed(source,**obs);cfg=root/'input_config.json';cfg.write_text(json.dumps(dict(source_roles='EXACT_MATCH',config=dict(L=6300,V=27000,U_used=False))))
  o=load_observations(source,cfg);result,fitted=evaluate(o['L'],o['V'],o['L0'],o['L1']);assert result['source_qualified'];np.savez_compressed(root/'fitted_statistics.npz',**fitted);(root/'result.json').write_text(json.dumps(result));(root/'resolved_config.json').write_text(json.dumps({'config':{'input_statistics':str(source)}}))
  class Dummy:
   def __getattr__(self,name):return lambda *a,**kw:None
  mpl=types.ModuleType('matplotlib');mpl.use=lambda *a:None;plt=types.ModuleType('matplotlib.pyplot');plt.subplots=lambda *a,**k:(Dummy(),[Dummy(),Dummy(),Dummy()]);plt.close=lambda *a:None;mpl.pyplot=plt
  with patch.dict(sys.modules,{'matplotlib':mpl,'matplotlib.pyplot':plt}):recount(root)
  q=json.loads((root/'independent_analysis/result.json').read_text());assert q['source_derived_values_read']==16916400 and q['all72_fit_indices_match'] and q['source_qualified'] and q['IQ_reads']==0
  print('PASS 16916400 synthetic derivedvalues/full90cells/72independentfits/0realIQ/NN/target;onlyplotmocked')
if __name__=='__main__':main()
