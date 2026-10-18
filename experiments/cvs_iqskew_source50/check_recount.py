"""Synthetic full-size independent recount integration;plotting mocked only."""
import json,tempfile,types,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import core as c
from .recount import main as recount

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);obs={};coh={}
  for pre,count in [('L',70),('V',300)]:
   n=90*count;f=np.empty((n,4,127),float);y=[];rx=[];day=[];offset=0
   for t in range(6):
    for ri,r in enumerate(c.RX):
     for di,d in enumerate((1,2,3)):
      phi=np.resize(np.asarray([-1,1,1,-1])*np.pi/3,count)[:,None];delta=c.DELAYS[[50,60,42,66,30]][ri];theta=.2*t+.3*di+(delta+.03*t)*c.OMEGA
      f[offset:offset+count,:2]=1;f[offset:offset+count,2]=np.cos(phi+theta);f[offset:offset+count,3]=np.sin(phi+theta);offset+=count;y.extend([t]*count);rx.extend([r]*count);day.extend([d]*count)
   y,rx,day=map(np.asarray,(y,rx,day));obs.update({pre+'_features':f,pre+'_y':y,pre+'_rx':rx,pre+'_day':day,pre+'_ids':np.asarray([pre+str(j) for j in range(n)])});coh[pre]=c.cells(f,y,rx,day,count)
   if pre=='L':
    for h in (0,1):coh['L'+str(h)]=c.cells(f,y,rx,day,count,h)
  result,fitted=c.evaluate(coh['L'],coh['V'],coh['L0'],coh['L1']);assert result['source_qualified']
  np.savez_compressed(root/'observations.npz',**obs,**fitted,delays=c.DELAYS,omega=c.OMEGA);(root/'result.json').write_text(json.dumps(result),encoding='utf8')
  class Dummy:
   def __getattr__(self,name):return lambda *a,**kw:None
  mpl=types.ModuleType('matplotlib');mpl.use=lambda *a:None;plt=types.ModuleType('matplotlib.pyplot');plt.subplots=lambda *a,**k:(Dummy(),[Dummy(),Dummy(),Dummy()]);plt.close=lambda *a:None;mpl.pyplot=plt
  with patch.dict(sys.modules,{'matplotlib':mpl,'matplotlib.pyplot':plt}):recount(root)
  out=json.loads((root/'independent_analysis/result.json').read_text());assert out['feature_real_values']==16916400 and out['all72_fit_indices_match'] and out['source_qualified'] and out['IQ_reads']==0
  print('PASS fullsize synthetic recount 16916400 observable values;9 unit tests separate;plotting mocked;0 real IQ/NN/query')
if __name__=='__main__':main()
