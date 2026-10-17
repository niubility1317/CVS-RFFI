"""Completed boundary row against two fixed E30 references; no IQ/CP/NN."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).with_name('cvs_fullsource_E30_results_20261017.py')
spec=importlib.util.spec_from_file_location('full_E30_results',BASE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RUN='20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01'
_original_parse=m.parse_source
_original_plots=m.plots

def boundary_evidence(result,progress):
 expected={(t,r,s,b) for t in range(6) for r in (1,3,4,6,8) for s in ('practical_low_suburban','practical_mid_urban','practical_low_urban') for b in ('reflect','edge')}
 cells=result.get('boundary_coverage',[])
 if result.get('training_boundary_counts')!=dict(reflect=426240,edge=426240):raise ValueError('Both full native training boundaries required')
 keys=[(r['TX'],r['RX'],r['scene'],r['boundary']) for r in cells]
 if set(keys)!=expected or len(keys)!=len(expected) or any(r['count']<=0 for r in cells):raise ValueError('All TX/RX/scene/boundary cells required')
 if any(sum(r['count'] for r in cells if r['boundary']==b)!=426240 for b in ('reflect','edge')):raise ValueError('Boundary coverage totals differ')
 rows=progress.get('records',[])
 if len(rows)!=30:raise ValueError('All30 cumulative native-boundary records required')
 for e,r in enumerate(rows,1):
  if r.get('epoch')!=e or r.get('steps')!=222*e or r.get('training_boundary_packets')!={'reflect':14208*e,'edge':14208*e}:raise ValueError('Native boundary schedule differs')
 return dict(epochs=30,updates=6660,coverage_cells=180,counts=dict(reflect=426240,edge=426240),coverage=cells,native_AGC_recomputed=True,formal_boundary='original reflect',causal_real_pre_history_identified=False)

def parse_source(root,done):
 result=_original_parse(root,done)
 if root.name=='BCTX-s2026092701':
  result['boundary_evidence']=boundary_evidence(result['source_result'],m.read(root/'source/mechanism_progress.json'))
 return result

def plots(out,results):
 files=_original_plots(out,results)
 if 'BCTX' not in results:return files
 import matplotlib.pyplot as plt
 import numpy as np
 cells=results['BCTX']['source']['boundary_evidence']['coverage']
 scenes=['practical_low_suburban','practical_mid_urban','practical_low_urban'];rxs=[1,3,4,6,8]
 fig,axes=plt.subplots(1,2,figsize=(10,4.8))
 matrices=[np.asarray([[sum(r['count'] for r in cells if r['RX']==rx and r['scene']==scene and r['boundary']==b) for scene in scenes] for rx in rxs]) for b in ('reflect','edge')]
 lo=min(x.min() for x in matrices);hi=max(x.max() for x in matrices)
 for ax,b,z in zip(axes,['reflect','edge'],matrices):
  im=ax.imshow(z,vmin=lo,vmax=hi,cmap='Blues',aspect='auto')
  ax.set(xticks=range(3),xticklabels=['Low suburban','Mid urban','Low urban'],yticks=range(5),yticklabels=['RX'+str(rx) for rx in rxs],title=b+' / 426240 packets')
  for j in range(5):
   for k in range(3):ax.text(k,j,str(z[j,k]),ha='center',va='center',color='white' if z[j,k]>lo+.6*(hi-lo) else 'black',fontsize=9)
  fig.colorbar(im,ax=ax,fraction=.045,label='Training observations')
 fig.suptitle('Training boundary support; counts are not accuracy');fig.tight_layout()
 for suffix in ('png','pdf'):
  p=out/('training_boundary_support.'+suffix);fig.savefig(p,dpi=200);files.append(str(p))
 plt.close(fig);return files

def install():
 m.ROWS=dict(m.ROWS,BCTX=(RUN,'BCTX-s2026092701'))
 m.COMPARISON_PAIRS=(('B01','B00'),('BCTX','B01'),('BCTX','B00'))
 m.parse_source=parse_source;m.plots=plots

if __name__=='__main__':install();m.main()
