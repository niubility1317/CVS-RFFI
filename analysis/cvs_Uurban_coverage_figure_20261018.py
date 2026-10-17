"""Replot one source coverage figure from completed analysis;zero model forwards."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ARMS=('B0','URAW','UAS');RX=(1,3,4,6,8);SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve existing figures;use new output root')
 data=json.loads((a.analysis/'analysis_summary.json').read_text())
 if data['status']!='ANALYZED' or set(data['rows'])!=set(ARMS):raise ValueError('Completed same-row analysis required')
 matrices={}
 for arm in ARMS:
  strata=[v for segment in data['rows'][arm]['source']['L_support'] for v in segment['strata']]
  matrices[arm]=np.array([[sum(v['count'] for v in strata if v['RX']==rx and v['scene']==scene) for scene in SCENES] for rx in RX])
  if matrices[arm].sum()!=426240 or (matrices[arm]<=0).any():raise ValueError('Actual L coverage incomplete')
 lo=min(m.min() for m in matrices.values());hi=max(m.max() for m in matrices.values());a.output.mkdir()
 fig,axes=plt.subplots(1,3,figsize=(14,4.6))
 for arm,ax in zip(ARMS,axes):
  m=matrices[arm];im=ax.imshow(m,cmap='Blues',vmin=lo,vmax=hi,aspect='auto');ax.set(xticks=range(3),xticklabels=['Low sub','Mid urban','Low urban'],yticks=range(5),yticklabels=['RX'+str(rx) for rx in RX],title=arm+' labeled full As')
  for j in range(5):
   for k in range(3):
    luminance=sum(w*c for w,c in zip((.2126,.7152,.0722),im.cmap(im.norm(m[j,k]))[:3]));ax.text(k,j,str(m[j,k]),ha='center',va='center',fontsize=8,color='white' if luminance<.45 else 'black')
  fig.colorbar(im,ax=ax,fraction=.04,label='Full As observations')
 fig.suptitle('Actual L coverage / all five source RX;per-packet TX-by-state sidecar unavailable');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(a.output/('training_L_support.'+ext),dpi=190)
 plt.close(fig)
 (a.output/'complete.json').write_text(json.dumps(dict(status='FIGURE_DISPLAY_REVISED',analysis=str(a.analysis),NN=0,IQ_reads=0,checkpoint_reads=0,metrics_changed=False,original_figures_preserved=True,contrast='actual cell luminance;shared count scale'),indent=2)+'\n')
if __name__=='__main__':main()
