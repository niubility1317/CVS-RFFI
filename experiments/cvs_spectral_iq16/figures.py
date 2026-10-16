"""All fixed source views;scientific export from independently recounted data."""
import json
import numpy as np
from . import design as d
OUT=d.ROOT/d.LOCAL/'final'
def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 a=d.read(OUT/'analysis_summary.json');names=[v['view'] for v in d.views()[1:]];labels=[n.replace('structured_RX','S ').replace('control_RX','C ').replace('_minus','−').replace('_plus','+').replace('_','/') for n in names];colors=['#306b9c' if n.startswith('structured') else '#bb7752' for n in names];x=np.arange(24)
 fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained')
 for ax,key,title,threshold in ((axes[0],'accuracy_drop_pp','Frozen default-head fit-L accuracy loss (pp)',.5),(axes[1],'finite_PSD_relative_RMSE','Actual finite-IQ PSD relative RMSE vs fixed circular DSP reference',.35),(axes[2],'boundary_error_fraction','Boundary discrepancy energy / circular FIR change energy',.25)):
  values=[a['identity_drops'][n][key] if key=='accuracy_drop_pp' else a['diagnostics'][n][key] for n in names];ax.bar(x,values,color=colors);ax.axhline(threshold,color='#666666',ls='--',label='Preregistered threshold');ax.set(xticks=x,xticklabels=labels,title=title);ax.tick_params(axis='x',rotation=65);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('F+CROSS_RX / fit RX3,4,6,8 / all25 views / zero updates\nSource IQ feasibility '+a['decision']+'; not Ag efficacy or held-RX/target performance');fig.savefig(OUT/'spectral_IQ_feasibility.png',dpi=180);plt.close(fig)
 bank=d.read(OUT/'kernel_bank.json');fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');freq=np.linspace(-.5,.5,d.GRID,endpoint=False)
 for ax,kind in zip(axes,('structured','control')):
  for name in names:
   if not name.startswith(kind):continue
   k=bank[name];h=np.asarray(k['h_real'])+1j*np.asarray(k['h_imag']);p=np.zeros(d.GRID,dtype=complex);p[np.arange(-16,17)%d.GRID]=h;gain=20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(p))));ax.plot(freq[::8],gain[::8],lw=.8,alpha=.8)
  ax.axhline(.5,color='#777777',ls='--');ax.axhline(-.5,color='#777777',ls='--');ax.set(title=kind+' / all12 fixed kernels',xlabel='Normalized frequency (cycles/sample)',ylabel='Actual FIR amplitude response before RMS (dB)',ylim=(-.55,.55))
 fig.suptitle('Kernel response and energy controls pass; finite-IQ PSD fidelity fails');fig.savefig(OUT/'spectral_kernel_responses.png',dpi=180);plt.close(fig);print('Saved two complete source figures')
if __name__=='__main__':main()
