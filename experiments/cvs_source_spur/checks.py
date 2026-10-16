import json
import numpy as np
from . import design as d
from .core import periodograms,cell_spectra,prominence,fit,choose,analyze

def run():
 rng=np.random.default_rng(20261017811)
 # Analytic cell spectra with exactly one strong RX-specific nonDC peak.
 a=np.ones((2,6,3,4,256));a[:,:,:,0,180]=100.;a[:,:,:,1,70]=10.
 good,p=analyze(a);assert good['status']=='PASS_SOURCE_ASSOCIATION_ONLY' and good['successful_folds']==18 and good['modal_count']==18
 changed=p.copy();changed[:,2]=rng.normal(size=changed[:,2].shape);changed[:,:,1]=rng.normal(size=changed[:,:,1].shape)
 assert np.array_equal(fit(p,2,1),fit(changed,2,1)) and choose(fit(p,2,1))==choose(fit(changed,2,1))
 common=np.ones_like(a);common[:,:,:,:,180]=100.;bad,_=analyze(common);assert bad['status']=='FAIL' and not bad['checks']['risk']
 entangled=a.copy();entangled[:,3:]=a[:,3:,:,::-1];bad,_=analyze(entangled);assert bad['status']=='FAIL'
 no_line=np.ones_like(a);bad,_=analyze(no_line);assert bad['status']=='FAIL' and not bad['checks']['successful_folds']
 # Full source-shaped synthetic waveforms: no real IQ or neural network.
 rows=[(t,rr,dd) for t in d.TX for dd in d.DAYS for rr in d.FIT for _ in range(70)];tx,rx,day=np.asarray(rows).T
 z=(rng.normal(size=(5040,256))+1j*rng.normal(size=(5040,256)))/np.sqrt(2)
 phase=rng.uniform(0,2*np.pi,(5040,1));n=np.arange(256)
 z[rx==3]+=2*np.exp(1j*(2*np.pi*52*n/256+phase[rx==3]));z[rx==4]+=.5*np.exp(1j*(2*np.pi*(-58)*n/256+phase[rx==4]))
 x=np.stack((z.real,z.imag),axis=1);spectra=periodograms(x);actual=cell_spectra(spectra,tx,rx,day);wave,_=analyze(actual)
 assert wave['status']=='PASS_SOURCE_ASSOCIATION_ONLY'
 scaled=periodograms(x[:32]*rng.uniform(.2,3,(32,1,1)));assert np.allclose(scaled,spectra[:,:32],rtol=1e-12,atol=1e-12)
 perm=np.roll(actual,1,axis=3);perms,_=analyze(perm);assert abs(perms['windows'][0]['risk_reduction']-wave['windows'][0]['risk_reduction'])<1e-12
 # An exact FFT bin tone: its observed peak and amplitude are predictable.
 pure=np.exp(2j*np.pi*52*n/256)[None];pp=periodograms(np.stack((pure.real,pure.imag),axis=1));assert abs(pp[0,0,180]-256)<1e-10 and pp[0,0,np.arange(256)!=180].max()<1e-20
 negatives=0
 def reject(call):
  nonlocal negatives
  try:call()
  except ValueError:negatives+=1;return
  raise AssertionError('Invalid input accepted')
 for xx in (np.zeros((1,2,256)),x[:1,:,:255],x[:1]*np.nan):reject(lambda:periodograms(xx))
 for args in ((spectra[:,:-1],tx[:-1],rx[:-1],day[:-1]),(spectra,tx,np.where(rx==3,1,rx),day),(spectra,tx-1,rx,day),(spectra,tx,rx,day-1),(spectra*np.nan,tx,rx,day)):
  reject(lambda:cell_spectra(*args))
 for aa in (a[:1],a*np.nan,-a,np.zeros_like(a)):reject(lambda:analyze(aa))
 for cc in (dict(d.config(d.rows()[0]),target_access=True),dict(d.config(d.rows()[0]),checkpoint_loads=1),dict(d.config(d.rows()[0]),model_forwards=2),dict(d.config(d.rows()[0]),U_used=True),dict(d.config(d.rows()[0]),source_RX=[1,3,4,6]),dict(d.config(d.rows()[0]),criteria={})):reject(lambda:d.validate(cc))
 return dict(status='PASS',protocol_negative_tests=negatives,analytic_RX_line_PASS=True,common_TX_entangled_no_line_FAIL=True,held_TX_AND_day_selection_isolation=True,same_fit_bin_both_windows=True,complete_synthetic_packets=5040,additional_synthetic_waveform_evaluations=33,source_cells=72,FFT_bin_oracle=True,amplitude_invariance=True,RX_permutation_equivariance=True,real_IQ=0,model_forwards=0,checkpoint_loads=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
