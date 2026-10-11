"""Route each preregistered statistic to its own fixed source calibration."""
def calibrate(c,model,L,device,out):
 if c['multilevel']['statistic']=='margin':
  from .multilevel import calibrate as run
 else:
  from experiments.cvs_e20_cross_style.multilevel import calibrate as run
 return run(c,model,L,device,out)
