"""Resolve only this run's own saved final or its declared cached parent."""
from pathlib import Path
def load(path,device,baseline,final,parent,original):
 if str(path)==str(baseline):return parent
 if Path(path)==Path(final):return original(path,device)
 raise ValueError('Unexpected checkpoint dependency')
