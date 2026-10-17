"""Compare recipes without changing native argument values."""
import math

def argument_differences(a,b,ignored):
 missing=object()
 def equal(x,y):
  if type(x) is float and type(y) is float and math.isnan(x) and math.isnan(y):return True
  return x==y
 return [k for k in sorted(set(a)|set(b)) if k not in ignored and not equal(a.get(k,missing),b.get(k,missing))]
