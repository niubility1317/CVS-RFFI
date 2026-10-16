"""Native zero-origin linear phase;named to avoid stdlib operator shadowing."""
import math,torch
from . import design as d
def apply(x,frequency):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or isinstance(frequency,bool) or not isinstance(frequency,(int,float)) or not math.isfinite(frequency) or abs(frequency)>d.FREQUENCY_MAX:raise ValueError('Finite float32 IQ256 and registered bounded frequency required')
 n=torch.arange(256,device=x.device,dtype=x.dtype);phase=(torch.full((len(x),1),float(frequency),device=x.device,dtype=x.dtype)*(2.*math.pi))*n
 c,s=phase.cos(),phase.sin();r,i=x.unbind(1);out=torch.stack((r*c-i*s,r*s+i*c),1)
 if not bool(torch.isfinite(out).all()):raise ValueError('Frequency view nonfinite')
 return out
def view(x,name):
 if name not in d.VIEWS or x.shape!=(32,2,256):raise ValueError('Fixed five-view32-packet matrix required')
 f=d.FREQUENCY_MAX if name in ('frequency_positive','positive_inverse') else -d.FREQUENCY_MAX if name in ('frequency_negative','negative_inverse') else 0.
 y=apply(x,f)
 if name.endswith('_inverse'):y=apply(y,-f)
 return y
