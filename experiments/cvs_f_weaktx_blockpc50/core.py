"""Two fixed weak-class components; block PCGrad-inspired L-feature correction."""
import torch
from torch.nn import functional as F
from experiments.cvs_f_weaktx_gradient50.core import select,blocks,PREFIX
MODES=('NORM_L_ID','BLOCK_PAIR')
def active(step):return step>1110 and step%4==0
def strength(step):return min(max((((step-1)//222+1)-5)/10.,0.),1.)
def contributions(first,second,y):
 if first.shape!=(128,6) or second.shape!=first.shape or y.shape!=(128,) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):raise ValueError('Only actual paired L128 logits/visible source TX allowed')
 ce=.5*(F.cross_entropy(first,y,reduction='none',label_smoothing=.01)+F.cross_entropy(second,y,reduction='none',label_smoothing=.01));counts=[int((y==c).sum()) for c in (1,3)]
 return [ce[y==c].sum()/128 for c in (1,3)],counts
def block_update(gl,a,b,eta,mode):
 if mode not in MODES or not 0<=eta<=1 or gl.ndim!=1 or a.shape!=gl.shape or b.shape!=gl.shape or not all(bool(torch.isfinite(v).all()) for v in (gl,a,b)):raise ValueError('Finite matching block gradients and fixed mode/strength required')
 gl,a,b=(v.double() for v in (gl,a,b));aa=a@a;bb=b@b;dot=a@b;ll=gl@gl;conflict=bool(dot<0 and aa>0 and bb>0);zero=bool(ll==0);h=a+b;change=torch.zeros_like(gl)
 if conflict and not zero:change=eta*(-dot/bb*b-dot/aa*a)
 target=gl+change;gain=target.norm()/gl.norm() if not zero else gl.new_tensor(1.);control=gl*gain;selected=change if mode=='BLOCK_PAIR' else control-gl
 # Both candidate directions are rounded to the actual leaf dtype downstream.
 info=dict(conflict=conflict,zero_main=zero,weak_cosine=float(dot/(aa*bb).sqrt()) if aa>0 and bb>0 else None,L_before_norm=float(gl.norm()),L_PC_target_norm=float(target.norm()),L_control_norm=float(control.norm()),norm_gain=float(gain),delta_norm=float(selected.norm()),strength=eta)
 if not all(bool(torch.isfinite(v).all()) for v in (selected,target,control)):raise FloatingPointError('Nonfinite block gradient correction')
 return selected,info
def corrected(parameters,gradients,eta,mode):
 if len(gradients)!=3 or any(len(v)!=len(parameters) for v in gradients):raise ValueError('Exactly Lidentity,weak1,weak3 feature VJPs required')
 offsets=[];offset=0
 for name,p in parameters:offsets.append((offset,offset+p.numel()));offset+=p.numel()
 rows=[]
 for values in gradients:rows.append(torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1) for (name,p),g in zip(parameters,values)]).double())
 result=[None]*len(parameters);details=[]
 for group in blocks(parameters):
  indices=[i for i,(lo,hi) in enumerate(offsets) if lo>=group['start'] and hi<=group['end']];lo,hi=group['start'],group['end'];gl,a,b=[v[lo:hi] for v in rows]
  if group['name']=='core/sinc' and {parameters[i][0].split('.')[-1] for i in indices}!={'low_coordinate','band_coordinate'}:raise ValueError('Sinc block must contain only common-scale q leaves')
  if any(gradients[0][i] is None and (gradients[1][i] is not None or gradients[2][i] is not None) for i in indices):raise ValueError('Weak CE feature graph outside Lidentity graph')
  delta,info=block_update(gl,a,b,eta,mode);pc,_=block_update(gl,a,b,eta,'BLOCK_PAIR');control,_=block_update(gl,a,b,eta,'NORM_L_ID');pc_cast=[];control_cast=[]
  for i in indices:
   left,right=offsets[i];left-=lo;right-=lo;p=parameters[i][1];pc_cast.append(pc[left:right].to(p.dtype).double());control_cast.append(control[left:right].to(p.dtype).double());part=delta[left:right].to(p.dtype).reshape_as(p)
   if gradients[0][i] is not None and bool(part.count_nonzero()):result[i]=part.detach()
  if any(not bool(torch.isfinite(v).all()) for v in (*pc_cast,*control_cast)):raise FloatingPointError('Nonfinite leaf-dtype gradient correction')
  pn=(gl+torch.cat(pc_cast)).norm();cn=(gl+torch.cat(control_cast)).norm();error=float((pn-cn).abs()/torch.maximum(gl.norm(),pn).clamp_min(1e-30))
  if error>1e-5:raise FloatingPointError('Float32 matched Lidentity component norm mismatch')
  details.append(dict(block=group['name'],parameters=len(indices),inactive_parameters=sum(gradients[0][i] is None for i in indices),fp32_norm_match_error=error,**info))
 return result,dict(blocks=details,conflicted_blocks=sum(v['conflict'] for v in details),nonzero_blocks=sum(v['delta_norm']>0 for v in details),maximum_norm_match_error=max(v['fp32_norm_match_error'] for v in details))
def surrogate(parameters,deltas):
 values=[((p-p.detach())*g).sum() for (_,p),g in zip(parameters,deltas) if g is not None]
 return torch.stack(values).sum() if values else parameters[0][1].new_zeros(())
