"""Keep native RDrop; compare point and differentiably centered feature consistency."""
import math
import torch
from torch.nn import functional as F
ARMS=('FULL','POINT','NORM_MATCH','CENTER')
EPS=1e-4

def new_audit():
    return dict(calls=0,packets=0,gradient_probe_calls=0,aux_positive_calls=0,auxiliary_loss_sum=0.,native_KL_sum=0.,Fisher_trace_sum=0.,
        max_common_translation_gradient_ratio=0.,max_norm_match_relative_error=0.,norm_ratio_sum=0.,norm_ratio_max=0.,
        point_gradient_norm_sum=0.,center_gradient_norm_sum=0.,point_aux_sum=0.,center_aux_sum=0.)

@torch.no_grad()
def trace(a,b,W):
    A=F.normalize(W.detach().double(),dim=1,eps=EPS)
    p=.5*(a.detach().double().softmax(1)+b.detach().double().softmax(1))
    T=900*((p*A.square().sum(1)).sum(1)-(p@A).square().sum(1)).clamp_min(0)
    if not bool(torch.isfinite(T).all()) or not bool((T<=900+1e-6).all()):raise ValueError('Finite native Fisher trace required')
    return T.to(a.dtype)

def penalty(z1,z2,T,centered):
    if centered:
        mu=.5*(z1+z2).mean(0,keepdim=True)
        z1,z2=z1-mu,z2-mu
    delta=F.normalize(z1,dim=1,eps=EPS)-F.normalize(z2,dim=1,eps=EPS)
    return .5*(T*delta.square().sum(1)/160).mean()

def boundary_geometry(point,center,z1,z2):
    # These backward traversals stop at the actual feature pair, not model parameters.
    gp=torch.autograd.grad(point,(z1,z2),retain_graph=True,create_graph=False)
    gc=torch.autograd.grad(center,(z1,z2),retain_graph=True,create_graph=False)
    np=torch.sqrt(sum(g.detach().double().square().sum() for g in gp))
    nc=torch.sqrt(sum(g.detach().double().square().sum() for g in gc))
    if not bool(torch.isfinite(np)&torch.isfinite(nc)):raise ValueError('Finite boundary gradient norms required')
    if float(np)==0 and float(nc)>0:raise ValueError('Cannot norm-match zero POINT to nonzero CENTER')
    ratio=(nc/np if float(np)>0 else np.new_tensor(1.)).to(z1.dtype).detach()
    if not bool(torch.isfinite(ratio)):raise ValueError('Finite detached matching ratio required')
    matched=torch.sqrt(sum((g.detach()*ratio).double().square().sum() for g in gp))
    relative=float(abs(matched-nc)/nc) if float(nc)>0 else float(matched)
    common=(gc[0].detach().double().sum(0)+gc[1].detach().double().sum(0)).norm()
    common_ratio=float(common/(math.sqrt(2*len(z1))*nc)) if float(nc)>0 else float(common)
    if relative>2e-5 or common_ratio>1e-5:raise ValueError('Boundary norm match or shared-translation cancellation differs')
    return ratio,float(np),float(nc),relative,common_ratio

def consistency(a,b,z1,z2,W,arm,audit):
    if arm not in ARMS or a.ndim!=2 or a.shape[1]!=6 or not len(a) or b.shape!=a.shape or z1.shape!=(len(a),160) or z2.shape!=z1.shape or W.shape!=(6,160):raise ValueError('Native six-class joint160 pair required')
    if a.dtype not in (torch.float32,torch.float64) or any(v.dtype!=a.dtype or v.device!=a.device or not bool(torch.isfinite(v).all()) for v in (a,b,z1,z2,W)):raise ValueError('Consistent finite full precision inputs required')
    if not torch.is_grad_enabled() or not z1.requires_grad or not z2.requires_grad:raise ValueError('Only actual student training graph permitted')
    with torch.no_grad():
        A=F.normalize(W.detach(),dim=1,eps=EPS)
        for logits,z in ((a,z1),(b,z2)):
            expected=30*F.normalize(z.detach(),dim=1,eps=EPS)@A.T
            if not torch.allclose(logits.detach(),expected,atol=3e-5,rtol=1e-5):raise ValueError('Actual native cosine classifier differs')
    p,q=a.log_softmax(1),b.log_softmax(1)
    native=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
    T=trace(a,b,W);point=penalty(z1,z2,T,False);center=penalty(z1,z2,T,True)
    ratio=z1.new_tensor(1.);np=nc=relative=common_ratio=0.
    if arm!='FULL':
        ratio,np,nc,relative,common_ratio=boundary_geometry(point,center,z1,z2)
        audit['gradient_probe_calls']+=1
    aux={'FULL':point*0,'POINT':point,'CENTER':center,'NORM_MATCH':ratio*point}[arm]
    loss=native if arm=='FULL' else native+aux
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite feature consistency loss')
    audit['calls']+=1;audit['packets']+=len(a);audit['aux_positive_calls']+=int(float(aux.detach())>0)
    for k,v in (('auxiliary_loss_sum',aux.detach()),('native_KL_sum',native.detach()),('Fisher_trace_sum',T.sum()),
                ('point_aux_sum',point.detach()),('center_aux_sum',center.detach()),('norm_ratio_sum',ratio),('point_gradient_norm_sum',np),('center_gradient_norm_sum',nc)):
        audit[k]+=float(v)
    audit['norm_ratio_max']=max(audit['norm_ratio_max'],float(ratio))
    audit['max_norm_match_relative_error']=max(audit['max_norm_match_relative_error'],relative)
    audit['max_common_translation_gradient_ratio']=max(audit['max_common_translation_gradient_ratio'],common_ratio)
    return loss,dict(native_symmetric_KL=native.detach(),auxiliary_loss=aux.detach(),consistency_loss=loss.detach(),
        Fisher_trace_mean=T.mean(),point_aux=point.detach(),center_aux=center.detach(),norm_ratio=ratio,
        point_boundary_gradient_norm=np,center_boundary_gradient_norm=nc,norm_match_relative_error=relative,
        common_translation_gradient_ratio=common_ratio,auxiliary_graph=aux)
