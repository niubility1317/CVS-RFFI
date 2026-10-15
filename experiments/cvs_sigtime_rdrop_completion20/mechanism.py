"""Original predictive RDrop plus equal-trace complementary feature metrics."""
import torch
from torch.nn import functional as F
ARMS=('FULL','SPAN','ISOTROPIC','NULL')

def new_audit():
    return dict(calls=0,packets=0,min_rank=6,max_rank=0,max_projector_error=0.,max_trace_relative_error=0.,aux_positive_calls=0,
        auxiliary_loss_sum=0.,native_KL_sum=0.,Fisher_trace_sum=0.,null_difference_energy_sum=0.,total_difference_energy_sum=0.)

@torch.no_grad()
def geometry(a,b,W):
    A=F.normalize(W.detach().double(),dim=1,eps=1e-4);C=A-A.mean(0)
    _,s,Vh=torch.linalg.svd(C,full_matrices=False)
    rank=int((s>s[0]*1e-6).sum()) if float(s[0])>0 else 0
    if rank!=5:raise ValueError('Fixed six-way centered native head requires numerical rank5')
    B=Vh[:rank].T
    error=max(float((B.T@B-torch.eye(rank,device=B.device,dtype=B.dtype)).abs().max()),float((C-(C@B)@B.T).abs().max()))
    if error>2e-5:raise ValueError('Classifier complement geometry differs')
    p=.5*(a.detach().double().softmax(1)+b.detach().double().softmax(1))
    T=900*((p*A.square().sum(1)).sum(1)-(p@A).square().sum(1)).clamp_min(0)
    if not bool(torch.isfinite(T).all()) or not bool((T<=900+1e-6).all()):raise ValueError('Finite native Fisher trace required')
    return B.to(W.dtype),T.to(a.dtype),rank,error

def penalty(z1,z2,B,T,rank,arm):
    delta=F.normalize(z1,dim=1,eps=1e-4)-F.normalize(z2,dim=1,eps=1e-4)
    span=(delta@B)@B.T;null=delta-span
    if arm=='SPAN':e=span.square().sum(1)/rank
    elif arm=='ISOTROPIC':e=delta.square().sum(1)/160
    elif arm=='NULL':e=null.square().sum(1)/(160-rank)
    elif arm=='FULL':e=delta.square().sum(1)*0
    else:raise ValueError('Fixed consistency geometry required')
    return .5*(T*e).mean(),null.square().sum().detach(),delta.square().sum().detach()

def consistency(a,b,z1,z2,W,arm,audit):
    if arm not in ARMS or a.ndim!=2 or a.shape[1]!=6 or not len(a) or b.shape!=a.shape or z1.shape!=(len(a),160) or z2.shape!=z1.shape or W.shape!=(6,160):raise ValueError('Finite native6class/160feature paired inputs required')
    if a.dtype not in (torch.float32,torch.float64) or any(v.dtype!=a.dtype or v.device!=a.device or not bool(torch.isfinite(v).all()) for v in (a,b,z1,z2,W)):raise ValueError('Consistent finite full precision inputs required')
    # Assert the geometry is that of the actual jointly trained default head.
    with torch.no_grad():
        A=F.normalize(W.detach(),dim=1,eps=1e-4)
        for logits,z in ((a,z1),(b,z2)):
            expected=30*F.normalize(z.detach(),dim=1,eps=1e-4)@A.T
            if not torch.allclose(logits.detach(),expected,atol=3e-5,rtol=1e-5):raise ValueError('Actual native cosine classifier reconstruction differs')
    p,q=a.log_softmax(1),b.log_softmax(1)
    native=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
    B,T,rank,error=geometry(a,b,W)
    aux,null_energy,total_energy=penalty(z1,z2,B,T,rank,arm)
    loss=native if arm=='FULL' else native+aux
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite completion loss')
    # Trace equality is exact algebra; verify the actual numerical basis too.
    trace_span=float(B.double().square().sum())
    trace_ratio=trace_span/rank if arm=='SPAN' else (160-trace_span)/(160-rank) if arm=='NULL' else 1.
    trace_error=abs(trace_ratio-1)
    if trace_error>2e-5:raise ValueError('Added metric trace differs')
    audit['calls']+=1;audit['packets']+=len(a)
    audit['min_rank']=min(audit['min_rank'],rank);audit['max_rank']=max(audit['max_rank'],rank)
    audit['max_projector_error']=max(audit['max_projector_error'],error)
    audit['max_trace_relative_error']=max(audit['max_trace_relative_error'],trace_error)
    audit['aux_positive_calls']+=int(float(aux.detach())>0)
    audit['auxiliary_loss_sum']+=float(aux.detach());audit['native_KL_sum']+=float(native.detach())
    audit['Fisher_trace_sum']+=float(T.sum());audit['null_difference_energy_sum']+=float(null_energy);audit['total_difference_energy_sum']+=float(total_energy)
    return loss,dict(native_symmetric_KL=native.detach(),auxiliary_loss=aux.detach(),consistency_loss=loss.detach(),
        Fisher_trace_mean=T.mean(),classifier_rank=rank,projector_error=error,trace_relative_error=trace_error,
        null_difference_fraction=null_energy/total_energy if float(total_energy)>0 else total_energy,auxiliary_graph=aux)
