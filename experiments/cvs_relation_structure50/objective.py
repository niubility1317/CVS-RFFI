import torch
import torch.nn.functional as F
RXS=(1,3,4,6,8)
TAU=.2
ALPHA=.5
# 20 anchors/TX and day counts7/7/6 in BOTH ALL and CROSS comparisons.
ALL_DAY_COUNTS=((2,1,1),(1,2,1),(1,1,2),(2,1,1),(1,2,1))
CROSS_DAY_COUNTS=((2,2,1),(2,1,2),(1,2,2),(2,2,1))

def positions(counts):
 return [day*2+j for day,count in enumerate(counts) for j in range(count)]

def anchor_groups(bank,query_rx,cross):
 if query_rx not in RXS or bank.shape!=(5,6,6,160):raise ValueError('Exact legal L anchor bank/rx required')
 chosen=[r for r in RXS if not cross or r!=query_rx];counts=CROSS_DAY_COUNTS if cross else ALL_DAY_COUNTS
 return [(r,bank[RXS.index(r),:,positions(counts[i]),:].reshape(-1,160).detach()) for i,r in enumerate(chosen)]

def distribution(z,a,tau=TAU):
 return F.softmax(F.normalize(z,dim=-1,eps=1e-4)@F.normalize(a.detach(),dim=-1,eps=1e-4).T/tau,dim=-1)

def objective(student,teacher,bank,rx,kind,tau=TAU,alpha=ALPHA):
 if kind not in ('ZERO','ALL_REL','CROSS_REL','CROSS_RESP') or not 0<alpha<1 or tau<=0:raise ValueError('Unregistered relation intervention')
 if rx.ndim!=1 or not set(rx.detach().cpu().tolist())<=set(RXS):raise ValueError('Source RX metadata required')
 for parts in (student,teacher):
  if set(parts)!={'joint','PA','ref'} or any(x.shape!=(len(rx),160) or not torch.isfinite(x).all() for x in parts.values()):raise ValueError('True joint160/additive evidence required')
 if bank.shape!=(5,6,6,160) or not torch.isfinite(bank).all():raise ValueError('Bad L bank')
 teacher={k:v.detach() for k,v in teacher.items()};bank=bank.detach()
 rel=student['joint'].new_zeros(len(rx));resp=rel.clone();ent=rel.clone();response_energy=rel.clone();dispersion=[];cross=kind in ('CROSS_REL','CROSS_RESP')
 for r in RXS:
  ix=rx==r
  if not ix.any():continue
  group_r=[];group_e=[];group_h=[];group_energy=[]
  for other,a in anchor_groups(bank,r,cross):
   t=distribution(teacher['joint'][ix],a,tau).detach();s=distribution(student['joint'][ix],a,tau)
   dispersion.append(t.var(dim=0,unbiased=False).sum())
   group_r.append((t*(t.clamp_min(1e-12).log()-s.clamp_min(1e-12).log())).sum(-1))
   group_h.append(-(t*t.clamp_min(1e-12).log()).sum(-1)/torch.log(t.new_tensor(t.shape[1])))
   branches=[];energies=[]
   # Teacher and student weaken the SAME evidence, with the same anchors.
   # The signed response is preserved; it is not forced to zero.
   for branch in ('PA','ref'):
    dt=(t-distribution(teacher['joint'][ix]-alpha*teacher[branch][ix],a,tau))/alpha
    ds=(s-distribution(student['joint'][ix]-alpha*student[branch][ix],a,tau))/alpha
    branches.append(F.smooth_l1_loss(ds,dt.detach(),reduction='none',beta=.02).sum(-1));energies.append(dt.detach().norm(dim=-1))
   group_e.append(torch.stack(branches).mean(0));group_energy.append(torch.stack(energies).mean(0))
  rel[ix]=torch.stack(group_r).mean(0);resp[ix]=torch.stack(group_e).mean(0);ent[ix]=torch.stack(group_h).mean(0);response_energy[ix]=torch.stack(group_energy).mean(0)
 lr=rel.mean();le=resp.mean();weighted=lr+le if kind=='CROSS_RESP' else lr if kind!='ZERO' else (lr+le)*0
 positive=rel.detach().clamp_min(0);top_count=max(1,(len(rx)+99)//100);tail_share=positive.topk(top_count).values.sum()/positive.sum().clamp_min(1e-12)
 return weighted,dict(relation=lr,response=le),dict(U_count=len(rx),all_U_used=True,class_W_used=False,anchors_per_query=120,anchors_per_TX=20,anchor_day_counts_per_TX=[7,7,6],groups_per_query=4 if cross else 5,teacher_entropy_normalized=float(ent.mean().detach()),teacher_response_L2=float(response_energy.mean()),teacher_U_relation_dispersion=float(torch.stack(dispersion).mean()),bank_normalized_variance=float(F.normalize(bank.reshape(-1,160),dim=-1,eps=1e-4).var(dim=0,unbiased=False).sum()),teacher_PA_increment_norm=float(teacher['PA'].norm(dim=-1).mean()),teacher_ref_increment_norm=float(teacher['ref'].norm(dim=-1).mean()),student_PA_increment_norm=float(student['PA'].detach().norm(dim=-1).mean()),student_ref_increment_norm=float(student['ref'].detach().norm(dim=-1).mean()),nonzero_relation_fraction=float((rel.detach()>1e-8).float().mean()),relation_top_one_percent_mass=float(tail_share),relation_median=float(rel.detach().median()),relation_max=float(rel.detach().max()),response_median=float(resp.detach().median()),response_max=float(resp.detach().max()),per_packet_relation=rel.detach(),per_packet_response=resp.detach())
