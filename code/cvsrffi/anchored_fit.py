"""Cache-only angular fitting with physical weights and explicit finite budgets."""
from dataclasses import dataclass,asdict,fields
import csv,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .anchored_geometry import AnchoredMetricHead,OrdinaryAngleHead,load_angle_head


@dataclass(frozen=True)
class FitConfig:
    candidate:str='A4'
    epochs:int=80
    rank:int=4
    rho:float=math.log(2)/2
    learning_rate:float=.001
    metric_weight:float=.01
    keep_scale:float=1.
    keep_margin_fraction:float=.1
    physical_per_group:int=2
    stability_window:int=5
    stability_ce:float=.002
    stability_metric:float=.001

    @classmethod
    def parse(cls,value):
        if isinstance(value,cls):obj=value
        else:
            if not isinstance(value,dict) or set(value)-{f.name for f in fields(cls)}:raise ValueError('unknown expert config field')
            obj=cls(**value)
        if obj.candidate not in {'A2','A3','A4','C_angle','C_angle_keep'}:raise ValueError('unsupported candidate')
        if not 1<=obj.epochs<=80 or obj.rank<1 or obj.physical_per_group<1 or obj.stability_window<2:
            raise ValueError('invalid fitting budget/rank/window')
        for name in ('rho','learning_rate','metric_weight','keep_scale','keep_margin_fraction','stability_ce','stability_metric'):
            value=getattr(obj,name)
            if not math.isfinite(value) or value<0:raise ValueError('invalid '+name)
        if obj.rho==0 or obj.learning_rate==0 or obj.keep_margin_fraction==0:raise ValueError('positive metric/lr/margin required')
        return obj

    @property
    def paired(self):return self.candidate!='A2'

    @property
    def keep(self):return self.candidate in {'A4','C_angle_keep'}

    @property
    def angular_control(self):return self.candidate.startswith('C_angle')


def physical_groups(rows):
    groups={};physical={}
    for i,pid in enumerate(rows.physical_ids):physical.setdefault(pid,[]).append(i)
    for pid,indices in physical.items():
        i=indices[0];key=(int(rows.labels[i]),int(rows.receiver[i]),int(rows.day[i]))
        groups.setdefault(key,[]).append((pid,indices))
    return groups


def physical_batches(rows,per_group,seed):
    rows.validate();groups=physical_groups(rows)
    counts={len(v) for v in groups.values()}
    ys=rows.labels.unique().tolist();rx=rows.receiver.unique().tolist();days=rows.day.unique().tolist()
    if len(counts)!=1 or len(groups)!=len(ys)*len(rx)*len(days) or next(iter(counts))%per_group:
        raise ValueError('balanced complete physical groups divisible by batch count required')
    generator=torch.Generator().manual_seed(int(seed));ordered=[]
    for key in sorted(groups):
        members=sorted(groups[key]);order=torch.randperm(len(members),generator=generator).tolist()
        ordered.append([members[i] for i in order])
    for start in range(0,next(iter(counts)),per_group):
        yield torch.tensor([i for group in ordered for _,indices in group[start:start+per_group] for i in indices],dtype=torch.long)


def paired_weights(rows,paired=True):
    by_id={}
    for i,pid in enumerate(rows.physical_ids):by_id.setdefault(pid,[]).append(i)
    weights=torch.zeros(len(rows),dtype=torch.float32,device=rows.h.device)
    for indices in by_id.values():
        clean=[i for i in indices if rows.view_ids[i]=='clean'];leo=[i for i in indices if rows.view_ids[i]!='clean']
        if len(clean)!=1 or (paired and not leo):raise ValueError('each physical sample needs one clean and paired LEO views')
        weights[clean]=(.5 if paired else 1.)/len(by_id)
        if paired:weights[leo]=.5/len(leo)/len(by_id)
    return weights


def class_margin(logits,labels):
    selected=logits.gather(1,labels[:,None]).squeeze(1)
    alternatives=logits.scatter(1,labels[:,None],float('-inf')).amax(-1)
    return selected-alternatives


def keep_loss(s0,sg,labels,*,gamma_max,valid=None,weights=None):
    g0=class_margin(s0.detach(),labels);gg=class_margin(sg,labels)
    valid=torch.ones_like(g0,dtype=torch.bool) if valid is None else valid
    weights=torch.ones_like(g0)/len(g0) if weights is None else weights
    valid_weights=weights*valid
    return (valid_weights*(g0>0)*F.relu(g0.clamp_max(gamma_max)-gg).square()).sum()/valid_weights.sum().clamp_min(torch.finfo(sg.dtype).tiny)


def write_csv(path,rows):
    if not rows:raise ValueError('empty history')
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@dataclass
class ExpertArtifact:
    state:dict
    train_rx:tuple
    seed:int
    config:dict
    history:list
    geometry:list
    stop_status:str

    def predict(self,rows,device='cpu'):
        head=load_angle_head(self.state).to(device).eval()
        with torch.no_grad():return head(rows.h.to(device)).cpu()

    def save(self,path):
        torch.save(asdict(self),path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if set(data)!={f.name for f in fields(cls)}:raise ValueError('invalid expert artifact')
        data['train_rx']=tuple(data['train_rx']);FitConfig.parse(data['config']);load_angle_head(data['state'])
        return cls(**data)


def _matrix(head):
    if isinstance(head,AnchoredMetricHead):
        q,a=head.spectral();return (torch.eye(q.shape[0],device=q.device,dtype=q.dtype)+(q*torch.expm1(a))@q.T).detach()
    return torch.eye(head.weight.shape[1],device=head.weight.device)


def _gradients(terms,params):
    vectors=[]
    for term in terms:
        if term.requires_grad:
            grads=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
            vector=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
        else:vector=torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
        vectors.append(vector)
    norms=[float(x.norm()) for x in vectors]
    pairs={}
    for i,j,name in ((0,1,'ce_keep'),(0,2,'ce_metric'),(1,2,'keep_metric')):
        denom=vectors[i].norm()*vectors[j].norm()
        pairs['grad_cos_'+name]=float(vectors[i].dot(vectors[j])/denom) if denom>0 else None
    return dict(zip(('grad_norm_ce','grad_norm_keep','grad_norm_metric'),norms)),pairs


def fit_expert(cache,train_rx,config,seed,output,*,w0,tau0,validation_rx=(),device='cpu'):
    config=FitConfig.parse(config);cache.validate()
    if cache.identity.role!='L_s':raise ValueError('expert fit requires L_s')
    train_rx=tuple(sorted(set(map(int,train_rx))));validation_rx=tuple(sorted(set(map(int,validation_rx))))
    if not train_rx or set(train_rx)&set(validation_rx):raise ValueError('empty training receivers or fit/validation overlap')
    available=set(cache.receiver.tolist())
    if not set(train_rx+validation_rx)<=available:raise ValueError('receiver absent from cache')
    if not cache.observed.all():raise ValueError('G requires complete observed features; use separate E')
    take=[i for i in range(len(cache)) if int(cache.receiver[i]) in train_rx]
    if not config.paired:take=[i for i in take if cache.view_ids[i]=='clean']
    rows=cache.take(take)
    held=[i for i in range(len(cache)) if int(cache.receiver[i]) in validation_rx]
    val=cache.take(held) if held else None
    # Validate balance before creating a partial output folder.
    list(physical_batches(rows,config.physical_per_group,seed))
    if w0.shape!=(cache.baseline_inference_logits.shape[1],cache.h.shape[1]):raise ValueError('H0 directions/cache schema mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(int(seed))
    head=(OrdinaryAngleHead(w0,tau0) if config.angular_control else AnchoredMetricHead(w0,tau0,config.rank,config.rho)).to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=config.learning_rate,weight_decay=0.)
    parameters=list(head.parameters());history=[];geometry=[];previous=_matrix(head);previous_a=None
    steps=physical_visits=view_visits=0;converged=False;first_converged_epoch=None
    for epoch in range(1,config.epochs+1):
        begin=time.perf_counter();head.train();totals={'ce_clean':0.,'ce_leo':0.,'keep_raw':0.,'metric_raw':0.,'loss':0.}
        epoch_physical=0;gradstats={};metric_change=0.
        for index in physical_batches(rows,config.physical_per_group,seed+epoch):
            batch=rows.take(index);h=batch.h.to(device);y=batch.labels.to(device)
            weights=paired_weights(batch,config.paired).to(device);sg=head(h)
            ce_each=F.cross_entropy(sg,y,reduction='none');ce=(weights*ce_each).sum()
            clean=torch.tensor([v=='clean' for v in batch.view_ids],device=device)
            clean_ce=(weights*ce_each*clean).sum()/(.5 if config.paired else 1.)
            leo_ce=(weights*ce_each*~clean).sum()/.5 if config.paired else sg.sum()*0
            keep=keep_loss(batch.baseline_inference_logits.to(device),sg,y,gamma_max=config.keep_margin_fraction*tau0,
                           valid=batch.valid.to(device),weights=weights) if config.keep else sg.sum()*0
            metric=head.spectral()[1].square().sum() if isinstance(head,AnchoredMetricHead) else sg.sum()*0
            terms=(ce,config.keep_scale/tau0**2*keep,config.metric_weight*metric)
            loss=sum(terms)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite head loss')
            if not gradstats:
                norms,angles=_gradients(terms,parameters);gradstats={**norms,**angles}
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('nonfinite head gradient')
            optimizer.step();steps+=1
            count=len(set(batch.physical_ids));epoch_physical+=count;physical_visits+=count;view_visits+=len(batch)
            for key,value in zip(totals,(clean_ce,leo_ce,keep,metric,loss)):totals[key]+=float(value.detach())*count
        head.eval();current=_matrix(head);metric_change=float((current-previous).norm()/previous.norm());previous=current
        diagnostic=head.geometry_diagnostics();a=diagnostic['a'];a_change=max((abs(x-y) for x,y in zip(a,previous_a)),default=0.) if previous_a is not None else 0.;previous_a=a
        with torch.no_grad():
            train_scores=head(rows.h.to(device)).cpu();margins=class_margin(train_scores,rows.labels.cpu())
            if val is not None:
                val_logits=head(val.h.to(device)).cpu()
                vce=float((paired_weights(val,True).cpu()*F.cross_entropy(val_logits,val.labels.cpu(),reduction='none')).sum())
            else:vce=None
        record=dict(candidate=config.candidate,train_rx=','.join(map(str,train_rx)),head_seed=int(seed),epoch=epoch,
                    optimizer_steps=steps,physical_visits_total=physical_visits,view_visits_total=view_visits,
                    **{k:v/epoch_physical for k,v in totals.items()},**gradstats,validation_ce=vce,
                    metric_relative_change=metric_change,a_max_change=a_change,epoch_seconds=time.perf_counter()-begin)
        history.append(record)
        print('[ANCHORED-EPOCH] '+json.dumps(record,allow_nan=False),flush=True)
        geometry.append(dict(candidate=config.candidate,head_seed=int(seed),epoch=epoch,condition_number=diagnostic['condition_number'],
                             orthogonality_error=diagnostic['orthogonality_error'],eigenvalues=json.dumps(diagnostic['eigenvalues']),
                             direction_angles_deg=json.dumps(diagnostic['direction_angles_deg']),a=json.dumps(a),
                             metric_relative_change=metric_change,feature_norm_p05=float(rows.h.norm(dim=-1).quantile(.05)),
                             feature_norm_p95=float(rows.h.norm(dim=-1).quantile(.95)),margin_p05=float(margins.quantile(.05)),margin_p50=float(margins.median())))
        window=history[-config.stability_window:]
        converged=(len(window)==config.stability_window and all(r['validation_ce'] is not None for r in window)
                   and max(r['validation_ce'] for r in window)-min(r['validation_ce'] for r in window)<=config.stability_ce
                   and max(r['metric_relative_change'] for r in window)<=config.stability_metric
                   and not all(window[i]['keep_raw']>window[i-1]['keep_raw'] for i in range(1,len(window))))
        if converged and first_converged_epoch is None:first_converged_epoch=epoch
        if epoch in {20,40,80,config.epochs}:torch.save(head.export_state(),output/f'expert_epoch_{epoch:04d}.pt')
    status='converged' if converged else 'budget_exhausted'
    for record in history:
        record['keep_weighted']=config.keep_scale/tau0**2*record['keep_raw']
        record['metric_weighted']=config.metric_weight*record['metric_raw']
        record['stop_status']=status if record['epoch']==config.epochs else 'running_fixed_budget'
    artifact=ExpertArtifact(head.export_state(),train_rx,int(seed),asdict(config),history,geometry,status)
    artifact.save(output/'expert.pt');write_csv(output/'head_fit_history.csv',history);write_csv(output/'geometry_diagnostics.csv',geometry)
    (output/'fit_manifest.json').write_text(json.dumps(dict(candidate=config.candidate,cache_identity=asdict(cache.identity),train_rx=train_rx,
        validation_rx=validation_rx,config=asdict(config),head_seed=int(seed),stop_status=status,
        convergence_validation_available=val is not None,first_converged_epoch=first_converged_epoch,optimizer_steps=steps,backbone_updated=False,internal_scale=tau0,
        physical_count=len(set(rows.physical_ids)),cache_rows=len(rows),fit_seconds=sum(r['epoch_seconds'] for r in history)),indent=2)+'\n',encoding='utf-8')
    return artifact
