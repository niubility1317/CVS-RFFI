"""Actual native SSL baseline, four state-preserving U-displacement probes."""
from contextlib import contextmanager
from pathlib import Path
import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .fold import prepare
from .geometry import project,match_stored_norm
from experiments.cvs_evidence_responsibility_train20.model import classifier
from experiments.cvs_evidence_responsibility_train20.training_observer import actual_postclip_oracle
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng,_rng_snapshot,_equal,_live_snapshot,_live_readback

def norm(values):return sum(float(v.double().square().sum()) for v in values)**.5

def backward_L(loss,parameters):
    """A real L-only backward; restore each original grad object afterward."""
    saved=tuple(p.grad for p in parameters)
    try:
        for p in parameters:p.grad=None
        loss.backward(retain_graph=True)
        return tuple(None if p.grad is None else p.grad.detach().clone() for p in parameters)
    finally:
        for p,g in zip(parameters,saved):p.grad=g

@torch.no_grad()
def L_after(optimizer,gradients,step):
    temporary=copy.deepcopy(optimizer);parameters=[p for group in temporary.param_groups for p in group['params']]
    for p,g in zip(parameters,gradients):p.grad=None if g is None else g.clone()
    torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
    step(temporary)
    return tuple(p.detach().clone() for p in parameters)

class Observer:
    def __init__(self,c):
        d.validate(c);self.c=c;self.model=self.reader=self.optimizer=None
        self.out=None;self.bound=False;self.pending=None;self.current_L=None
        self.steps=self.calls=self.L_calls=self.U_packets=0;self.records=[]
    def data(self,native,ctx,out):
        self.anchor,self.held=prepare(native,ctx)
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        self.roles=role_ids_from_native(ctx)
        d.write(Path(out)/'source_fold.json',dict(held_RX=1,fit_RX=list(d.FIT),effective_counts={k:len(v) for k,v in self.roles.items()},
            held_L_packets=1260,anchor_L_packets=144,source_L_IQ_reads=1404,U_truth_read=False,target_access=False,
            original_raw_carrier_and_static_domain_ID_map_retained=True,all_training_calibration_probe_loaders_RX1_excluded=True,
            four_probe_candidate_updates_are_temporary=True,U_observations_per_pseudo_epoch=56624))
    def source_contract(self,expected,ctx):
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if role_ids_from_native(ctx)!=self.roles or {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Effective fold contract differs')
        if any(not set(v)<=set(expected['role_ids'][key]) for key,v in self.roles.items()):raise ValueError('Fold roles escape validated original source roles')
        return dict(expected,role_ids=self.roles,native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',
            source_receivers=list(d.FIT),held_RX=1,carrier_contract_ref=d.SOURCE,
            role_subset_rule='RX1 excluded from original L/U/V; no physical role reassignment',
            checkpoint_effective_training_contract='Four RX only; not compatible with full five-RX initialization reuse')
    def initialize(self,model,c,out):
        if c!=self.c or self.model is not None:raise ValueError('Own single scratch model required')
        self.model=model;self.out=Path(out)
        # Before student hooks and own EMA creation: no closures/graphs in reader.
        self.reader=copy.deepcopy(model).eval()
        self.weight_name=next(name for name,p in model.named_parameters() if p is classifier(model).weight)
    def capture(self,**kwargs):
        if kwargs['model'] is not self.model or kwargs['optimizer_step']!=self.L_calls+1:raise ValueError('Current own L context differs')
        if bool((kwargs['extra'][1]['rx_i']==1).any()):raise ValueError('Held RX entered current L')
        self.L_calls+=1;self.current_L=kwargs['identity_loss']
        return self.current_L.detach()*0.,{}
    def bind(self):
        if self.bound:return
        if self.model is None:raise ValueError('Student must initialize before binding')
        self.model._source_labeled_feature_hook=self.capture
        self.model._source_optimizer_update_probe_hook=self.probe;self.bound=True
    def risks(self,names,feature):
        self.reader.load_state_dict(self.model.state_dict(),strict=True);self.reader.eval()
        reader_params=dict(self.reader.named_parameters());device=next(self.model.parameters()).device
        with _preserve_rng(),fresh_parameter_caches(self.reader):
            output=self.reader.forward_identity_only(self.anchor['x'].to(device));logits=output['tx_logits']
            y=self.anchor['y'].to(device);rx=self.anchor['rx'].to(device);rows=[]
            for receiver in d.FIT:
                loss=F.cross_entropy(logits[rx==receiver],y[rx==receiver])
                gradients=torch.autograd.grad(loss,[reader_params[names[i]] for i in feature],retain_graph=True,allow_unused=True)
                rows.append(torch.cat([(torch.zeros_like(reader_params[names[i]]) if g is None else g).detach().flatten().double() for i,g in zip(feature,gradients)]))
        return torch.stack(rows)
    def probe(self,*,model,optimizer,epoch,batch_idx,phase,optimizer_step,total_loss,unlabeled_loss,labeled_count,unlabeled_count):
        if (model is not self.model or optimizer_step!=self.calls+1 or optimizer_step!=self.L_calls
                or optimizer_step!=(epoch-1)*222+batch_idx or labeled_count!=128):raise ValueError('Native step/L context differs')
        self.calls+=1;self.U_packets+=unlabeled_count
        if self.optimizer is None:self.optimizer=optimizer
        if optimizer is not self.optimizer:raise ValueError('Single original optimizer required')
        if optimizer_step not in d.PROBES:self.current_L=None;return
        if phase!='pseudo' or unlabeled_count!=256 or self.pending is not None:raise ValueError('Four fixed actual U probes required')
        parameters=tuple(p for group in optimizer.param_groups for p in group['params']);lookup={id(p):name for name,p in model.named_parameters()}
        names=[lookup[id(p)] for p in parameters];wi=names.index(self.weight_name)
        feature=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and i!=wi]
        if len(parameters)!=188 or len(feature)!=71:raise ValueError('Original188 / actual feature71 scope differs')
        if abs(float((total_loss-self.current_L-unlabeled_loss).detach()))>2e-6*max(1.,abs(float(total_loss.detach()))):raise ValueError('Baseline loss contains an undeclared task')
        live=_live_snapshot(optimizer);rng=_rng_snapshot()
        try:
            gradients=backward_L(self.current_L,parameters);risk=self.risks(names,feature)
            if gradients[wi] is None or float(gradients[wi].double().norm())<=0:raise ValueError('Original default W must jointly receive L gradients')
            if not all(_live_readback(optimizer,live).values()) or not _equal(rng,_rng_snapshot()):raise ValueError('Risk/L counterfactual changed live state')
            self.pending=dict(parameters=parameters,names=names,wi=wi,feature=feature,L_gradients=gradients,risk=risk,
                pre=tuple(p.detach().clone() for p in parameters),step=optimizer_step,epoch=epoch,U_loss=float(unlabeled_loss.detach()))
        finally:self.current_L=None
    def evaluate(self,names,after,wi,pre_weight):
        device=next(self.reader.parameters()).device
        sums={key:dict(count=0,CE_sum=0.,errors=0,margin_sum=0.) for key in ('ACTUAL','PRE_W')}
        with _preserve_rng(),torch.no_grad():
            self.reader.load_state_dict(self.model.state_dict(),strict=True);self.reader.eval();parameters=dict(self.reader.named_parameters())
            for name,value in zip(names,after):parameters[name].copy_(value)
            with fresh_parameter_caches(self.reader):
                for start in range(0,1260,128):
                    x=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device)
                    output=self.reader.forward_identity_only(x);z=output['id_feat_joint'].double()
                    actual=output['tx_logits'].double()
                    formula=30*F.linear(F.normalize(z,dim=1,eps=1e-4),F.normalize(after[wi].double(),dim=1,eps=1e-4))
                    if not torch.allclose(formula,actual,atol=2e-5,rtol=2e-6):raise ValueError('Default cosine formula differs')
                    pre_w=30*F.linear(F.normalize(z,dim=1,eps=1e-4),F.normalize(pre_weight.double(),dim=1,eps=1e-4))
                    for key,logits in (('ACTUAL',actual),('PRE_W',pre_w)):
                        row=sums[key];other=logits.clone();other.scatter_(1,y[:,None],float('-inf'))
                        row['count']+=len(y);row['CE_sum']+=float(F.cross_entropy(logits,y,reduction='sum'))
                        row['errors']+=int((logits.argmax(1)!=y).sum());row['margin_sum']+=float((logits.gather(1,y[:,None]).flatten()-other.max(1).values).sum())
        return {key:dict(count=1260,mean_CE=value['CE_sum']/1260,errors=value['errors'],accuracy=1-value['errors']/1260,mean_margin=value['margin_sum']/1260) for key,value in sums.items()}
    def proposals(self,optimizer):
        p=self.pending;feature=p['feature'];pre=p['pre'];names=p['names'];wi=p['wi']
        oracle=actual_postclip_oracle(optimizer,self.original_step);raw=oracle['after'];base=L_after(optimizer,p['L_gradients'],self.original_step)
        flat=lambda state:torch.cat([state[i].flatten() for i in feature])
        bf,rf=flat(base),flat(raw);u=rf.double()-bf.double();projection=project(u,p['risk'])
        cf=(bf.double()+projection['projected']).float();sf,match=match_stored_norm(bf,rf,cf)
        def materialize(vector):
            after=list(raw);offset=0
            for i in feature:
                count=after[i].numel();after[i]=vector[offset:offset+count].reshape_as(after[i]);offset+=count
            return tuple(after)
        afters=dict(RAW=raw,L_ONLY_F=materialize(bf),SHRINK=materialize(sf),CONE=materialize(cf))
        effective=cf.double()-bf.double();rawnorm=float(u.norm());fullnorm=norm([raw[i].double()-pre[i].double() for i in feature])
        normalized_risk=p['risk']/p['risk'].norm(dim=1)[:,None]
        results={mode:self.evaluate(names,state,wi,pre[wi]) for mode,state in afters.items()}
        pairs={control:{readout+'_'+suffix:results['CONE'][readout][metric]-results[control][readout][metric]
            for readout in ('ACTUAL','PRE_W') for metric,suffix in (('mean_CE','CE_difference'),('errors','error_difference'))} for control in ('RAW','SHRINK','L_ONLY_F')}
        cosine=float(torch.dot(effective,u)/(effective.norm()*u.norm())) if float(effective.norm()) and rawnorm else None
        record=dict(step=p['step'],epoch=p['epoch'],U_loss=p['U_loss'],held_RX=1,training_RX=list(d.FIT),held_L_packets=1260,
            geometry=dict(raw_U_F_increment_norm=rawnorm,full_F_step_norm=fullnorm,raw_U_F_to_full_F_step=rawnorm/fullnorm if fullnorm else 0.,
                CONE_U_F_increment_norm=float(effective.norm()),retained_U_F_norm_ratio=float(effective.norm())/rawnorm if rawnorm else 0.,
                CONE_RAW_cosine=cosine,projection_active_RX=[d.FIT[i] for i in projection['active']],
                continuous_cone_violation_relative=projection['normalized_max_violation'],
                rounded_cone_violation_relative=float((normalized_risk@effective).max())/rawnorm if rawnorm else 0.,stored_norm_control=match),
            held_results=results,paired=pairs,default_W_same_RAW_all_modes=True,feature_gains_in_F=True,head_fit_updates=0,
            actual_candidate_corrections=0,temporary_AdamW_steps=2,L_counterfactual_uses_actual_backward=True,
            default_W_actual_step_norm=float((raw[wi].double()-pre[wi].double()).norm()),
            RAW_reference='actual postclip gradient oracle, no second clip',live_state_preserved=True)
        return record,raw
    def epoch(self,model,ctx,rows,out,device):
        epoch=len(rows)
        if self.pending is not None or self.steps!=epoch*222 or self.calls!=self.steps or self.L_calls!=self.steps:raise ValueError('Native fold step closure differs')
        if self.U_packets!=max(0,epoch-13)*56624:raise ValueError('Declared four-RX U exposure differs')
        value=dict(status='SOURCE_MECHANISM_RUNNING',epoch=epoch,actual_optimizer_updates=self.steps,actual_candidate_corrections=0,
            U_packet_observations=self.U_packets,records=self.records,gate=d.gate(self.records),target_access=False,U_truth_read=False,head_fit_updates=0)
        if epoch==20:
            names=[name for name,p in model.named_parameters()];after=tuple(p.detach().clone() for p in model.parameters());wi=names.index(self.weight_name)
            value.update(status='SOURCE_MECHANISM_COMPLETE',final_held_RX_L=self.evaluate(names,after,wi,after[wi]),
                diagnostic_packet_forwards_including_initial_smoke=21998,additional_L_IQ_reads=1404,temporary_AdamW_steps=8,
                interpretation='One genuinely trained RX1-held-out SSL fold, four paired local counterfactual probes; no candidate full trajectory or target claim')
        d.write(Path(out)/'source_U_increment_mechanism.json',value)
    @contextmanager
    def native_scope(self,native):
        core=native.compute_core_losses;step=torch.optim.AdamW.step;self.original_step=step
        def losses(*a,**kw):
            result=core(*a,**kw);self.bind();return result
        def actual(optimizer,*a,**kw):
            owned=optimizer is self.optimizer
            if owned and self.pending is not None:
                before=_live_snapshot(optimizer);rng=_rng_snapshot();record,expected=self.proposals(optimizer)
                if not all(_live_readback(optimizer,before).values()) or not _equal(rng,_rng_snapshot()):raise ValueError('Temporary proposals altered live optimizer/RNG')
            else:record=None
            result=step(optimizer,*a,**kw)
            if owned:self.steps+=1
            if record is not None:
                errors=[float((p.detach()-v).abs().max()) for p,v in zip(self.pending['parameters'],expected)]
                record['actual_RAW_oracle_max_error']=max(errors);record['actual_RAW_oracle_bitwise_equal']=all(v==0 for v in errors)
                self.records.append(record);d.write(self.out/('U_increment_step'+str(record['step'])+'.json'),record)
                self.pending=None
                if not record['actual_RAW_oracle_bitwise_equal']:raise ValueError('Actual raw AdamW differs from actual-gradient reference')
                print('U_INCREMENT '+json.dumps(dict(step=record['step'],geometry=record['geometry'],paired=record['paired'])),flush=True)
            return result
        native.compute_core_losses=losses;torch.optim.AdamW.step=actual
        try:yield
        finally:
            native.compute_core_losses=core;torch.optim.AdamW.step=step
            if self.bound:
                delattr(self.model,'_source_labeled_feature_hook');delattr(self.model,'_source_optimizer_update_probe_hook')
            self.bound=False
