"""Online state-action training beside the unchanged clean+LEO+pseudo loop.

Only visible source-L enters auxiliaries. A source-physical split calibrates
virtual weights; real-IQ CE never depends on that calibration. A frozen copy of
this run's EMA defines one coordinate version at a time, refreshed explicitly.
"""
from contextlib import contextmanager
from collections import defaultdict
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_multi_disentangle.runtime import deterministic_forward, frozen_parameters, finite_grad_norm
from experiments.cvs_multi_disentangle.physics import factorial_batch
from .model import (intermediate, identity_from_intermediate, classify_intermediate,
                    reference_response, UnifiedAction, ReceiverAdapter)

ARM_PATHS={'native':(), 'real_views':(), 'unified':('linear','temporal','receiver'),
    'L':('linear',), 'LT':('linear','temporal'), 'LTR':('linear','temporal','receiver')}
ARM_PATHS.update({key+'_EG':ARM_PATHS[key] for key in ('L','LT','LTR')})


def physical_audit_role(pid):
    # Same partition as OnlineReceiver, essential when one shared network fits
    # L/T/R: an L/T calibration packet must not train the shared core via R.
    return (int(hashlib.sha256(str(pid).encode('utf-8')).hexdigest(),16)//3)%2==1


def _scalar(v):
    return float(v.detach()) if torch.is_tensor(v) else v


def gradient_vector(loss, parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p,g in zip(parameters,values)])


def gradient_comparison(reference, candidate):
    a,b=reference.norm(),candidate.norm()
    return dict(norm=float(b),cosine=float(torch.dot(reference,candidate)/(a*b).clamp_min(1e-12)),
                reference_norm=float(a),both_nonzero=bool(a>0 and b>0))


class OnlineTrainer:
    def __init__(self,c,model):
        from .actions import StateAction
        from .receiver import OnlineReceiver
        self.config=c;self.recipe=c['state_recipe'];self.output_root=Path(c['output_root'])
        self.arm=c['arm'];self.base_arm=self.arm.removesuffix('_EG')
        self.paths=ARM_PATHS[self.base_arm];self.virtual_eg=self.arm.endswith('_EG')
        self.device=next(model.parameters()).device
        self.generator=torch.Generator().manual_seed(int(c['model_seed'])+49071)
        self.auxiliary=nn.ModuleDict();self.receiver=None;self.reference=None
        self.reference_version=0;self.reference_step=None;self.updates=0
        self.last={};self.totals=defaultdict(float);self.epoch_count=0
        self.fit_steps=defaultdict(int);self.identity_steps=defaultdict(int)
        self.attempted_identity_steps=defaultdict(int);self.receiver_relations_total=0
        self.exposure_totals=defaultdict(int);self.calibration={};self.seconds=0.
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(c['model_seed'])+21939)
            if self.base_arm=='unified':
                # Match the total L/T/R trainable capacity within one shared
                # core. Input adapters remain separate but no task has its own D.
                target=sum(sum(p.numel() for p in StateAction(k,hidden=int(self.recipe.get('hidden',48))).parameters())
                           for k in ('linear','temporal'))
                target+=sum(p.numel() for p in OnlineReceiver().action.parameters())
                width=min(range(32,1025),key=lambda w:abs(w*w+751*w+2213-target))
                self.auxiliary['shared']=UnifiedAction(hidden=width)
                self.last['unified_capacity_reference']=target
            else:
                for kind in self.paths:
                    if kind!='receiver':
                        self.auxiliary[kind]=StateAction(kind,use_state=True,second_order=True,
                            exact_reference=True,hidden=int(self.recipe.get('hidden',48)))
            if 'receiver' in self.paths:
                adapter=ReceiverAdapter(self.auxiliary['shared']) if self.base_arm=='unified' else None
                self.receiver=OnlineReceiver(action=adapter,max_age_steps=int(self.recipe.get('receiver_max_age_steps',888)))
                self.auxiliary['receiver']=self.receiver.action
        self.auxiliary.to(self.device)
        self.optimizer=(torch.optim.AdamW(self.auxiliary.parameters(),lr=self.recipe['auxiliary_lr'],weight_decay=1e-4)
                        if len(self.auxiliary) else None)
        self.cost=dict(identity_parameters=sum(p.numel() for p in model.id_backbone.parameters()),
            native_model_parameters=sum(p.numel() for p in model.parameters()),
            native_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            auxiliary_parameters=sum(p.numel() for p in self.auxiliary.parameters()),
            inference_auxiliary_parameters=0,identity_forward_unchanged=True,
            auxiliary_data='visible_source_L_only',native_optimizer_excludes_auxiliaries=True,
            virtual_updates='E+G+C' if self.virtual_eg else 'G+C',target_access=False)

    def predict(self,kind,h,x,p):
        def ref(z):
            self.exposure_totals['exact_reference_calls']+=1
            self.exposure_totals['exact_reference_packets']+=len(z)
            return reference_response(self.reference,z)
        if self.base_arm=='unified':return self.auxiliary['shared'].digital(kind,h,x,p,ref)
        return self.auxiliary[kind](h,x,p,reference_fn=ref)

    def _refresh(self,ema,step):
        period=int(self.recipe.get('reference_refresh_steps',888))
        if self.reference_step is not None and step-self.reference_step<period:return
        previous=self.reference
        self.reference=deepcopy(ema.id_backbone).eval()
        for p in self.reference.parameters():p.requires_grad_(False)
        self.reference_version+=1;self.reference_step=step;self.calibration={}
        self.last['reference_refreshed']=1.
        self.last['reference_version']=self.reference_version
        # Receiver handles version changes on update, expiring all old entries.
        del previous

    def _calibrate(self,kind,condition,h,target,pred,y,identity=None,physical_ids=()):
        from .actions import vector_margin
        identity=self.reference if identity is None else identity
        with torch.no_grad():
            z0=F.normalize(identity_from_intermediate(identity,h),dim=1)
            z1=F.normalize(identity_from_intermediate(identity,h+target),dim=1)
            zp=F.normalize(identity_from_intermediate(identity,h+pred),dim=1)
            m0=vector_margin(classify_intermediate(identity,h),y)
            m1=vector_margin(classify_intermediate(identity,h+target),y)
            mp=vector_margin(classify_intermediate(identity,h+pred),y)
            values=dict(zerror=float((zp-z1).square().sum()),zzero=float((z0-z1).square().sum()),
                merror=float((mp-m1).square().sum()),mzero=float((m0-m1).square().sum()),count=len(h))
        key=(kind,condition);old=self.calibration.setdefault(key,defaultdict(float))
        # Bounded exponential calibration within a coordinate version, after fit
        # on physically disjoint packets. We do not train on calibration loss.
        for k,v in values.items():old[k]=.95*old[k]+v
        seen=old.setdefault('physical_ids',set());seen.update(physical_ids)
        old['unique_packets']=len(seen)
        zskill=1-old['zerror']/max(old['zzero'],1e-12)
        mskill=1-old['merror']/max(old['mzero'],1e-12)
        weight=max(0.,min(1.,zskill,mskill)) if len(seen)>=16 else 0.
        old.update(weight=weight,zskill=zskill,mskill=mskill)
        self.last.update({f'{kind}_{condition}_'+k:old[k] for k in ('weight','zskill','mskill','count','unique_packets')})

    def _weight(self,kind,condition):
        return self.calibration.get((kind,condition),{}).get('weight',0.)

    def loss(self,model,ema_model,x,y,receivers,epoch,step,*,days=None,physical_ids=None,
             leo_x=None,native_loss=None,leo_applied=None):
        self.last=dict(active=0.,epoch=epoch,native_update=step,reference_refreshed=0.,
            full_native_gradient_reference=False,source_L_only=True,
            real_weight=float(self.recipe.get('identity_real_weight',.05)),
            virtual_total_budget=float(self.recipe.get('total_auxiliary_weight',.05)))
        if self.base_arm=='native' or epoch<=self.recipe['warmup_epochs'] or step%self.recipe['auxiliary_every_steps']:
            return None
        if ema_model is None:raise ValueError('Own current-run EMA required')
        if days is None or physical_ids is None:raise ValueError('Physical source IDs and day required')
        if len(physical_ids)!=len(x) or len(y)!=len(x) or len(receivers)!=len(x) or len(days)!=len(x):
            raise ValueError('Source metadata misalignment')
        if bool((y<0).any()) or bool((receivers<0).any()) or bool((days<0).any()):
            raise ValueError('Hidden U metadata is forbidden in auxiliary steps')
        if leo_x is None or leo_x.shape!=x.shape:
            raise ValueError('Same native source-LEO view required for joint action training')
        start=time.perf_counter();count=min(len(x),int(self.recipe['auxiliary_batch_size']),32)
        ix=torch.randperm(len(x),generator=self.generator)[:count].to(x.device)
        ids=[physical_ids[i] for i in ix.cpu().tolist()]
        y,receivers,days=y[ix],receivers[ix],days[ix]
        inputs={'clean':x[ix],'native_LEO_view':leo_x[ix]}
        self.last['native_LEO_view_changed_packets']=int((leo_x[ix]!=x[ix]).flatten(1).any(1).sum())
        self.last['native_sat_augmentation_applied_packets']=None if leo_applied is None else count*int(bool(leo_applied))
        self.last['native_LEO_view_semantics']='native scheduled satellite view, including clean duplicates when augmentation is skipped'
        audit=torch.tensor([physical_audit_role(p) for p in ids],device=x.device)
        fit=~audit;self._refresh(ema_model,step)
        identity=model.id_backbone;devices=[x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices),deterministic_forward(identity):
            # Both registered source conditions use exactly the same increment
            # schedule and one parameter set. Native RNG remains untouched.
            generator_state=self.generator.get_state();batches={}
            for condition,inp in inputs.items():
                self.generator.set_state(generator_state)
                batches[condition]=factorial_batch(inp,self.generator)
            anchors={}
            with torch.no_grad():
                for condition,b in batches.items():
                    anchors[condition]={k:intermediate(self.reference,b[k]).detach() for k in ('x00','x10','x01','x11')}
                    if self.receiver is not None:
                        self.receiver.update(b['x00'],anchors[condition]['x00'],y,receivers,days,ids,
                            condition,step,self.reference_version,self.reference)
            fit_losses=[]
            if self.optimizer is not None:
                from .actions import action_loss
                self.optimizer.zero_grad(set_to_none=True)
                for condition,b in batches.items():
                    a=anchors[condition]
                    for kind in self.paths:
                        if kind=='receiver' or not bool(fit.any()):continue
                        target_key='x10' if kind=='linear' else 'x01'
                        p=b[kind+'_parameters'];target=a[target_key]-a['x00']
                        predicted=self.predict(kind,a['x00'][fit],b['x00'][fit],p[fit])
                        parts=action_loss(self.reference,a['x00'][fit],target[fit],predicted,y[fit])
                        fit_losses.append(parts['loss']);self.fit_steps[kind]+=1
                        self.last.update({f'{kind}_{condition}_fit_{k}':_scalar(v) for k,v in parts.items()})
                if self.receiver is not None:
                    rloss,rstats=self.receiver.fit_loss(self.reference,step,self.reference_version)
                    self.last.update({'R_fit_'+k:_scalar(v) for k,v in rstats.items()})
                    if rloss is not None and rloss.requires_grad:
                        fit_losses.append(rloss);self.fit_steps['receiver']+=1
                        self.receiver_relations_total+=int(rstats.get('relations',0))
                if fit_losses:
                    fit_loss=torch.stack(fit_losses).mean()
                    if not torch.isfinite(fit_loss):raise FloatingPointError('Nonfinite auxiliary fit')
                    fit_loss.backward();self.last['auxiliary_gradient_norm']=finite_grad_norm(self.auxiliary)
                    torch.nn.utils.clip_grad_norm_(self.auxiliary.parameters(),1.)
                    fraction=min(1.,max(0.,(epoch-self.recipe['warmup_epochs'])/max(1,self.config.get('epochs',200)-self.recipe['warmup_epochs'])))
                    lr=self.recipe.get('auxiliary_lr_min',1e-6)+.5*(self.recipe['auxiliary_lr']-self.recipe.get('auxiliary_lr_min',1e-6))*(1+math.cos(math.pi*fraction))
                    for group in self.optimizer.param_groups:group['lr']=lr
                    self.optimizer.step();self.updates+=1
                    self.last.update(auxiliary_fit_loss=float(fit_loss.detach()),auxiliary_lr=lr)
                self.optimizer.zero_grad(set_to_none=True)
            rweights={}
            if self.receiver is not None:
                calibration=self.receiver.calibrate(self.reference,step,self.reference_version)
                rweights=calibration.get('weights',{})
                for condition,weight in rweights.items():self.last['R_'+condition+'_weight']=weight
                self.last['R_calibration_active']=calibration.get('active',False)
            real_terms=[];consistency_terms=[];virtual_terms=[];rterms=[];gradient_tasks=[]
            for condition,b in batches.items():
                a=anchors[condition]
                # Equal real-IQ factorial exposure in every non-native arm,
                # including L-only; normalization prevents more branches buying
                # extra CE weight or extra input exposures.
                real_logits=[];real_features={}
                for name in ('x10','x01','x11'):
                    hreal=intermediate(identity,b[name]);logits=classify_intermediate(identity,hreal)
                    real_features[name]=hreal
                    real_logits.append(logits);real_terms.append(F.cross_entropy(logits,y))
                h=intermediate(identity,b['x00'])
                clean_logits=classify_intermediate(identity,h)
                consistency=torch.stack([F.kl_div(F.log_softmax(z,1),F.softmax(clean_logits.detach(),1),reduction='batchmean') for z in real_logits]).mean()
                consistency_terms.append(consistency)
                self.last[f'{condition}_coordinate_drift_h_mse']=float((h.detach()-a['x00']).square().mean())
                for kind in self.paths:
                    if kind=='receiver':continue
                    p=b[kind+'_parameters'];target=a['x10' if kind=='linear' else 'x01']-a['x00']
                    with torch.no_grad():predicted=self.predict(kind,a['x00'],b['x00'],p).detach()
                    if bool(audit.any()):
                        # Calibrate the learned reference-coordinate action in
                        # the actually changing student's decision geometry.
                        # This exposes online E/G drift, not merely old-reference
                        # fit quality, while calibration has no fitting gradient.
                        actual_target=real_features['x10' if kind=='linear' else 'x01'].detach()-h.detach()
                        heldout_ids=[pid for pid,keep in zip(ids,audit.cpu().tolist()) if keep]
                        self._calibrate(kind,condition,h.detach()[audit],actual_target[audit],predicted[audit],y[audit],identity,heldout_ids)
                    # Frozen learned shift: action parameters and coordinate
                    # targets never receive identity optimizer gradients.
                    virtual_h=(h if self.virtual_eg else h.detach())+predicted
                    vloss=F.cross_entropy(classify_intermediate(identity,virtual_h),y)
                    weight=self._weight(kind,condition)
                    virtual_terms.append(weight*vloss)
                    self.attempted_identity_steps[kind]+=1
                    self.identity_steps[kind]+=int(weight>0)
                    self.last[f'{kind}_{condition}_virtual_ce']=float(vloss.detach())
                    if condition=='clean':
                        real=F.cross_entropy(real_logits[0 if kind=='linear' else 1],y)
                        exact_delta=(real_features['x10' if kind=='linear' else 'x01']-h).detach()
                        exact_loss=F.cross_entropy(classify_intermediate(identity,
                            (h if self.virtual_eg else h.detach())+exact_delta),y)
                        gradient_tasks.append((kind,real,vloss,exact_loss))
                if self.receiver is not None:
                    self.attempted_identity_steps['receiver']+=1
                    live_calibration=self.receiver.calibrate_live(identity,h.detach(),y,receivers,days,ids,
                        condition,step,self.reference_version)
                    live_weight=live_calibration.get('weights',{}).get(condition,0.)
                    self.last[f'R_{condition}_live_weight']=live_weight
                    self.last[f'R_{condition}_live_calibration_active']=live_calibration.get('active',False)
                    self.last[f'R_{condition}_target_semantics']='fixed own-EMA teacher destination distribution; calibrated in live student geometry'
                    rh=h if self.virtual_eg else h.detach()
                    rloss,rstats=self.receiver.identity_loss(identity,rh,y,receivers,days,ids,
                        condition,step,self.reference_version)
                    self.last.update({f'R_{condition}_identity_'+k:_scalar(v) for k,v in rstats.items()})
                    if rloss is not None:
                        weight=min(rweights.get(condition,0.),live_weight)
                        self.last[f'R_{condition}_effective_weight']=weight
                        rterms.append(weight*rloss)
                        self.identity_steps['receiver']+=int(weight>0)
            real_loss=torch.stack(real_terms).mean()+float(self.recipe.get('paired_consistency_weight',.01))*torch.stack(consistency_terms).mean()
            virtual=torch.stack(virtual_terms).mean() if virtual_terms else real_loss*0
            receiver=torch.stack(rterms).mean() if rterms else real_loss*0
            vw=float(self.recipe.get('identity_virtual_weight',.05)) if virtual_terms else 0.
            rw=float(self.recipe.get('receiver_identity_weight',.01)) if rterms else 0.
            normalization=min(1.,float(self.recipe.get('total_auxiliary_weight',.05))/max(vw+rw,1e-12))
            result=float(self.recipe.get('identity_real_weight',.05))*real_loss+normalization*(vw*virtual+rw*receiver)
            if native_loss is not None and step%int(self.recipe.get('gradient_audit_every_steps',888))==0:
                groups={'E':[],'G_C':[]}
                for name,p in identity.named_parameters():
                    if not p.requires_grad:continue
                    is_g=('.cls_head.' in name or '.response_projection.' in name or name.endswith('response_gain'))
                    groups['G_C' if is_g else 'E'].append(p)
                groups['all']=groups['E']+groups['G_C']
                for partition,parameters in groups.items():
                    full=gradient_vector(native_loss,parameters)
                    for kind,real,virtual_loss,exact_loss in gradient_tasks:
                        rg=gradient_vector(real,parameters);vg=gradient_vector(virtual_loss,parameters)
                        eg=gradient_vector(exact_loss,parameters)
                        for label,metrics in (('real_native',gradient_comparison(full,rg)),
                                ('learned_virtual_native',gradient_comparison(full,vg)),
                                ('exact_virtual_native',gradient_comparison(full,eg)),
                                ('learned_virtual_real',gradient_comparison(rg,vg)),
                                ('exact_virtual_real',gradient_comparison(rg,eg))):
                            self.last.update({f'{kind}_{partition}_{label}_{k}':v for k,v in metrics.items()})
                self.last['full_native_gradient_reference']=True
            self.last.update(active=1.,real_IQ_loss=float(real_loss.detach()),virtual_loss=float(virtual.detach()),
                receiver_loss=float(receiver.detach()),auxiliary_total_loss=float(result.detach()),
                effective_virtual_weight=normalization*vw,effective_receiver_weight=normalization*rw,
                physical_source_L_packets=count,source_conditions=2,fit_packets=int(fit.sum()),audit_packets=int(audit.sum()),
                reference_version=self.reference_version,auxiliary_successful_updates=self.updates)
        self.seconds+=time.perf_counter()-start;self.epoch_count+=1
        self.exposure_totals['source_L_physical_packets']+=count
        self.exposure_totals['extra_real_IQ_views']+=6*count
        for k,v in self.last.items():
            if isinstance(v,(float,int)) and math.isfinite(v):self.totals[k]+=v
        return result

    def execution(self):
        return dict(fit_steps=dict(self.fit_steps),identity_steps=dict(self.identity_steps),
            attempted_identity_steps=dict(self.attempted_identity_steps),receiver_relations_total=self.receiver_relations_total,
            exposure_totals=dict(self.exposure_totals),reference_versions=self.reference_version,
            target_access=False,calibration_scope='physical source-L auxiliary holdout',
            gated_off_is_valid_negative=True)

    def checkpoint(self,epoch):
        return dict(schema='state_action_training_only_v1',epoch=epoch,config=self.config,
            auxiliary=self.auxiliary.state_dict(),optimizer=None if self.optimizer is None else self.optimizer.state_dict(),
            reference_version=self.reference_version,private_generator=self.generator.get_state(),
            mechanism_execution=self.execution(),checkpoint_sources=[],resume_supported=False,
            inference_uses_auxiliary=False,target_access=False,cost=self.cost)

    def annotate(self,rows):
        if not rows:return
        epoch=int(rows[-1]['epoch']);values=dict(self.last)
        values.update({k:v/max(1,self.epoch_count) for k,v in self.totals.items()})
        values.update(auxiliary_active_steps_epoch=self.epoch_count,auxiliary_training_seconds=self.seconds,
            auxiliary_parameters=self.cost['auxiliary_parameters'],auxiliary_successful_updates=self.updates)
        rows[-1].update({'state_action_'+k:v for k,v in values.items()})
        self.output_root.mkdir(parents=True,exist_ok=True)
        self.cost['mechanism_execution']=self.execution()
        self.cost['process_peak_cuda_allocated_bytes']=(torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0)
        self.cost['peak_cuda_scope']='whole training process since startup, not isolated auxiliary increment'
        self.cost['reference_parameter_buffer_bytes']=(0 if self.reference is None else
            sum(t.numel()*t.element_size() for t in list(self.reference.parameters())+list(self.reference.buffers())))
        self.cost['auxiliary_parameter_buffer_bytes']=sum(t.numel()*t.element_size()
            for t in list(self.auxiliary.parameters())+list(self.auxiliary.buffers()))
        self.cost['auxiliary_optimizer_state_bytes']=(0 if self.optimizer is None else sum(
            v.numel()*v.element_size() for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v)))
        try:
            import psutil
            self.cost['process_resident_memory_bytes']=psutil.Process().memory_info().rss
        except ImportError:self.cost['process_resident_memory_bytes']=None
        self.cost['additional_inference_transmission_bytes']=0
        self.cost['FLOPs']=None
        self.cost['receiver_contribution_bytes']=(0 if self.receiver is None else sum(
            v.numel()*v.element_size() for entry in self.receiver.entries.values() for v in entry.values() if torch.is_tensor(v)))
        torch.save(self.checkpoint(epoch),self.output_root/('auxiliary_final.pth' if epoch==self.config.get('epochs',200) else 'auxiliary_latest.pth'))
        (self.output_root/'auxiliary_cost.json').write_text(json.dumps(self.cost,indent=2),encoding='utf-8')
        print('STATE_ACTION '+json.dumps(dict(values,epoch=epoch,arm=self.arm)),flush=True)
        self.totals.clear();self.epoch_count=0


def instrument_train(native,callback):
    original=native.train;code=inspect.getsource(original)
    marker='            loss_is_finite = bool(torch.isfinite(loss.detach()).item())'
    if code.count(marker)!=1:raise RuntimeError('Native objective insertion not unique')
    insertion=(
        '            _state_leo = (concat_sat_ce_view.x if concat_sat_ce_view is not None else\n'
        '                x_l_main[labeled_clean_count:] if len(x_l_main) == 2*labeled_clean_count else None)\n'
        '            _state_loss = _state_action_step(model, ema_model, r3_x_l,\n'
        '                y_l[:labeled_clean_count], receiver_l_base, epoch, r3_optimizer_steps,\n'
        '                days=day_l_base, physical_ids=stable_sample_keys(_meta_from_extra(extra_l) or {}),\n'
        '                leo_x=_state_leo, native_loss=loss, leo_applied=concat_sat_info.get("applied"))\n'
        '            if _state_loss is not None:\n'
        '                loss = loss + _state_loss\n')
    native.__dict__['_state_action_step']=callback
    exec(compile(code.replace(marker,insertion+marker),inspect.getsourcefile(original)+'::state_action','exec'),native.__dict__)
    return original


@contextmanager
def installed(c,training=True):
    if c.get('checkpoint_sources') or c['state_recipe'].get('resume'):raise ValueError('Identity training scratch-only')
    with native_installed(c) as native:
        if not training:yield native;return
        build=native.build_baseline_model;writer=native._write_ssdg_epoch_telemetry;detach=native._detach_log_mapping
        old_step=getattr(native,'_state_action_step',None)
        old_annotation=getattr(native,'_state_action_annotate',None);state={}
        def newbuild(args,device):
            model=build(args,device);state['trainer']=OnlineTrainer(c,model)
            state['trainer'].output_root=Path(getattr(args,'output_dir',c['output_root']))
            print('STATE_ACTION_CONFIG '+json.dumps(dict(recipe=c['state_recipe'],**state['trainer'].cost)),flush=True)
            return model
        def annotate(rows):
            if 'trainer' in state:state['trainer'].annotate(rows)
        def telemetry(cp,jp,rows):
            annotate(rows)
            return writer(cp,jp,rows)
        def metrics(values):
            if 'train/loss' in values and 'trainer' in state:
                values=dict(values,**{'train/state_action_'+k:v for k,v in state['trainer'].last.items()})
            return detach(values)
        native.build_baseline_model=newbuild;native._write_ssdg_epoch_telemetry=telemetry;native._detach_log_mapping=metrics
        native._state_action_annotate=annotate
        original_train=instrument_train(native,lambda *a,**kw:state['trainer'].loss(*a,**kw))
        try:yield native
        finally:
            native.train=original_train;native.build_baseline_model=build
            native._write_ssdg_epoch_telemetry=writer;native._detach_log_mapping=detach
            if old_step is None:del native._state_action_step
            else:native._state_action_step=old_step
            if old_annotation is None:del native._state_action_annotate
            else:native._state_action_annotate=old_annotation
