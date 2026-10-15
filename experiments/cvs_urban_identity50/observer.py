"""Paired CE plus one L-only EMA auxiliary loss; no new deployed parameters."""
from collections import Counter
import gzip,json
import torch
from torch.nn import functional as F
from . import design as d,features as f,objective
from .channel import PairedChannel,audit
from .calibrate import group_gradients
from experiments.cvs_urban_sigtime50.observer import Observer as Original


class Observer(Original):
    # Reuse only the native hook/move/concat context; ordinary concat is disabled.
    def __init__(self,c):
        d.validate(c);self.c=c;self.steps=0;self.eta=c['frozen_eta'][0]
        self.coverage=Counter();self.counts=Counter();self.channel=PairedChannel(c)
        self.panels=[];self.epoch_records=[];self.update_records=[]

    def initialize(self,model,c,out):
        self.model=model;self.out=out;self.groups=f.parameter_groups(model)
        self.boundary=f.boundary_parameters(model)
        self.update_groups=dict(self.groups,**{k:[v] for k,v in self.boundary.items()})
        if any(not p.requires_grad for ps in self.update_groups.values() for p in ps):
            raise ValueError('CE must jointly update the original backbone/head/gains')
        self.previous=self.snapshot()
        self.trace=gzip.open(out/'channel_training.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(out/'channel_execution.json',audit())

    def snapshot(self):
        return {k:torch.cat([p.detach().flatten() for p in ps]).clone() for k,ps in self.update_groups.items()}

    def gradients(self,epoch,step,ce,extra,model):
        cg=group_gradients(.5*ce,self.groups)
        ag=group_gradients(self.c['auxiliary']['coefficient']*extra,self.groups)
        aux_boundary=torch.autograd.grad(extra,list(self.boundary.values()),allow_unused=True,retain_graph=True)
        if any(g is not None and float(g.detach().abs().max())>1e-8 for g in aux_boundary):
            raise ValueError('Auxiliary gradient leaked to classifier W or fusion gains')
        cb=torch.autograd.grad(.5*ce,list(self.boundary.values()),allow_unused=True,retain_graph=True)
        entries=[]
        for name,a in cg.items():
            b=ag[name];an=float(a.norm());bn=float(b.norm())
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite branch gradient')
            entries.append(dict(group=name,satellite_CE_weighted_norm=an,auxiliary_weighted_norm=bn,
                ratio=bn/(an+1e-12),cosine=float(a@b)/(an*bn) if an*bn else None))
        self.epoch_records.append(dict(epoch=epoch,step=step,gradients=entries,
            auxiliary_boundary_norms={k:0. if g is None else float(g.detach().norm()) for k,g in zip(self.boundary,aux_boundary)},
            CE_boundary_norms={k:0. if g is None else float(g.detach().norm()) for k,g in zip(self.boundary,cb)},
            gates=dict(PA=model.id_backbone.encoder.core.id_backbone.cls_head.gain.detach().tanh().cpu().tolist(),
                reference=model.id_backbone.encoder.response_gain.detach().tanh().cpu().tolist(),
                signature=(.2*model.id_backbone.encoder.sequence_pool.gain_raw.detach().tanh()).cpu().tolist())))
        d.write(self.out/'gradient_gate_history.json',dict(records=self.epoch_records,head_fit_updates=0,
            denominator='.5 satellite CE; full labeled batch mean',source_only=True))

    def observe(self,*,model,args,epoch,batch_idx,clean_iq,labels,clean_meta,clean_output,clean_ce,
                unlabeled_iq,unlabeled_meta,teacher_prob,weak_mask,teacher_model=None):
        step=(epoch-1)*222+batch_idx
        if model is not self.model or step!=self.steps+1:raise ValueError('Native auxiliary update sequence differs')
        if teacher_model is None:raise ValueError('Own scratch EMA must be available in both native phases')
        if labels.ndim!=1 or bool((labels<0).any()):raise ValueError('Only legal labeled source IQ enters feature distillation')
        if any(k in (unlabeled_meta or {}) for k in ('tx','tx_i','y_tx','true_tx_i')):
            raise ValueError('Hidden U TX truth must remain unavailable')
        self.steps=step;self.eta=self.c['frozen_eta'][epoch-1]
        if teacher_prob is not None:self.counts['native_U_packets']+=len(teacher_prob)
        satellite,records=self.channel.apply(clean_iq,clean_meta,epoch,batch_idx,eta=self.eta)
        with f.capture(model) as values:sat=model.forward_identity_only(satellite)
        z=f.auxiliary_joint(model,sat,values)
        target=f.frozen_teacher(teacher_model,clean_iq)
        reliability=objective.labeled_reliability(target['tx_logits'],labels)
        urban=reliability.new_tensor([m['scenario'] in d.URBAN for m in records])
        reliability=reliability*urban
        head=model.id_backbone.encoder.core.id_backbone.cls_head
        extra=(objective.output_kl(z,target['tx_logits'],reliability,head)
            if self.c['auxiliary']['kind']=='output_kl' else objective.direction(z,target['id_feat_joint'],reliability))
        ce_sat=F.cross_entropy(sat['tx_logits'],labels,label_smoothing=args.label_smoothing)
        adjustment=.5*(ce_sat-clean_ce)+self.c['auxiliary']['coefficient']*extra
        self.counts.update(dict(clean_L_packets=len(labels),satellite_L_packets=len(labels),
            teacher_L_clean_packets=len(labels),satellite_calls=1,
            auxiliary_eligible_L=int(reliability.sum()),urban_L_packets=int(urban.sum())))
        sc=(sat['tx_logits'].detach().argmax(1)==labels).cpu().tolist()
        cc=(clean_output['tx_logits'].detach().argmax(1)==labels).cpu().tolist()
        rx=clean_meta['rx_i'].detach().cpu().tolist();ys=labels.detach().cpu().tolist()
        for i,m in enumerate(records):self.coverage[(ys[i],int(rx[i]),m['scenario'],m['state_end'],m['receiver_lock_status'])]+=1
        weights=reliability.detach().cpu().tolist()
        self.trace.write(json.dumps(dict(step=step,epoch=epoch,records=[dict(
            sample_id=m['sample_id'],session_id=m['session_id'],scene=m['scenario'],state=m['state_end'],
            lock=m['receiver_lock_status'],quality=m['quality_snr_db'],added_noise_SNR=m['signal_to_added_noise_db'],
            cfo=m['output_frequency_hz'],kernel_length=m['residual_kernel_length'],kernel_energy=m['residual_kernel_energy'],
            AGC_limited=m['agc_limited'],q=m['q'],hard=m['hard_requested'],hard_accepted=m['hard_accepted'],
            proposals=m['proposal_count'],correct=sc[i],clean_correct=cc[i],auxiliary_weight=weights[i])
            for i,m in enumerate(records)]),allow_nan=False)+'\n')
        if step==1 or step%222==0:self.gradients(epoch,step,ce_sat,extra,model)
        cosine=(F.normalize(z.detach(),dim=1,eps=objective.EPS)*F.normalize(target['id_feat_joint'],dim=1,eps=objective.EPS)).sum(1)
        return adjustment,dict(zip(('urban/clean_CE','urban/satellite_CE','urban/auxiliary_loss'),
            (clean_ce.detach(),ce_sat.detach(),extra.detach())),**{
            'urban/auxiliary_weighted_loss':self.c['auxiliary']['coefficient']*extra.detach(),
            'urban/eta':self.eta,'urban/accuracy':sum(sc)/len(sc),'urban/clean_accuracy':sum(cc)/len(cc),
            'urban/labeled_satellite_count':len(labels),'urban/teacher_correct_urban_count':reliability.sum().detach(),
            'urban/joint_direction_cosine':cosine.mean(),
            'urban/satellite_joint_variance':z.detach().var(0,unbiased=False).mean(),
            **{'urban/'+name+'_branch_variance':v.detach().var(0,unbiased=False).mean() for name,v in values.items()}})

    def epoch(self,model,ctx,rows,out,device):
        self.source_ctx=ctx;epoch=int(rows[-1]['epoch']);self.trace.flush()
        current=self.snapshot()
        self.update_records.append(dict(epoch=epoch,groups={k:dict(
            update_norm=float((v-self.previous[k]).norm()),parameter_norm=float(v.norm())) for k,v in current.items()}))
        self.previous=current
        d.write(out/'parameter_update_history.json',dict(records=self.update_records,source_only=True,
            interpretation='Total joint training update, not causal auxiliary-only update'))
        from experiments.cvs_urban_sigtime50.panel import evaluate
        result=evaluate(model,device)
        self.eta=self.c['frozen_eta'][min(epoch,49)]
        self.panels.append(dict(epoch=epoch,eta_next=self.eta,**result))
        d.write(out/'source_learning_progress.json',dict(records=self.panels,source_only=True,target_feedback=False,
            curriculum_policy='fixed replay from source A, no advancement from these new metrics'))
        d.write(out/'training_coverage.json',dict(records=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=n)
            for k,n in sorted(self.coverage.items())],counts=dict(self.counts),all_labeled_satellite_supervised=True))

    def finish(self):
        self.trace.close()
        if self.steps!=11100 or len(self.panels)!=50:raise ValueError('Incomplete fixed E50 training')
        for key in ('clean_L_packets','satellite_L_packets','teacher_L_clean_packets'):
            if self.counts[key]!=11100*128:raise ValueError('Missing real-label pair or clean EMA target')
        if self.counts['native_U_packets']!=17*56700:raise ValueError('Native old-gate U phase budget differs')
        cells={(k[0],k[1],k[3]) for k in self.coverage}
        if cells!={(tx,rx,state) for tx in range(6) for rx in (1,3,4,6,8) for state in ('LOS','PARTIAL','BLOCKED')}:
            raise ValueError('Incomplete TX x source RX x visibility coverage')
        result=dict(status='VERIFIED',updates=11100,epochs=50,counts=dict(self.counts),
            head_fit_updates=0,U_truth_read=False,training_scenes=list(d.TRAIN_SCENES),eta_final=self.eta,
            auxiliary=self.c['auxiliary'],auxiliary_losses=1,extra_U_feature_objective=False)
        d.write(self.out/'training_mechanism_complete.json',result);return result
