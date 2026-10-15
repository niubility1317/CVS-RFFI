"""Native CE/U unchanged; relative TX identity acts on actual joint anchors."""
from collections import Counter
from contextlib import contextmanager
import json
import torch
from torch.nn import functional as F
from . import design as d,objective
from .channel import PairedChannel
from experiments.cvs_urban_branch50.observer import Observer as Original
from experiments.cvs_urban_branch50 import branches as b,gradient as g


class Observer(Original):
    def __init__(self,c):
        d.validate(c)
        if c['reused']:raise ValueError('Never launch a trainer for archived controls')
        self.c=c;self.steps=0;self.channel=PairedChannel(c);self.armed=None
        self.counts=Counter();self.coverage=Counter();self.panels=[];self.gradients=[];self.updates=[];self.coefficients=None
        self.clean_coverage=Counter();self.clean_seen=set()

    def initialize(self,model,c,out):
        super().initialize(model,c,out)
        self.allowed=set(d.read(out/'native_train_indices.json')['base_indices'])

    @contextmanager
    def native_scope(self,n):
        if n._urban_practical_training_hook is not None or n._source_feature_auxiliary_hook is not None:raise ValueError('Another native hook owns process')
        prepare,move=n._prepare_concat_sat_batch_for_training,n.move_batch
        def moved(batch,device):
            value=move(batch,device)
            if bool((value[1]>=0).all()):self.meta=value[2][1]
            return value
        def concatenate(augment,x,y,domain,*,args,epoch,batch_idx):
            if self.armed is not None:raise ValueError('Previous graph cache not consumed')
            if self.c['arm']=='D0':xx,yy,dd=x,y,domain;self.concat_records=[]
            else:
                satellite,self.concat_records=self.channel.apply(x,self.meta,epoch,batch_idx)
                xx=torch.cat((x,satellite));yy=torch.cat((y,y));dd=None if domain is None else torch.cat((domain,domain))
                self.counts['satellite_L_packets']+=len(x)
            self.armed=b.NativeCapture(self.model,xx)
            active=float(self.c['arm']!='D0')
            return xx,yy,dd,None,dict(active=active,expanded=active,applied=active,clean_batch_size=float(len(x)),
                total_batch_size=float(len(xx)),view_prob=active,stage_start_epoch=1.,stage_index=0.,scenario='nominal3' if active else 'clean_only',scenario_code=0.)
        n.move_batch=moved;n._prepare_concat_sat_batch_for_training=concatenate;n._urban_practical_training_hook=self.observe
        try:yield
        finally:
            if self.armed is not None:self.armed.close();self.armed=None
            n.move_batch=move;n._prepare_concat_sat_batch_for_training=prepare;n._urban_practical_training_hook=None

    def observe(self,*,model,args,epoch,batch_idx,clean_iq,labels,clean_meta,clean_output,clean_ce,
                unlabeled_iq,unlabeled_meta,teacher_prob,weak_mask,teacher_model=None):
        step=(epoch-1)*222+batch_idx
        if step!=self.steps+1 or model is not self.model or self.armed is None or not self.armed.done:raise ValueError('Native identity cache/steps differ')
        if teacher_model is None or any(k in (unlabeled_meta or {}) for k in ('tx','tx_i','y_tx','true_tx_i')):
            raise ValueError('Keep own EMA and original hidden U truth contract')
        values=self.armed.values;self.armed=None;self.steps=step;self.teacher=teacher_model
        batch=128;y=labels[:batch];augmented=self.c['arm']!='D0'
        if len(labels)!=(256 if augmented else 128) or (augmented and not torch.equal(y,labels[batch:])):
            raise ValueError('Real labeled clean/satellite batch contract differs')
        base=clean_meta['base_index'].detach().cpu().tolist()
        if not set(base)<=self.allowed:raise ValueError('L_dev/unregistered physical sample reached CE/semantic objective')
        self.clean_seen.update(base);rx=clean_meta['rx_i'];day=clean_meta['day_i']
        self.clean_coverage.update(zip(y.detach().cpu().tolist(),rx.detach().cpu().tolist(),day.detach().cpu().tolist()))
        # Preserve the archived controls' extra L EMA forward budget. The
        # semantic keys themselves are current-view detached copies, not EMA.
        b.frozen_teacher(teacher_model,clean_iq[:batch]);self.counts.update(clean_L_packets=batch,teacher_L_clean_packets=batch)
        if teacher_prob is not None:self.counts['native_U_packets']+=len(teacher_prob)
        logits=clean_output['tx_logits'];ce_clean=F.cross_entropy(logits[:batch],y,label_smoothing=args.label_smoothing)
        ce_sat=F.cross_entropy(logits[batch:],y,label_smoothing=args.label_smoothing) if augmented else ce_clean*0.
        expected=.5*(ce_clean+ce_sat) if augmented else ce_clean
        if not torch.allclose(clean_ce,expected,rtol=2e-5,atol=2e-6):raise ValueError('Declared clean/augmented CE normalization differs')
        extra=clean_ce*0.;relative=clean_ce*0.;statistics={}
        if augmented:
            q=logits.new_tensor([r['q'] if r['scenario'] in d.URBAN else 0. for r in self.concat_records])
            relative,statistics=objective.semantic(values['J'][:batch],values['J'][batch:],y,rx,day,q,
                bidirectional=self.c['auxiliary']['bidirectional'],margin=self.c['auxiliary']['margin'])
            self.counts['semantic_triplets']+=statistics['triplets'];self.counts['anchors_with_references']+=statistics['anchors_with_references']
            if epoch>5:
                if self.coefficients is None:self.coefficients=d.recipe()['coefficients']
                extra=self.coefficients[self.c['arm']]['coefficient']*relative
        if step==1 or step%222==0 or (epoch==6 and batch_idx==1):
            theta=b.feature_parameters(model,clean_output['id_feat_joint'])
            if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Actual original feature scope changed')
            gc,gs,ga=(g.vectors(loss,theta) for loss in ((.5 if augmented else 1)*ce_clean,.5*ce_sat,extra))
            boundary=torch.autograd.grad(extra,list(self.boundary.values()),allow_unused=True,retain_graph=True)
            if any(v is not None and float(v.abs().max())>1e-8 for v in boundary):raise ValueError('Semantic extra gradient leaked to W/gains')
            self.gradients.append(dict(epoch=epoch,step=step,global_vs_satellite=g.compare(gs,ga),global_vs_clean=g.compare(gc,ga),
                clean_vs_satellite=g.compare(gc,gs),layer_vs_clean={name:g.compare({name:gc[name]},{name:ga[name]}) for name,_ in theta},
                auxiliary_boundary_norms={key:0. if v is None else float(v.norm()) for key,v in zip(self.boundary,boundary)},
                semantic={k:v for k,v in statistics.items() if not torch.is_tensor(v)},max_grad_norm=float(args.max_grad_norm)))
            d.write(self.out/'gradient_gate_history.json',dict(records=self.gradients,source_only=True,default_head_not_fitted=True))
        correct=(logits[batch:].detach().argmax(1)==y).cpu().tolist() if augmented else []
        records=[]
        for i,r in enumerate(self.concat_records):
            self.coverage[(int(y[i]),int(rx[i]),r['scenario'],r['state_end'],r['receiver_lock_status'])]+=1
            records.append(dict(sample_id=r['sample_id'],session_id=r['session_id'],scene=r['scenario'],state=r['state_end'],
                lock=r['receiver_lock_status'],quality=r['quality_snr_db'],added_noise_SNR=r['signal_to_added_noise_db'],
                cfo=r['output_frequency_hz'],kernel_length=r['residual_kernel_length'],q=r['q'],correct=correct[i]))
        self.trace.write(json.dumps(dict(step=step,epoch=epoch,clean_base_index=base,records=records),allow_nan=False)+'\n')
        return extra,{'urban/clean_CE':ce_clean.detach(),'urban/satellite_CE':ce_sat.detach(),
            'urban/auxiliary_weighted_loss':extra.detach(),'urban/semantic_loss':relative.detach(),
            'urban/semantic_forward':statistics.get('urban_to_clean',relative).detach(),
            'urban/semantic_reverse':statistics.get('clean_to_urban',relative).detach(),
            'urban/semantic_triplets':statistics.get('triplets',0),'urban/eta':0.,'urban/auxiliary_enabled':float(epoch>5 and augmented)}

    def finish(self):
        self.trace.close();augmented=self.c['arm']!='D0'
        if self.steps!=11100 or len(self.panels)!=14 or self.clean_seen!=self.allowed:raise ValueError('Incomplete fixed E50/L physical coverage')
        if any(self.counts[k]!=11100*128 for k in ('clean_L_packets','teacher_L_clean_packets')):raise ValueError('Matched L/EMA budget differs')
        if self.counts['satellite_L_packets']!=(11100*128 if augmented else 0) or self.counts['native_U_packets']!=17*56700:
            raise ValueError('Augmentation/old-gated U budget differs')
        if augmented:
            cells={(k[0],k[1],k[3]) for k in self.coverage}
            if cells!={(tx,rx,s) for tx in range(6) for rx in (1,3,4,6,8) for s in ('LOS','PARTIAL','BLOCKED')}:
                raise ValueError('Missing physical training support')
        d.write(self.out/'clean_training_coverage.json',dict(records=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(self.clean_coverage.items())],unique_samples=len(self.clean_seen)))
        result=dict(status='VERIFIED',updates=11100,epochs=50,counts=dict(self.counts),head_fit_updates=0,U_truth_read=False,
            source_dev_forward_packets=14*5040,coefficients=d.recipe()['coefficients'] if augmented else {},
            additional_objectives=1 if augmented else 0,no_extra_U_objective=True,teacher_L_forward_budget_matched=True)
        d.write(self.out/'training_mechanism_complete.json',result);return result
