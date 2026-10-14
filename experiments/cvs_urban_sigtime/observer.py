"""Consume native labeled logits and existing EMA targets; never fit a head."""
from contextlib import contextmanager
from collections import Counter
import gzip,json,math
import torch
from torch.nn import functional as F
from . import design as d
from .channel import PairedChannel,audit
from .objective import paired_margin,soft_urban_ce

PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight',
    'id_backbone.encoder.core.id_backbone.t2.pw.weight',
    'id_backbone.encoder.sequence_pool.views.0.local_projection.weight',
    'id_backbone.encoder.core.id_backbone.cls_head.gain',
    'id_backbone.encoder.core.id_backbone.cls_head.weight')

def gradient(loss,params):
    gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    return [torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(params,gs)]

class Observer:
    def __init__(self,c):
        d.validate(c);self.c=c;self.steps=0;self.eta=0.;self.coverage=Counter();self.counts=Counter()
        self.channel=PairedChannel(c);self.panels=[];self.epoch_records=[];self.last_margins={}
    def initialize(self,model,c,out):
        self.model=model;self.out=out;self.params=[dict(model.named_parameters())[k] for k in PARAMETERS]
        if any(not p.requires_grad for p in self.params):raise ValueError('Existing network/default head must train jointly')
        self.trace=gzip.open(out/'channel_training.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(out/'channel_execution.json',audit())

    @contextmanager
    def native_scope(self,n):
        if n._urban_practical_training_hook is not None or n._source_feature_auxiliary_hook is not None:
            raise ValueError('Another training hook owns this process')
        prep=n._prepare_concat_sat_batch_for_training;move=n.move_batch
        def moved(batch,device):
            value=move(batch,device)
            if bool((value[1]>=0).all()):self.meta=value[2][1]
            return value
        def concatenate(augment,x,y,domain,*,args,epoch,batch_idx):
            if not self.c['ordinary_concat']:return prep(augment,x,y,domain,args=args,epoch=epoch,batch_idx=batch_idx)
            satellite,records=self.channel.apply(x,self.meta,epoch,batch_idx)
            self.concat_records=records
            self.counts['satellite_L_packets']+=len(x);self.counts['satellite_calls']+=1
            info=dict(active=1.,expanded=1.,applied=1.,clean_batch_size=float(len(x)),
                total_batch_size=float(2*len(x)),view_prob=1.,stage_start_epoch=1.,stage_index=0.,
                scenario='same per-sample three-scene nominal',scenario_code=0.)
            return torch.cat((x,satellite)),torch.cat((y,y)),None if domain is None else torch.cat((domain,domain)),None,info
        n.move_batch=moved;n._prepare_concat_sat_batch_for_training=concatenate;n._urban_practical_training_hook=self.observe
        try:yield
        finally:
            n.move_batch=move;n._prepare_concat_sat_batch_for_training=prep;n._urban_practical_training_hook=None

    def observe(self,*,model,args,epoch,batch_idx,clean_iq,labels,clean_meta,clean_output,clean_ce,
                unlabeled_iq,unlabeled_meta,teacher_prob,weak_mask):
        step=(epoch-1)*222+batch_idx
        if model is not self.model or step!=self.steps+1:raise ValueError('Native urban hook update sequence differs')
        self.steps=step;zero=clean_ce*0;adjustment=zero;extra=zero;ssl=zero
        if teacher_prob is not None:self.counts['native_U_packets']+=len(teacher_prob)
        if self.c['ordinary_concat']:
            b=len(labels)//2;y=labels[:b];lc=clean_output['tx_logits'][:b];ls=clean_output['tx_logits'][b:]
            ce_clean=F.cross_entropy(lc,y,label_smoothing=args.label_smoothing)
            ce_sat=F.cross_entropy(ls,y,label_smoothing=args.label_smoothing);records=self.concat_records
            if len(records)!=b or not torch.equal(labels[:b],labels[b:]):raise ValueError('Concatenated real-label pair differs')
        elif self.c['paired']:
            y=labels;lc=clean_output['tx_logits'];ce_clean=clean_ce
            satellite,records=self.channel.apply(clean_iq,clean_meta,epoch,batch_idx,eta=self.eta)
            sat_output=model.forward_identity_only(satellite);ls=sat_output['tx_logits']
            ce_sat=F.cross_entropy(ls,y,label_smoothing=args.label_smoothing)
            # Native clean CE is replaced, not added twice.
            adjustment=.5*(ce_sat-clean_ce)
            self.counts['satellite_L_packets']+=len(y);self.counts['satellite_calls']+=1
            if self.c['margin']:
                q=lc.new_tensor([m['q'] if m['scenario'] in d.URBAN else 0. for m in records])
                extra=paired_margin(lc,ls,y,q);adjustment=adjustment+.5*extra
        else:
            y=labels;lc=clean_output['tx_logits'];ce_clean=clean_ce;ce_sat=zero;records=[];ls=None
        if self.c['urban_ssl'] and teacher_prob is not None:
            if any(k in unlabeled_meta for k in ('tx','tx_i','y_tx','true_tx_i')):
                raise ValueError('U TX truth must remain hidden')
            xu,um=self.channel.apply(unlabeled_iq,unlabeled_meta,epoch,batch_idx,view='U',eta=self.eta)
            lu=model.forward_identity_only(xu)['tx_logits'];ssl=soft_urban_ce(lu,teacher_prob,weak_mask)
            adjustment=adjustment+.16*ssl
            self.counts['satellite_U_packets']+=len(xu);self.counts['source_accepted_U']+=int(weak_mask.sum())
        self.counts['clean_L_packets']+=len(y)
        metrics={'urban/clean_CE':ce_clean.detach(),'urban/satellite_CE':ce_sat.detach(),
            'urban/margin_loss':extra.detach(),'urban/SSL_soft_CE':ssl.detach(),
            'urban/eta':self.eta,'urban/labeled_satellite_count':len(records),
            'urban/added_noise_SNR_mean':sum(m['signal_to_added_noise_db'] or 0 for m in records)/max(1,len(records))}
        rx=clean_meta['rx_i'].detach().cpu().tolist()
        tx_values=y.detach().cpu().tolist()
        sat_correct=(ls.detach().argmax(1)==y).cpu().tolist() if records else []
        clean_correct=(lc.detach().argmax(1)==y).cpu().tolist() if records else []
        for i,m in enumerate(records):
            key=(tx_values[i],int(rx[i]),m['scenario'],m['state_end'],m['receiver_lock_status'])
            self.coverage[key]+=1
        if records:
            margins=ls.gather(1,y[:,None])-ls;clean_margin=lc.gather(1,y[:,None])-lc
            self.trace.write(json.dumps(dict(step=step,epoch=epoch,records=[dict(
                sample_id=m['sample_id'],session_id=m['session_id'],scene=m['scenario'],state=m['state_end'],
                lock=m['receiver_lock_status'],quality=m['quality_snr_db'],added_noise_SNR=m['signal_to_added_noise_db'],
                cfo=m['output_frequency_hz'],kernel_length=m['residual_kernel_length'],kernel_energy=m['residual_kernel_energy'],
                AGC_limited=m['agc_limited'],q=m['q'],hard=m['hard_requested'],hard_accepted=m['hard_accepted'],
                proposals=m['proposal_count'],correct=sat_correct[i],
                clean_correct=clean_correct[i]) for i,m in enumerate(records)]),allow_nan=False)+'\n')
            metrics['urban/accuracy']=sum(sat_correct)/len(sat_correct)
            metrics['urban/clean_accuracy']=sum(clean_correct)/len(clean_correct)
        if step%222==0:
            vectors={'clean':gradient(ce_clean,self.params)}
            if records:vectors['satellite']=gradient(ce_sat,self.params)
            if self.c['margin']:vectors['margin']=gradient(.5*extra,self.params)
            report=[]
            for j,name in enumerate(PARAMETERS):
                a=vectors['clean'][j];aa=float(a.square().sum())
                entry=dict(parameter=name,clean_norm=aa**.5)
                for k,v in vectors.items():
                    b=v[j];bb=float(b.square().sum());dot=float(a@b)
                    entry[k+'_norm']=bb**.5;entry[k+'_cosine']=dot/(aa*bb)**.5 if aa*bb else None
                report.append(entry)
            self.epoch_records.append(dict(epoch=epoch,gradients=report,
                gates=dict(PA_gain=model.id_backbone.encoder.core.id_backbone.cls_head.gain.detach().tanh().cpu().tolist(),
                    reference_gain=model.id_backbone.encoder.response_gain.detach().tanh().cpu().tolist(),
                    signature_gain=(.2*model.id_backbone.encoder.sequence_pool.gain_raw.detach().tanh()).cpu().tolist())))
            d.write(self.out/'gradient_gate_history.json',dict(records=self.epoch_records,head_fit_updates=0))
        return adjustment,metrics

    def epoch(self,model,ctx,rows,out,device):
        epoch=int(rows[-1]['epoch']);self.trace.flush()
        from .panel import evaluate
        result=evaluate(model,device)
        if self.c['curriculum'] and epoch>=3 and len(self.panels)>=2:
            base=self.panels[0];last=self.panels[-1]
            if result['clean_accuracy']>=base['clean_accuracy']-.005 and result['hard_margin']>=last['hard_margin']+.002:
                self.eta=min(.5,self.eta+.05)
        self.panels.append(dict(epoch=epoch,eta_next=self.eta,**result))
        d.write(out/'source_learning_progress.json',dict(records=self.panels,source_only=True,target_feedback=False))
        d.write(out/'training_coverage.json',dict(records=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=n)
            for k,n in sorted(self.coverage.items())],counts=dict(self.counts),all_labeled_satellite_supervised=True))
    def finish(self):
        self.trace.close()
        if self.steps!=6660 or len(self.panels)!=30:raise ValueError('Incomplete E30 native training')
        if self.c['paired'] or self.c['ordinary_concat']:
            if self.counts['satellite_L_packets']!=6660*128:raise ValueError('Missing real-label satellite supervision')
            cells={(k[0],k[1],k[3]) for k in self.coverage}
            if cells!={(tx,rx,state) for tx in range(6) for rx in (1,3,4,6,8) for state in ('LOS','PARTIAL','BLOCKED')}:
                raise ValueError('Incomplete TX x source RX x visibility-state training coverage')
        result=dict(status='VERIFIED',updates=6660,epochs=30,counts=dict(self.counts),
            head_fit_updates=0,U_truth_read=False,training_scenes=list(d.TRAIN_SCENES),eta_final=self.eta)
        d.write(self.out/'training_mechanism_complete.json',result);return result
