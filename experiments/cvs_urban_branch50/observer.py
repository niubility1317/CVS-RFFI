"""Ordinary concat and unchanged native U supervision; one selectable L auxiliary."""
from contextlib import contextmanager
from collections import Counter
import gzip,json
import torch
from torch.nn import functional as F
from . import design as d,branches as b,objective,gradient as g
from .channel import PairedChannel,audit
from experiments.cvs_urban_identity50 import features as joint


class Observer:
    def __init__(self,c):
        d.validate(c);self.c=c;self.steps=0;self.channel=PairedChannel(c)
        self.counts=Counter();self.coverage=Counter();self.panels=[];self.gradients=[];self.updates=[]
        self.armed=None;self.coefficients=None

    def initialize(self,model,c,out):
        self.model=model;self.out=out;self.boundary=joint.boundary_parameters(model)
        self.update_groups=dict(joint.parameter_groups(model),**{k:[p] for k,p in self.boundary.items()})
        if any(not p.requires_grad for p in model.parameters()):raise ValueError('All original parameters must remain jointly trainable')
        self.previous=self.snapshot()
        self.trace=gzip.open(out/'channel_training.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(out/'channel_execution.json',audit())

    def snapshot(self):
        return {key:torch.cat([p.detach().flatten() for p in params]).clone() for key,params in self.update_groups.items()}

    @contextmanager
    def native_scope(self,n):
        if n._urban_practical_training_hook is not None or n._source_feature_auxiliary_hook is not None:
            raise ValueError('Another native loss hook owns this process')
        prep,move=n._prepare_concat_sat_batch_for_training,n.move_batch
        def moved(batch,device):
            value=move(batch,device)
            if bool((value[1]>=0).all()):self.meta=value[2][1]
            return value
        def concatenate(augment,x,y,domain,*,args,epoch,batch_idx):
            if self.armed is not None:raise ValueError('Previous native graph cache was not consumed')
            satellite,self.concat_records=self.channel.apply(x,self.meta,epoch,batch_idx)
            xx=torch.cat((x,satellite));self.armed=b.NativeCapture(self.model,xx)
            self.counts['satellite_L_packets']+=len(x)
            info=dict(active=1.,expanded=1.,applied=1.,clean_batch_size=float(len(x)),
                total_batch_size=float(2*len(x)),view_prob=1.,stage_start_epoch=1.,stage_index=0.,
                scenario='same per-sample three-scene nominal',scenario_code=0.)
            return xx,torch.cat((y,y)),None if domain is None else torch.cat((domain,domain)),None,info
        n.move_batch=moved;n._prepare_concat_sat_batch_for_training=concatenate;n._urban_practical_training_hook=self.observe
        try:yield
        finally:
            if self.armed is not None:self.armed.close();self.armed=None
            n.move_batch=move;n._prepare_concat_sat_batch_for_training=prep;n._urban_practical_training_hook=None

    def extra(self,values):
        position=self.c['auxiliary']['position']
        if position=='none':return sum(values.values())*0.
        if self.coefficients is None:self.coefficients=d.recipe()['coefficients']
        if position=='ALL':
            scales={key:self.coefficients[key]['coefficient'] for key in b.BRANCHES}
            return self.coefficients['ALL']['coefficient']*objective.mixture(values,scales,self.c['auxiliary']['ALL_weights'])
        return self.coefficients[position]['coefficient']*values[position]

    def observe(self,*,model,args,epoch,batch_idx,clean_iq,labels,clean_meta,clean_output,clean_ce,
                unlabeled_iq,unlabeled_meta,teacher_prob,weak_mask,teacher_model=None):
        step=(epoch-1)*222+batch_idx
        if model is not self.model or step!=self.steps+1 or self.armed is None or not self.armed.done:
            raise ValueError('Native labeled concat graph capture/step sequence differs')
        if teacher_model is None or any(k in (unlabeled_meta or {}) for k in ('tx','tx_i','true_tx_i','y_tx')):
            raise ValueError('Own frozen EMA and hidden U truth required')
        values=self.armed.values;self.armed=None;self.steps=step;self.teacher=teacher_model
        batch=len(labels)//2;y=labels[:batch]
        if len(labels)!=2*batch or not torch.equal(y,labels[batch:]) or len(self.concat_records)!=batch:
            raise ValueError('Original real-label 1:1 concat differs')
        student={key:values[key][batch:] for key in (*b.BRANCHES,'J')}
        target_logits,target=b.frozen_teacher(teacher_model,clean_iq[:batch])
        urban=y.new_tensor([r['scenario'] in d.URBAN for r in self.concat_records],dtype=torch.float32)
        reliability=objective.labeled_reliability(target_logits,y)*urban
        losses=objective.losses(student,target,reliability)
        extra=self.extra(losses) if epoch>5 else clean_ce*0.
        logits=clean_output['tx_logits'];lc,ls=logits[:batch],logits[batch:]
        ce_clean=F.cross_entropy(lc,y,label_smoothing=args.label_smoothing)
        ce_sat=F.cross_entropy(ls,y,label_smoothing=args.label_smoothing)
        # Native equal-size concatenated mean CE already equals .5*(clean+sat).
        if not torch.allclose(clean_ce,.5*(ce_clean+ce_sat),rtol=2e-5,atol=2e-6):
            raise ValueError('Native concat CE no longer has the declared 1:1 normalization')
        self.counts.update(clean_L_packets=batch,teacher_L_clean_packets=batch,auxiliary_eligible_L=int(reliability.sum()))
        if teacher_prob is not None:self.counts['native_U_packets']+=len(teacher_prob)
        if step==1 or step%222==0 or (epoch==6 and batch_idx==1):
            theta=b.feature_parameters(model,clean_output['id_feat_joint'])
            if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Actual feature scope changed')
            cg,ag=g.vectors(.5*ce_sat,theta),g.vectors(extra,theta)
            boundary=torch.autograd.grad(extra,list(self.boundary.values()),allow_unused=True,retain_graph=True)
            if any(v is not None and float(v.abs().max())>1e-8 for v in boundary):raise ValueError('Auxiliary leaked to W or fusion gains')
            ts={key:g.vectors(losses[key],theta) for key in ('T','S')}
            self.gradients.append(dict(epoch=epoch,step=step,position=self.c['auxiliary']['position'],
                global_gradient=g.compare(cg,ag),layer_gradients={name:g.compare({name:cg[name]},{name:ag[name]}) for name,_ in theta},
                T_S_gradient_cosine=g.compare(ts['T'],ts['S'])['cosine'],
                auxiliary_boundary_norms={key:0. if value is None else float(value.norm()) for key,value in zip(self.boundary,boundary)},
                max_grad_norm=float(args.max_grad_norm),clipping='native complete-step telemetry records actual clip/optimizer status',
                gains={key:float(p.detach().norm()) for key,p in self.boundary.items() if key!='default_head'}))
            d.write(self.out/'gradient_gate_history.json',dict(records=self.gradients,source_only=True,
                scope='actual global feature extractor excluding default W; absent per-loss gradients zero'))
        correct=(ls.detach().argmax(1)==y).cpu().tolist();rx=clean_meta['rx_i'].detach().cpu().tolist();tx=y.cpu().tolist()
        for i,r in enumerate(self.concat_records):self.coverage[(tx[i],int(rx[i]),r['scenario'],r['state_end'],r['receiver_lock_status'])]+=1
        self.trace.write(json.dumps(dict(step=step,epoch=epoch,records=[dict(
            sample_id=r['sample_id'],session_id=r['session_id'],scene=r['scenario'],state=r['state_end'],
            lock=r['receiver_lock_status'],quality=r['quality_snr_db'],added_noise_SNR=r['signal_to_added_noise_db'],
            cfo=r['output_frequency_hz'],kernel_length=r['residual_kernel_length'],q=r['q'],correct=correct[i],
            auxiliary_weight=float(reliability[i])) for i,r in enumerate(self.concat_records)]),allow_nan=False)+'\n')
        return extra,{'urban/clean_CE':ce_clean.detach(),'urban/satellite_CE':ce_sat.detach(),
            'urban/auxiliary_weighted_loss':extra.detach(),'urban/auxiliary_enabled':float(epoch>5),
            'urban/labeled_satellite_count':batch,'urban/teacher_correct_urban_count':reliability.sum().detach(),
            'urban/eta':0.,**{'urban/'+key+'_direction_loss':v.detach() for key,v in losses.items()},
            **{'urban/'+key+'_variance':v.detach().var(0,unbiased=False).mean() for key,v in student.items()}}

    def epoch(self,model,ctx,rows,out,device):
        from .inner_source import panel
        from .panel import evaluate,EVAL_EPOCHS
        epoch=int(rows[-1]['epoch']);self.trace.flush()
        current=self.snapshot();self.updates.append(dict(epoch=epoch,groups={key:dict(
            update_norm=float((v-self.previous[key]).norm()),parameter_norm=float(v.norm())) for key,v in current.items()}))
        self.previous=current
        d.write(out/'parameter_update_history.json',dict(records=self.updates,
            interpretation='Actual joint AdamW update; includes CE/U/auxiliary/decay, not auxiliary-only causal increment'))
        # Materializing a DataLoader consumes a torch iterator seed even with
        # shuffle=False. Preserve RNG so calibration/health cannot alter draws.
        devices=[device.index or 0] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            if not hasattr(self,'panel'):self.panel=panel(ctx['inner_dev_loader'])
            if epoch==5 and self.c['arm']=='A0':
                checkpoint=dict(model=model.state_dict(),ema_model=self.teacher.state_dict(),epoch=5,
                    optimizer_steps=1110,source_only=True,target_access=False,initialization='scratch',
                    config=self.c,inner_source_contract=ctx['inner_source_contract'])
                torch.save(checkpoint,out/'warmup_epoch5.pth')
                from .calibrate import run,publish
                result=run(model,self.teacher,self.panel)
                result.update(source_checkpoint=str(out/'warmup_epoch5.pth'),commit=(d.ROOT/'release_commit.txt').read_text().strip())
                publish(result,d.calibration_path())
                print('SOURCE CALIBRATION PASS global feature norms; fixed coefficients; query_read=false',flush=True)
            if epoch in EVAL_EPOCHS:
                self.panels.append(dict(epoch=epoch,**evaluate(model,self.panel)))
                d.write(out/'source_learning_progress.json',dict(records=self.panels,source_only=True,target_feedback=False,
                    role='L_dev only; V health-only',selection='final50 only, no candidate choice during training'))
        d.write(out/'training_coverage.json',dict(records=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=n)
            for k,n in sorted(self.coverage.items())],counts=dict(self.counts),all_labeled_satellite_supervised=True))

    def finish(self):
        self.trace.close()
        if self.steps!=11100 or len(self.panels)!=14:raise ValueError('Incomplete E50 position budget/developer observations')
        if any(self.counts[k]!=11100*128 for k in ('clean_L_packets','satellite_L_packets','teacher_L_clean_packets')):
            raise ValueError('Matched labeled/teacher forward budget differs')
        if self.counts['native_U_packets']!=17*56700:raise ValueError('Native U old-gate budget changed')
        cells={(k[0],k[1],k[3]) for k in self.coverage}
        if cells!={(tx,rx,state) for tx in range(6) for rx in (1,3,4,6,8) for state in ('LOS','PARTIAL','BLOCKED')}:
            raise ValueError('Missing TX/source RX/visibility support')
        result=dict(status='VERIFIED',updates=11100,epochs=50,counts=dict(self.counts),head_fit_updates=0,
            U_truth_read=False,extra_U_feature_objective=False,source_dev_forward_packets=14*5040,
            coefficients=d.recipe()['coefficients'],auxiliary_losses=0 if self.c['arm']=='A0' else 1)
        d.write(self.out/'training_mechanism_complete.json',result);return result
