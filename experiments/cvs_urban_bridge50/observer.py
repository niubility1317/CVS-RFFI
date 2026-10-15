"""Two native L views remain 1:1; extra HQ is jointly supervised through original CVS."""
from contextlib import contextmanager
from collections import Counter
import gzip,json
import torch
from torch.nn import functional as F
from . import design as d,features as f,objective
from .channel import PairedChannel,audit
from experiments.cvs_urban_branch50 import branches as b,gradient as g
from experiments.cvs_urban_identity50 import features as joint_features


class Observer:
    def __init__(self,c):
        d.validate(c);self.c=c;self.channel=PairedChannel(c);self.steps=0;self.armed=None;self.recipe=None
        self.counts=Counter();self.coverage=Counter();self.panels=[];self.gradients=[];self.updates=[]

    def initialize(self,model,c,out):
        self.model=model;self.out=out;self.boundary=joint_features.boundary_parameters(model)
        self.update_groups=dict(joint_features.parameter_groups(model),**{k:[p] for k,p in self.boundary.items()})
        if any(not p.requires_grad for p in model.parameters()):raise ValueError('All original parameters remain jointly trainable')
        self.previous=self.snapshot();self.trace=gzip.open(out/'channel_training.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(out/'channel_execution.json',audit())

    def snapshot(self):return {key:torch.cat([p.detach().flatten() for p in params]).clone() for key,params in self.update_groups.items()}

    @contextmanager
    def native_scope(self,n):
        if n._urban_practical_training_hook is not None or n._source_feature_auxiliary_hook is not None:
            raise ValueError('Another source loss hook owns this process')
        prep,move=n._prepare_concat_sat_batch_for_training,n.move_batch
        def moved(batch,device):
            value=move(batch,device)
            if bool((value[1]>=0).all()):self.meta=value[2][1]
            return value
        def concatenate(augment,x,y,domain,*,args,epoch,batch_idx):
            if self.armed is not None:raise ValueError('Previous graph was not consumed')
            full,self.hq,self.records=self.channel.apply(x,self.meta,epoch,batch_idx)
            xx=torch.cat((x,full));self.armed=f.NativeCapture(self.model,xx)
            self.counts['satellite_L_packets']+=len(x)
            info=dict(active=1.,expanded=1.,applied=1.,clean_batch_size=float(len(x)),total_batch_size=float(len(xx)),
                view_prob=1.,stage_start_epoch=1.,stage_index=0.,scenario='nominal same-H independent-noise HQ',scenario_code=0.)
            return xx,torch.cat((y,y)),None if domain is None else torch.cat((domain,domain)),None,info
        n.move_batch=moved;n._prepare_concat_sat_batch_for_training=concatenate;n._urban_practical_training_hook=self.observe
        try:yield
        finally:
            if self.armed is not None:self.armed.close();self.armed=None
            n.move_batch=move;n._prepare_concat_sat_batch_for_training=prep;n._urban_practical_training_hook=None

    def observe(self,*,model,args,epoch,batch_idx,clean_iq,labels,clean_meta,clean_output,clean_ce,
                unlabeled_iq,unlabeled_meta,teacher_prob,weak_mask,teacher_model=None):
        step=(epoch-1)*222+batch_idx
        if model is not self.model or step!=self.steps+1 or self.armed is None:
            raise ValueError('Native labeled graph/step sequence differs')
        if teacher_model is None or any(k in (unlabeled_meta or {}) for k in ('tx','tx_i','true_tx_i','y_tx')):
            raise ValueError('Own EMA and hidden U truth required')
        values=self.armed.result();self.armed=None;self.steps=step;self.teacher=teacher_model
        batch=len(labels)//2;y=labels[:batch]
        if len(labels)!=2*batch or not torch.equal(y,labels[batch:]) or len(self.records)!=batch or len(self.hq)!=batch:
            raise ValueError('Original true-label 1:1 clean/satellite pair differs')
        # Additional HQ uses the existing identity-only network/default head.
        # It does not call the native labeled wrapper or increment L step count.
        hq_output,hq_values=f.forward(model,self.hq)
        teacher_logits,target=f.frozen_teacher(teacher_model,torch.cat((clean_iq[:batch],self.hq)))
        jw,sw=objective.weights(teacher_logits[:batch],y,self.records)
        transfer_joint=objective.joint(values['J'][batch:],hq_values['J'],target['J'][:batch],jw)
        if epoch>5:
            if self.recipe is None:raise ValueError('Own epoch5 source calibration missing')
            local,groups=objective.local(values['D'][batch:],target['D'][batch:],self.recipe['scales'],sw)
            weighted_joint=self.recipe['coefficients']['joint']['coefficient']*transfer_joint
            weighted_local=self.recipe['coefficients']['local']['coefficient']*local if self.c['auxiliary']['local_SIG'] else local*0.
        else:
            local=values['D'].sum()*0.;groups={k:local for k in ('increment','feature_area','time_area')}
            weighted_joint=transfer_joint*0.;weighted_local=local
        logits=clean_output['tx_logits'];ce_clean=F.cross_entropy(logits[:batch],y,label_smoothing=args.label_smoothing)
        ce_sat=F.cross_entropy(logits[batch:],y,label_smoothing=args.label_smoothing)
        ce_hq=F.cross_entropy(hq_output['tx_logits'],y,label_smoothing=args.label_smoothing)
        if not torch.allclose(clean_ce,.5*(ce_clean+ce_sat),rtol=2e-5,atol=2e-6):
            raise ValueError('Original 1:1 independently normalized CE changed')
        transfer=weighted_joint+weighted_local;extra=self.c['bridge']['CE_coefficient']*ce_hq+transfer
        self.counts.update(clean_L_packets=batch,HQ_L_packets=batch,teacher_L_clean_packets=batch,teacher_HQ_packets=batch,
            auxiliary_eligible_L=int(jw.sum()))
        if teacher_prob is not None:self.counts['native_U_packets']+=len(teacher_prob)
        if step==1 or step%222==0 or (epoch==6 and batch_idx==1):
            theta=b.feature_parameters(model,clean_output['id_feat_joint'])
            if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Original feature-gradient scope changed')
            gc,ga=g.vectors(.5*ce_sat,theta),g.vectors(transfer,theta)
            boundary=torch.autograd.grad(transfer,list(self.boundary.values()),allow_unused=True,retain_graph=True)
            if any(v is not None and float(v.abs().max())>1e-8 for v in boundary):raise ValueError('Extra transfer fitted W/gains')
            self.gradients.append(dict(epoch=epoch,step=step,global_vs_satellite=g.compare(gc,ga),
                layer_gradients={key:g.compare({key:gc[key]},{key:ga[key]}) for key,_ in theta},
                auxiliary_boundary_norms={key:0. if v is None else float(v.norm()) for key,v in zip(self.boundary,boundary)},
                HQ_CE_updates_head=True,interpretation='Transfer-only boundary; all CE and native U update original head/gains'))
            d.write(self.out/'gradient_gate_history.json',dict(records=self.gradients,source_only=True))
        correct=(logits[batch:].detach().argmax(1)==y).cpu().tolist();rx=clean_meta['rx_i'].detach().cpu().tolist();tx=y.cpu().tolist()
        for i,r in enumerate(self.records):self.coverage[(tx[i],int(rx[i]),r['scenario'],r['state_end'],r['receiver_lock_status'])]+=1
        self.trace.write(json.dumps(dict(step=step,epoch=epoch,records=[dict(sample_id=r['sample_id'],session_id=r['session_id'],
            scene=r['scenario'],state=r['state_end'],lock=r['receiver_lock_status'],quality=r['quality_snr_db'],
            added_noise_SNR=r['signal_to_added_noise_db'],HQ_added_noise_SNR=r['bridge']['bridge_independent_signal_to_added_noise_db'],
            cfo=r['output_frequency_hz'],kernel_length=r['residual_kernel_length'],q=r['q'],correct=correct[i],
            auxiliary_weight=float(jw[i]),local_weight=float(sw[i]),frozen_HQ=True) for i,r in enumerate(self.records)]),allow_nan=False)+'\n')
        return extra,{'urban/clean_CE':ce_clean.detach(),'urban/satellite_CE':ce_sat.detach(),'urban/HQ_CE':ce_hq.detach(),
            'urban/auxiliary_weighted_loss':transfer.detach(),'urban/joint_anchor_loss':transfer_joint.detach(),
            'urban/local_SIG_loss':local.detach(),'urban/local_SIG_weighted_loss':weighted_local.detach(),
            'urban/true_labeled_CE':(clean_ce+.25*ce_hq).detach(),'urban/eta':0.,
            'urban/labeled_satellite_count':batch,'urban/labeled_HQ_count':batch,
            **{'urban/local_'+k:v.detach() for k,v in groups.items()}}

    def epoch(self,model,ctx,rows,out,device):
        from experiments.cvs_urban_branch50.inner_source import panel
        from experiments.cvs_urban_branch50.panel import evaluate,EVAL_EPOCHS
        epoch=int(rows[-1]['epoch']);self.trace.flush();current=self.snapshot()
        self.updates.append(dict(epoch=epoch,groups={k:dict(update_norm=float((v-self.previous[k]).norm()),parameter_norm=float(v.norm())) for k,v in current.items()}))
        self.previous=current;d.write(out/'parameter_update_history.json',dict(records=self.updates))
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            if not hasattr(self,'panel'):self.panel=panel(ctx['inner_dev_loader'])
            if epoch==5:
                from .calibrate import run
                self.recipe=run(model,self.teacher,self.panel,self.c)
                d.write(out/'bridge_calibration.json',self.recipe)
                torch.save(dict(model=model.state_dict(),ema_model=self.teacher.state_dict(),epoch=5,optimizer_steps=1110,
                    source_only=True,target_access=False,initialization='scratch',config=self.c,inner_source_contract=ctx['inner_source_contract']),out/'warmup_epoch5.pth')
                print('SOURCE BRIDGE CALIBRATION PASS own epoch5; source-only; no updates',flush=True)
            if epoch in EVAL_EPOCHS:
                self.panels.append(dict(epoch=epoch,**evaluate(model,self.panel)))
                d.write(out/'source_learning_progress.json',dict(records=self.panels,source_only=True,target_feedback=False,
                    role='L_dev720 only; V health-only',selection='final50 only'))
        d.write(out/'training_coverage.json',dict(records=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(self.coverage.items())],
            counts=dict(self.counts),all_labeled_satellite_supervised=True,all_labeled_HQ_supervised=True))

    def finish(self):
        self.trace.close()
        if self.steps!=11100 or len(self.panels)!=14 or self.recipe is None:raise ValueError('Incomplete scratch E50 bridge budget')
        expected=('clean_L_packets','satellite_L_packets','HQ_L_packets','teacher_L_clean_packets','teacher_HQ_packets')
        if any(self.counts[k]!=1420800 for k in expected) or self.counts['native_U_packets']!=963900:
            raise ValueError('Matched original U and five labeled view budgets differ')
        cells={(k[0],k[1],k[3]) for k in self.coverage}
        if cells!={(tx,rx,state) for tx in range(6) for rx in (1,3,4,6,8) for state in ('LOS','PARTIAL','BLOCKED')}:
            raise ValueError('Missing true TX/source RX/visibility support')
        result=dict(status='VERIFIED',epochs=50,updates=11100,counts=dict(self.counts),main_forward_packets=9031800,
            head_fit_updates=0,U_truth_read=False,extra_U_feature_objective=False,source_dev_forward_packets=70560,
            source_calibration_forward_packets=5760,coefficients=self.recipe['coefficients'],scales=self.recipe['scales'],
            full_receiver_state_frozen_in_HQ=True,formal_test_changed=False)
        d.write(self.out/'training_mechanism_complete.json',result);return result
