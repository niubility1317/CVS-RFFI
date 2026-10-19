"""Full native continuation accounting; no target access or added forward."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import math
import sys
import torch
from . import design as d, model as api
from .source import ordinary_factory
from .optimizer import native_scope


def continuation_terminal_status(original,**kwargs):
    # Legacy all-mechanism promotion checks do not describe this fixed
    # continuation. Source eligibility is decided by report.source_decision.
    kwargs['mechanism_gates_required']=False
    return original(**kwargs)


class RecordingLoader:
    def __init__(self,loader,role,stream):
        self.loader,self.role,self.stream=loader,role,stream
        self.batches=self.packets=0
    def __getattr__(self,key):return getattr(self.loader,key)
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            x,y,domain,meta=batch
            if set(meta['rx_i'].tolist())-set(d.FIT) or (self.role=='U' and bool((y>=0).any())):
                raise ValueError('Original source membership and hidden U TX required')
            keys=('eq_i','rx_i','day_i','sig_i','base_index')
            ids=list(zip(*(meta[k].tolist() for k in keys)))
            self.batches+=1;self.packets+=len(x)
            self.stream.write(json.dumps(dict(role=self.role,batch=self.batches,ids=ids))+'\n')
            yield batch


class Observer:
    def __init__(self,c):
        self.c=c;self.optimizer=None;self.ordinary_records=[];self.epoch_records=[]
        self.out=Path(c['output_root'])

    def data(self,native,ctx,out):
        self.out.mkdir(parents=True,exist_ok=True)
        self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
        samplers={}
        for key,role,offset in (('train_loader','L',101),('unlabeled_loader','U',202)):
            loader=ctx[key]
            generator=torch.Generator().manual_seed(self.c['model_seed']+offset)
            loader.generator=generator
            if isinstance(loader.sampler,torch.utils.data.RandomSampler):loader.sampler.generator=generator
            samplers[role]=dict(seed=self.c['model_seed']+offset,loader_generator=True,
                sampler=type(loader.sampler).__name__,sampler_generator=isinstance(loader.sampler,torch.utils.data.RandomSampler),
                batch_size=loader.batch_size,drop_last=loader.drop_last,reset_at_segment_start=True)
            ctx[key]=RecordingLoader(loader,role,self.sequence)
        self.loaders={role:ctx[key] for key,role in (('train_loader','L'),('unlabeled_loader','U'))}
        d.write(self.out/'sampler_initialization.json',dict(roles=samplers,all_rows_same_role_seeds=True,
            parent_sampler_state_restored=False,actual_full_sequences='training_sequence.jsonl.gz',
            RNG_whole_trajectory_identical_claim=False))

    def initialize(self,m,c,out):
        self.model=m
        named=dict(m.named_parameters())
        self.original_names=tuple(m._dta_original_parameter_names)
        self.initial_parameters={k:p.detach().cpu().clone() for k,p in named.items()}
        self.initial_original_state={k:v.detach().cpu().clone() for k,v in self.parent['model'].items()}
        self.frozen_names=tuple(k for k,p in named.items() if not p.requires_grad)
        if len(m.state_dict())!=214 or len(m._rawiq_state_keys)!=8 or sum(p.numel() for p in named.values() if p.requires_grad)!=4256:
            raise ValueError('Frozen F206 plus the raw-IQ adapter8/4256 state is required')
        if any(p.requires_grad for k,p in named.items() if '.cls_head.' in k):
            raise ValueError('Every original classifier parameter must stay frozen')
        if any(not torch.equal(m.state_dict()[k].detach().cpu(),value) for k,value in self.parent['model'].items()):
            raise ValueError('Actual original F state must equal the strict own B0 parent')
        self.student_offset=dict(train=m.id_backbone.encoder._full_train_packets,eval=m.id_backbone.encoder._full_eval_packets)
        d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items()
            if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
        d.write(self.out/'actual_channel.json',m._dta_channel.audit())
        d.write(self.out/'parameter_scope.json',dict(original_parameter_names=self.original_names,
            frozen_parameter_names=self.frozen_names,trainable_parameter_names=[k for k,p in named.items() if p.requires_grad],
            frozen_head=True,adapter_enabled=c['adapter']['enabled'],parent_state_exact=True))

    def capture_ordinary(self,raw,ordinary,labels,step):
        return api.capture_L(self.model,raw,ordinary,labels,step)

    def optimizer_created(self,opt):
        if opt.state:raise ValueError('New segment AdamW state must start empty')
        d.write(self.out/'optimizer_initialization.json',dict(reset=True,state_entries=0,
            schedule=self.c['optimizer']['schedule'],epochs=self.c['epochs'],
            groups=[dict(lr=g['lr'],eps=g['eps'],weight_decay=g['weight_decay'],parameters=sum(p.numel() for p in g['params'])) for g in opt.param_groups],
            mode=opt._sinc_unit_policy['mode'],head_trainable=False))

    @contextmanager
    def native_scope(self,native):
        deepcopy,factory,aug=native.deepcopy,native.make_augmentor,native.configure_augmentor_for_epoch
        terminal=native._resolve_phase1_terminal_status
        self.ordinary_factory=ordinary_factory(factory,self.ordinary_records,self.capture_ordinary)
        native.make_augmentor=self.ordinary_factory
        parent_args=SimpleNamespace(**self.parent['args'])
        parent_aug_base=native.build_aug_base_cfg(parent_args)
        self.ordinary_schedule_calls=[]
        def configured(obj,cfg,epoch,args):
            result=aug(obj,parent_aug_base,13,parent_args)
            self.ordinary_schedule_calls.append(dict(local_epoch=len(self.ordinary_schedule_calls)+1,
                native_requested_epoch=int(epoch),actual_parent_augmentation_epoch=13))
            return result
        native.configure_augmentor_for_epoch=configured
        def copied(value,*args,**kwargs):
            result=deepcopy(value,*args,**kwargs)
            if value is getattr(self,'model',None):
                # Original EMA is the parent's own 206-key model. Only newly
                # attached adapter keys come from this student's initialization.
                merged=dict(self.parent['ema_model'])
                for key,tensor in self.model.state_dict().items():
                    if key not in merged:merged[key]=tensor.detach().clone()
                if set(merged)!=set(result.state_dict()):raise ValueError('Own EMA original plus adapter key set differs')
                if len(merged)!=214 or set(merged)-set(self.parent['ema_model'])!=set(self.model._rawiq_state_keys):
                    raise ValueError('Own EMA must preserve original206 and add exactly the student raw-IQ eight keys')
                result.load_state_dict(merged,strict=True)
                self.ema=result
                enc=result.id_backbone.encoder
                self.ema_offset=dict(train=enc._full_train_packets,eval=enc._full_eval_packets)
                self.ema_raw_offset=api.adapters.adapter_state_report(result)
                d.write(self.out/'EMA_initialization.json',dict(parent_EMA_restored=True,
                    original_state_keys=len(self.parent['ema_model']),added_adapter_keys=sorted(set(merged)-set(self.parent['ema_model'])),
                    original_state_strict=True,adapter_initialization='identical student initialization',
                    local_successful_update_start=0,parent_successful_updates=self.c['parent_updates'],
                    startup_average=False,decay=self.c['ownEMA']['decay']))
            return result
        native.deepcopy=copied
        native._resolve_phase1_terminal_status=lambda **kwargs:continuation_terminal_status(terminal,**kwargs)
        try:
            with native_scope(self):yield
        finally:
            native.deepcopy,native.make_augmentor,native.configure_augmentor_for_epoch=deepcopy,factory,aug
            native._resolve_phase1_terminal_status=terminal
            if hasattr(self,'sequence'):self.sequence.close()

    def _assert_frozen(self):
        params=dict(self.model.named_parameters())
        if any(params[k].requires_grad or params[k].grad is not None or not torch.equal(params[k].detach().cpu(),self.initial_parameters[k]) for k in self.frozen_names):
            raise ValueError('A frozen classifier or original F parameter changed')
        state=self.model.state_dict()
        if any(not torch.equal(state[k].detach().cpu(),v) for k,v in self.initial_original_state.items()):
            raise ValueError('Every original F206 state tensor must remain unchanged')

    def epoch(self,m,ctx,rows,out,device):
        c=self.c;ep=len(rows);n=ep*c['steps_per_epoch'];U=ep*c['data']['U']
        L=n*128*(2 if c['As'] else 1)
        enc=m.id_backbone.encoder;ema=self.ema.id_backbone.encoder
        records=m._dta_records[(ep-1)*c['steps_per_epoch']:n]
        if (m._dta_steps!=n or m._feature_l_steps!=n or len(m._dta_records)!=n or m._dta_L_packets!=L
                or len(self.ordinary_records)!=n or self.ordinary_factory.calls!=1
                or len(self.ordinary_schedule_calls)!=ep or len(records)!=c['steps_per_epoch']
                or m._dta_witness['raw'] is not None
                or enc._full_train_packets-self.student_offset['train']!=L+U
                or ema._full_train_packets-self.ema_offset['train']!=0
                or ema._full_eval_packets-self.ema_offset['eval']!=U
                or m._dta_channel.calls!=n or m._dta_channel.packets!=n*128
                or m._dta_teacher_packets!={'Tc':n*128,'Ts':n*128}
                or self.optimizer._sinc_unit_steps!=n or self.optimizer._sinc_physical_clip_calls!=n
                or self.loaders['L'].batches!=n or self.loaders['L'].packets!=n*128
                or self.loaders['U'].batches!=n or self.loaders['U'].packets!=U):
            raise ValueError('Actual complete native L/U/EMA/teacher/channel/optimizer budget differs')
        for rec in records:
            expected=(c['loss']['clean_CE_weight']*rec['clean_CE']+c['loss']['satellite_CE_weight']*rec['As_CE']
                +c['loss']['clean_KD_weight']*rec['clean_KL']+c['loss']['satellite_KD_weight']*rec['satellite_KL'])
            if not math.isclose(rec['total_L'],expected,rel_tol=2e-5,abs_tol=2e-5):
                raise ValueError('Actual true CE plus separate masked teacher KL formula differs')
            if (rec['first_input']!=c['first_view'] or rec['encoder_first_calls']!=1
                    or rec['extra_encoder_calls']!=int(c['As']) or rec['encoder_first_MSE']!=0.
                    or rec['hook_raw_MSE']!=0. or not rec['Tc_same_first_IQ'] or not rec['Ts_same_fullAs_IQ']
                    or rec['channel_generated_packets']!=128 or rec['Tc_packets']!=128 or rec['Ts_packets']!=128):
                raise ValueError('Actual full L/teacher view witness differs')
            if (rec['raw_n']!=128 or rec['As_n']!=128*int(c['As'])
                    or not 0<=rec['raw_correct']<=rec['raw_n'] or not 0<=rec['As_correct']<=rec['As_n']
                    or rec['online_readout_status']!='TRAIN_STOCHASTIC_NOT_FROZEN_FIT'
                    or not rec['student_training'] or not rec['grad_enabled'] or rec['accuracy_extra_NN']):
                raise ValueError('Online fit counts must come from the actual stochastic training logits')
        self._assert_frozen()
        teachers=m.__dict__['_dta_teachers'];teachers.assert_frozen()
        raw_actual=api.adapters.adapter_state_report(m)
        ema_raw_actual=api.adapters.adapter_state_report(self.ema)
        if (raw_actual['successful_received_packets']!=enc._full_train_packets+enc._full_eval_packets
                or ema_raw_actual['successful_received_packets']-self.ema_raw_offset['successful_received_packets']!=U):
            raise ValueError('One current-IQ raw encoder must accompany every actual student/own-EMA identity forward')
        if teachers.stats()['packets']!={'Tc':n*128,'Ts':n*128}:
            raise ValueError('Actual teacher model forwards differ from the hook packet counters')
        self.epoch_records.append(dict(epoch=ep,cumulative_epoch=c['parent_epochs']+ep,
            optimizer_steps=n,cumulative_optimizer_steps=c['parent_updates']+n,L_student_packets=L,
            U_student_packets=U,own_EMA_eval_packets=U,Tc_packets=n*128,Ts_packets=n*128,
            fresh_As_packets=n*128,ordinary_generated_packets=n*128,ordinary_used_packets=n*128 if c['first_view']=='ordinary' else 0,
            frozen_parameters_verified=len(self.frozen_names),head_frozen=True,
            mean_clean_CE=sum(r['clean_CE'] for r in records)/len(records),mean_As_CE=sum(r['As_CE'] for r in records)/len(records),
            mean_clean_KL=sum(r['clean_KL'] for r in records)/len(records),mean_satellite_KL=sum(r['satellite_KL'] for r in records)/len(records)))
        self.epoch_records[-1]['online_training']=self.online_summary(records)
        d.write(out/'reference_progress.json',self.epoch_records[-1])
        d.write(out/'continuation_epochs.json',self.epoch_records)

    @staticmethod
    def online_summary(records):
        raw_n=sum(r['raw_n'] for r in records);raw_correct=sum(r['raw_correct'] for r in records)
        as_n=sum(r['As_n'] for r in records);as_correct=sum(r['As_correct'] for r in records)
        return dict(status='TRAIN_STOCHASTIC_NOT_FROZEN_FIT',raw_correct=raw_correct,raw_n=raw_n,
            raw_accuracy=None if not raw_n else raw_correct/raw_n,As_correct=as_correct,As_n=as_n,
            As_accuracy=None if not as_n else as_correct/as_n,extra_NN=0,
            note='actual existing model.train logits with dropout; not a frozen training-set fit readout')

    def finish(self):
        c=self.c;m=self.model;out=self.out
        cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
        done=d.read(out/'completion.json');n=c['updates'];self._assert_frozen()
        if (done['epoch']!=c['cumulative_epochs'] or done['segment_epoch']!=c['epochs']
                or done['optimizer_steps']!=n or done['cumulative_optimizer_steps']!=c['cumulative_updates']
                or cp['epoch']!=c['epochs'] or cp['a1_ema_successful_updates']!=n
                or cp['phase1_lineage']!=api.phase1_lineage(c) or len(self.epoch_records)!=c['epochs']
                or cp.get('ema_model') is None or done['head_fit_updates'] or done['target_access']
                or cp['args']['a1_ema_startup_average']):
            raise ValueError('Final parent+segment cumulative lineage or own EMA budget differs')
        state=cp['model'];named=dict(m.named_parameters())
        changed={k:dict(l2=float((state[k].float()-v.float()).norm()),initial_l2=float(v.float().norm()))
            for k,v in self.initial_parameters.items() if not torch.equal(state[k],v)}
        if not changed or any(k in self.original_names for k in changed):raise ValueError('Only the new raw-IQ adapter may update')
        for record in self.ordinary_records:
            key='returned_ordinary_MSE' if c['first_view']=='ordinary' else 'returned_raw_MSE'
            if record[key]!=0. or record['full_native_draws']!=1:raise ValueError('Actual first ordinary/raw input return differs')
        enc=m.id_backbone.encoder;ema=self.ema.id_backbone.encoder
        if (not enc._joint160_seen_calls or enc._joint160_last_shape[1]!=160
                or enc._joint160_seen_packets!=enc._full_train_packets+enc._full_eval_packets):
            raise ValueError('Actual finite encoder returns must prove joint160 without an extra forward')
        for name,records in (('mechanism_steps',m._dta_records),('ordinary_draw_steps',self.ordinary_records),
                             ('unit_steps',self.optimizer._sinc_unit_lr_records)):
            with gzip.open(out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
                for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
        native=sys.modules['SSDG.train_ssdg'];args=SimpleNamespace(**self.parent['args'])
        d.write(out/'ordinary_augmentation_actual.json',dict(use_aug=cp['args']['use_aug'],
            base_config=native.build_aug_base_cfg(args),args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},
            schedule_frozen_at_parent_epoch=13,schedule_calls=self.ordinary_schedule_calls,
            first_L_view=c['first_view'],ordinary_calls=n,ordinary_generated_packets=n*128,
            ordinary_used_packets=n*128 if c['first_view']=='ordinary' else 0,
            ordinary_shadow_packets=n*128 if c['first_view']=='raw' else 0,
            practical_input='exact captured raw L',U_augmentor_wrapped=False,U_weak_strong_unchanged=True,
            whole_training_RNG_identical_claim=False))
        teachers=m.__dict__['_dta_teachers'];teachers.assert_frozen()
        d.write(out/'teacher_identity.json',teachers.identity())
        d.write(out/'teacher_calibration_interface.json',teachers.calibration_requirements())
        result=dict(status='RAW_IQ_ADAPTER_SOURCE_TRAINED',config=c,phase1_lineage=cp['phase1_lineage'],
            steps=n,segment_epochs=c['epochs'],parent_epochs=c['parent_epochs'],cumulative_epochs=c['cumulative_epochs'],
            parent_optimizer_steps=c['parent_updates'],optimizer_steps=n,cumulative_optimizer_steps=c['cumulative_updates'],
            changed_parameters=changed,frozen_parameters=len(self.frozen_names),head_frozen=True,
            F_frozen=c['adapter']['enabled'],Sinc_updates=0 if c['adapter']['enabled'] else n,
            trainable_parameters=sum(p.numel() for p in named.values() if p.requires_grad),
            total_parameters=sum(p.numel() for p in named.values()),
            L_packets=m._dta_L_packets,student_train_packets=enc._full_train_packets,
            student_eval_packets=enc._full_eval_packets,student_counter_start=self.student_offset,
            U_hidden_steps=n,U_packets=self.loaders['U'].packets,
            own_EMA_train_packets=ema._full_train_packets-self.ema_offset['train'],
            own_EMA_eval_packets=ema._full_eval_packets-self.ema_offset['eval'],own_EMA_inherited_counter_offset=self.ema_offset,
            own_EMA_successful_updates=n,own_EMA_cumulative_updates=c['cumulative_updates'],
            teacher_packets=teachers.stats()['packets'],teacher_execution=teachers.stats(),channel=m._dta_channel.summary(),
            ordinary_packets=n*128,ordinary_used_packets=n*128 if c['first_view']=='ordinary' else 0,
            ordinary_shadow_packets=n*128 if c['first_view']=='raw' else 0,
            fresh_As_packets=m._dta_channel.packets,student_As_packets=n*128 if c['As'] else 0,
            teacher_only_As_packets=0 if c['As'] else n*128,
            joint160_actual=dict(calls=enc._joint160_seen_calls,packets=enc._joint160_seen_packets,
                last_shape=enc._joint160_last_shape,finite_checked_each_forward=True,extra_verification_NN=0),
            epochs=self.epoch_records,source_metrics=done['final_source_metrics'],
            online_training=self.online_summary(m._dta_records),
            online_training_epochs=[dict(epoch=r['epoch'],**r['online_training']) for r in self.epoch_records],
            raw_IQ_adapter=dict(parameters=4256,state_tensors=8,state_keys=list(m._rawiq_state_keys),
                input='current_forward_IQ',old_local_adapter=False,original_F206_unchanged=True,
                actual=api.adapters.adapter_state_report(m),ownEMA_actual=api.adapters.adapter_state_report(self.ema),
                ownEMA_counter_start=self.ema_raw_offset,
                cost_scope='one auxiliary raw convolution path per identity forward; no additional original F forward'),
            native_U='original hiddenTX batch_neighbor/strong/ownEMA',crossbatch_U_cache=False,
            source_sequence='training_sequence.jsonl.gz',head_fit_updates=0,target_access=False)
        d.write(out/'training_result.json',result)
        return result


def train(c):
    from .source import segment_train
    observer=Observer(c)
    segment_train(c,observer=observer)
    return observer.finish()
