"""Count actual L-only calls/coverage without attaching state to the EMA model."""
from collections import Counter
from contextlib import contextmanager
import gzip
import json
import torch
from . import train_design as d
from .train_channel import audit

KEYS=('sample_id','session_id','scenario','state_end','receiver_lock_status','quality_snr_db',
      'signal_to_added_noise_db','output_frequency_hz','residual_kernel_length','residual_kernel_energy',
      'agc_limited','TX','RX','correct')


class Observer:
    def __init__(self,c):self.c=c;self.steps=0;self.channel_packets=0;self.coverage=Counter();self.epochs=[]

    def initialize(self,m,c,out):
        self.m=m;self.out=out;self.initial={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
        self.trace=gzip.open(out/'channel_training.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        d.write(out/'channel_execution.json',audit())

    @contextmanager
    def native_scope(self,n):
        original=n._detach_log_mapping;original_deepcopy=n.deepcopy
        def copied(value,*args,**kwargs):
            result=original_deepcopy(value,*args,**kwargs)
            if value is getattr(self,'m',None):
                self.teacher=result
                result.id_backbone.encoder._mixed_train_packets=0
                result.id_backbone.encoder._mixed_eval_packets=0
            return result
        def logged(values):
            result=original(values)
            if 'train/loss' in result:
                self.steps+=1
                if self.m._mixed_record_step!=self.steps:raise ValueError('Exactly one real L mixed hook per update required')
                records=self.m._mixed_last_records;self.m._mixed_last_records=[]
                for m in records:
                    self.coverage[(int(m['TX']),int(m['RX']),m['scenario'],m['state_end'],m['receiver_lock_status'])]+=1
                self.channel_packets+=len(records)
                if records:self.trace.write(json.dumps(dict(step=self.steps,epoch=(self.steps-1)//222+1,
                    records=[{k:m.get(k) for k in KEYS} for m in records]),allow_nan=False)+'\n')
            return result
        n._detach_log_mapping=logged;n.deepcopy=copied
        try:yield
        finally:n._detach_log_mapping=original;n.deepcopy=original_deepcopy

    def epoch(self,m,ctx,rows,out,device):
        steps=len(rows)*222;factor=self.c['training_mechanism']['L_forwards_per_sample']
        if m._feature_l_steps!=steps or m._mixed_calls!=steps or m._mixed_forward_packets!=steps*128*factor or self.steps!=steps or m._mixed_actual_input is not None:
            raise ValueError('Real L forward/update sequence differs from E20 design')
        self.trace.flush()
        params=dict(m.named_parameters());head='id_backbone.encoder.core.id_backbone.cls_head.'
        if any(not params[head+k].requires_grad for k in ('weight','gain')):raise ValueError('Default head and fusion must remain jointly trainable')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params.values()):raise FloatingPointError('Nonfinite original parameter gradient')
        gates=dict(PA_gain=params[head+'gain'].detach().tanh().cpu().tolist(),
                   reference_gain=params['id_backbone.encoder.response_gain'].detach().tanh().cpu().tolist(),
                   signature_gain=(.2*params['id_backbone.encoder.sequence_pool.gain_raw'].detach().tanh()).cpu().tolist())
        movement={k:float((p.detach().cpu()-self.initial[k]).norm()) for k,p in params.items()
                  if k in (head+'weight',head+'gain') or k.endswith('sequence_pool.local_projection.weight')}
        self.epochs.append(dict(epoch=len(rows),optimizer_steps=steps,L_forward_packets=m._mixed_forward_packets,
                                satellite_packets=self.channel_packets,gates=gates,parameter_movement=movement,
                                student_training_forward_packets=m.id_backbone.encoder._mixed_train_packets,
                                student_eval_forward_packets=m.id_backbone.encoder._mixed_eval_packets,
                                EMA_eval_forward_packets=self.teacher.id_backbone.encoder._mixed_eval_packets))
        d.write(out/'training_mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False))

    def finish(self):
        self.trace.close();expected=self.c['training_mechanism']['expected_L_forward_packets']
        if self.steps!=4440 or self.m._mixed_forward_packets!=expected or self.m._mixed_actual_input is not None:
            raise ValueError('Incomplete actual E20 or mixed forward budget')
        required=568320 if self.c['training_mechanism']['A_s'] else 0
        if self.channel_packets!=required:raise ValueError('Satellite L coverage count differs')
        observed={(tx,rx,scene) for tx,rx,scene,_,_ in self.coverage}
        if required and observed!={(tx,rx,s) for tx in range(6) for rx in (1,3,4,6,8) for s in d.SCENES}:
            raise ValueError('Missing TX x source RX x declared scenario coverage')
        d.write(self.out/'training_mechanism_complete.json',dict(status='VERIFIED',config=self.c,calls=self.steps,
            L_forward_packets=expected,satellite_packets=self.channel_packets,head_fit_updates=0,U_truth_read=False,
            new_U_augmentation=False,default_head_joint_training=True,same_IQ_pairs=True,
            student_training_forward_packets=self.m.id_backbone.encoder._mixed_train_packets,
            student_eval_forward_packets=self.m.id_backbone.encoder._mixed_eval_packets,
            native_student_U_forward_packets=self.m.id_backbone.encoder._mixed_train_packets-expected,
            EMA_eval_forward_packets=self.teacher.id_backbone.encoder._mixed_eval_packets,
            initial_smoke_packets_in_student_eval=2,
            coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(self.coverage.items())]))
