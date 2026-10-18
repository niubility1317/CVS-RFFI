"""Preserve original L/U/EMA training and count extra teaching forwards once."""
import gzip,json,sys
from types import SimpleNamespace
from pathlib import Path
from . import design as d,model
from .source import parent_payload,teacher_payload,segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original

class Observer(Original):
    def __init__(self,c):
        self.c=d.validate(c);self.optimizer=None;self.out=Path(c['output_root']);self.parent=parent_payload(c)

    def initialize(self,m,c,out):
        super().initialize(m,c,out)
        bank=model.prepare_frozen_teachers(m,c,payloads={'B01':self.parent,'B00':teacher_payload(c,'B00')})
        bank.assert_frozen()
        actual=dict(config=c,channel=m._full_channel.audit(),U_channel=m._clean_teacher_channel.audit(),
            frozen_teacher_sources=c['teacher_sources'],frozen_teachers=bank.summary(),
            frozen_teachers_training_only=True,student_parameters=sum(p.numel() for p in m.parameters()),
            student_state_keys=len(m.state_dict()),native_U_teacher='own evolving EMA, unchanged',
            modules={k:str(v.__file__) for k,v in sys.modules.items() if k in [
                'experiments.cvs_clean_teacher65.teaching','experiments.cvs_clean_teacher65.model',
                'experiments.cvs_clean_teacher65.channel','SSDG.train_ssdg','DataAugmentation',
                'leo_practical.channel','leo_practical.residual']})
        d.write(self.out/'clean_teacher_actual.json',actual)
        print(json.dumps(dict(stage='ACTUAL_MODULES_AND_CLEAN_TEACHERS',**actual)),flush=True)

    def epoch(self,m,ctx,rows,out,device):
        super().epoch(m,ctx,rows,out,device)
        if len(m._clean_teacher_records)!=len(rows)*d.STEPS:
            raise ValueError('Exactly one matched extra teaching record per original U update')
        bank=model.teacher_bank(m)
        if bank.summary()['registered_in_student'] or any(t.training or any(p.requires_grad for p in t.parameters()) for t in bank.teachers.values()):
            raise ValueError('Frozen teachers left eval/no-gradient training-only scope')
        progress=d.read(out/'reference_progress.json')
        progress.update(extra_U_NN=m._clean_teacher_packets,
            extra_student_U_NN=m._clean_teacher_packets,
            extra_frozen_teacher_U_NN=bank.summary()['total_forward_NN'],
            frozen_teacher_NN=bank.summary()['forward_NN'],
            U_As=m._clean_teacher_channel.packets,existing_U_only=False,
            native_U_unchanged=True)
        d.write(out/'reference_progress.json',progress)
        d.write(out/'clean_teacher_progress.json',self.teaching_summary())

    def teaching_summary(self):
        records=self.model._clean_teacher_records;n=len(records);packets=sum(r['packets'] for r in records)
        sums={k:sum(r[k] for r in records) for k in ['KL','weighted_KL','native_packets','original_selected','extra_original_selected','extra_student_NN','extra_frozen_teacher_NN']}
        names=('EVOLVING_AS_EMA','FROZEN_AS20_EMA','FROZEN_NOAS20_EMA')
        teachers={}
        for name in names:
            sub=[r['teachers'][name] for r in records]
            teachers[name]=dict(q_mass_sum=sum(t['q_mass'] for t in sub),positive_q_sum=sum(t['positive_q'] for t in sub),
                q_mean=sum(t['q_mass'] for t in sub)/packets if packets else 0.,
                positive_q_fraction=sum(t['positive_q'] for t in sub)/packets if packets else 0.,
                KL_sum=sum(t['KL'] for t in sub),unweighted_KL_sum=sum(t['unweighted_KL'] for t in sub),
                q_histogram=[sum(t['q_histogram'][i] for t in sub) for i in range(5)],
                pseudo_counts=[sum(t['pseudo_counts'][i] for t in sub) for i in range(6)],
                positive_pseudo_counts=[sum(t['positive_pseudo_counts'][i] for t in sub) for i in range(6)])
        selected='EVOLVING_AS_EMA' if self.c['arm']=='ZERO_KL' else self.c['arm']
        bank=model.teacher_bank(self.model)
        return dict(steps=n,arm=self.c['arm'],selected_teacher=selected,packets_sum=packets,
            **{k+'_sum':v for k,v in sums.items()},
            extra_student_NN=sums['extra_student_NN'],extra_frozen_teacher_NN=sums['extra_frozen_teacher_NN'],
            frozen_teacher_NN={key:sum(r['frozen_teacher_NN'][key] for r in records) for key in ('B01','B00')},
            KL_step_mean=sums['KL']/n if n else 0.,weighted_KL_step_mean=sums['weighted_KL']/n if n else 0.,
            KL_packet_sum=sum(r['KL']*r['packets'] for r in records),
            weighted_KL_packet_sum=sum(r['weighted_KL']*r['packets'] for r in records),
            q_mass_sum=teachers[selected]['q_mass_sum'],positive_q_sum=teachers[selected]['positive_q_sum'],
            teacher_statistics=teachers,
            teacher_disagreement={key:sum(r['teacher_disagreement'][key]*r['packets'] for r in records)/packets if packets else 0.
                for key in (records[0]['teacher_disagreement'] if records else ())},
            frozen_teachers=bank.summary(),teacher_sources=self.c['teacher_sources'],
            all_frozen_eval=True,teacher_parameters_registered_in_student=False,
            student_state_keys=len(self.model.state_dict()),student_parameters=sum(p.numel() for p in self.model.parameters()),
            native_U_mask_unchanged=True,all_labeled_As_CE_retained=True,U_TX_truth_read=False,
            source_global_disable=False,student_urban_confidence_gate=False,
            subset_rule='first_min_128_native_U_batch_no_padding',temperature=1.,tau=.95,
            max_weight=.16,ramp_epochs=5,selected_count_normalization=False,
            extra_teacher_training_updates=0,extra_deployment_parameters=0)

    def finish(self):
        result=super().finish();records=self.model._clean_teacher_records
        model.teacher_bank(self.model).assert_frozen()
        if len(records)!=d.UPDATES:raise ValueError('Complete2220 matched teaching records required')
        summary=self.teaching_summary();extra=summary['extra_student_NN'];frozen=summary['extra_frozen_teacher_NN']
        if (extra!=284120 or frozen!=568240 or summary['native_packets_sum']!=567000
                or self.model._clean_teacher_packets!=extra
                or summary['frozen_teacher_NN']!={'B01':284120,'B00':284120}
                or summary['frozen_teachers']['forward_NN']!=summary['frozen_teacher_NN']):
            raise ValueError('Exact first128/tail124 extra U or two frozen teacher cost differs')
        with gzip.open(self.out/'clean_teacher_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
        # Native encoder counters include the added student forward. Remove it
        # once here, then report native+extra separately; frozen teachers never
        # enter the student's encoder or the original EMA counters.
        result['observed_student_train_NN_including_extra']=result['student_train_NN']
        result['student_train_NN']-=extra
        result['native_extra_U_NN']=result['extra_U_NN'];result['native_U_As']=result['U_As']
        result.update(extra_U_NN=extra,extra_student_U_NN=extra,extra_frozen_teacher_U_NN=frozen,
            U_As=extra,U_steps=d.UPDATES,U_channel=self.model._clean_teacher_channel.summary(),
            clean_teacher_summary=summary,new_epochs=10,parent_epochs=20,cumulative_epochs=30,
            new_updates=2220,parent_updates=4440,cumulative_updates=6660,source_RX_holdout=None,
            frozen_teacher_training_updates=0,extra_deployment_parameters=0,
            training_only_teachers=dict(summary['frozen_teachers'],frozen_parameters_and_buffers_unchanged=True),
            TX_RX_scene_state_lock=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v)
                for k,v in sorted(self.model._full_coverage.items())])
        if (result['L_NN']!=1704960 or result['L_As']!=142080 or result['Ag_packets']
                or result['head_fit_updates'] or result['U_truth_read'] or result['native_extra_U_NN']):
            raise ValueError('Original sixL/nominalAs/oldU/default-head budget differs')
        native=sys.modules['SSDG.train_ssdg'];scratch=SimpleNamespace(**self.parent['args'])
        d.write(self.out/'ordinary_augmentation_actual.json',dict(base=native.build_aug_base_cfg(scratch),
            schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},
            schedule_args_epochs=self.parent['args']['epochs'],source='own scratch20 checkpoint args',
            no_ordinary_before_practical=True,RDrop_same_IQ=True))
        result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
        result['training_total_NN']=result['native_NN']+extra+frozen
        d.write(self.out/'result.json',result);return result

def train(c):
    observer=Observer(c);segment_train(c,design=d,model_api=model,observer=observer);return observer.finish()
