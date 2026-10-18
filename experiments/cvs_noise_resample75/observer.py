"""Record actual snapshots separately from the two existing As forward slots."""
from contextlib import contextmanager
import gzip
import json
import sys
from pathlib import Path
from . import design as d, model
from .source import parent_payload, segment_train
from experiments.cvs_satellite_dropout18.observer import Observer as Original

PAIR_MEANS=('original_mixed_KL','ordinary_half_KL','satellite_half_KL','used_mixed_KL',
    'removed_satellite_KL','satellite_CE0','satellite_CE1','As_IQ_pair_MSE')

def joint(counter):
    return [dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v)
        for k,v in sorted(counter.items())]

class Observer(Original):
    def __init__(self,c):
        self.c=d.validate(c);self.optimizer=None;self.out=Path(c['output_root'])
        self.parent=parent_payload(c);self.physics_records=0;self.physics_file=None

    def initialize(self,m,c,out):
        super().initialize(m,c,out)
        self.physics_file=gzip.open(self.out/'full_as_pair_physics.jsonl.gz','xt',encoding='utf8',compresslevel=3)
        model.physics_sink(m,self.write_physics)
        actual=dict(config=c,first_channel=m._full_channel.audit(),
            second_channel=c['noise_pair'],modules={k:str(v.__file__) for k,v in sys.modules.items() if k in (
                'experiments.cvs_noise_resample75.model','experiments.cvs_noise_resample75.channel',
                'experiments.cvs_full_channel_pair68.model','SSDG.train_ssdg','DataAugmentation',
                'leo_practical.channel','leo_practical.residual')})
        d.write(self.out/'noise_pair_actual.json',actual)
        print(json.dumps(dict(stage='ACTUAL_NATIVE_NOISE_PAIR',**actual)),flush=True)

    def write_physics(self,record):
        n2=64 if self.c['arm']=='N01' else 0
        if (record['step']!=self.physics_records+1 or len(record['view0'])!=64
                or len(record['view1'])!=64 or record['secondary_noise_redraws']!=n2
                or record['physical_latent_draws']!=64 or self.physics_file is None):
            raise ValueError('Exactly two ordered As64 slots with declared actual snapshots required')
        self.physics_file.write(json.dumps(record,allow_nan=False)+'\n');self.physics_records+=1

    @contextmanager
    def native_scope(self,native):
        try:
            with super().native_scope(native):yield
        finally:
            if hasattr(self,'model'):model.physics_sink(self.model)
            if self.physics_file is not None:self.physics_file.close();self.physics_file=None

    def epoch(self,m,ctx,rows,out,device):
        super().epoch(m,ctx,rows,out,device)
        n=len(rows)*d.STEPS;n2=n*64 if self.c['arm']=='N01' else 0
        if (len(m._noise_records)!=n or len(m._pair_records)!=n or m._pair_generated!=n*64
                or m._noise_generated!=n*64+n2 or m._noise_secondary_generated!=n2
                or m._noise_latent_draws!=n*64 or m._noise_slots!=n*128
                or sum(m._noise_coverage.values())!=n*128 or self.physics_records!=n
                or m._pair_sessions!=m._full_channel.sessions):
            raise ValueError('Actual noise/latent/forward/session budget differs')
        if self.physics_file is not None:self.physics_file.flush()
        d.write(out/'noise_pair_progress.json',self.pair_summary())

    def pair_summary(self):
        m=self.model;r=m._noise_records;n=len(r)
        return dict(method=self.c['method'],arm=self.c['arm'],steps=n,
            **{k+'_mean':sum(t[k] for t in r)/n if n else 0. for k in PAIR_MEANS},
            generated_As0=m._full_channel.packets,generated_As1=m._noise_secondary_generated,
            generated_As_total=m._noise_generated,physical_latent_draws=m._noise_latent_draws,
            logical_second_slots=m._pair_generated,As_forward_NN=m._noise_slots,L_NN=n*768,
            total_L_CE_scale=1.25,mixed_group_weight=.5,As_fraction=.5,As_CE_coefficient=.25,
            per_pass_As_CE_coefficient=.125,ordinary_raw_KL_unchanged=True,As_KL=False,
            original_dropout=True,dropout_bypassed_elements=sum(t['bypassed_elements'] for t in r),
            same_hardware_steps=sum(t['same_hardware'] for t in r),full_physics_steps=self.physics_records,
            view0_namespace_unchanged=True,first_noisestream_unchanged=True,nonnoise_replayed=True,
            AGC_per_view_native=True,second_from_same_raw=True,second_from_first_As=False,
            session_count=len(m._pair_sessions),extra_NN=0,extra_U_NN=0,extra_parameters=0,
            extra_losses=0,U_truth_read=False,target_access=False)

    def finish(self):
        result=super().finish();m=self.model;n2=142080 if self.c['arm']=='N01' else 0
        if (len(m._noise_records)!=2220 or self.physics_records!=2220
                or m._noise_generated!=142080+n2 or m._noise_secondary_generated!=n2
                or m._noise_latent_draws!=142080 or m._noise_slots!=284160
                or result['L_NN']!=1704960 or result['Ag_packets']
                or result['extra_U_NN'] or result['head_fit_updates'] or result['U_truth_read']):
            raise ValueError('Fixed six-L conditional noise cost differs')
        with gzip.open(self.out/'full_as_pair_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
            for r in m._noise_records:f.write(json.dumps(r,allow_nan=False)+'\n')
        summary=self.pair_summary()
        result.update(new_epochs=10,parent_epochs=20,cumulative_epochs=30,new_updates=2220,
            parent_updates=4440,cumulative_updates=6660,L_As_total=142080+n2,L_As_forward_NN=284160,
            L_As_note='L_As is first-view snapshots; total counts only actual native calls, not copied second slots',
            noise_pair_summary=summary,full_as_pair_summary=summary,
            TX_RX_scene_state_lock=joint(m._full_coverage+m._pair_coverage),
            TX_RX_scene_state_lock_view0=joint(m._full_coverage),
            TX_RX_scene_state_lock_view1=joint(m._pair_coverage),source_RX_holdout=None,
            L_channel_second=dict(calls=2220 if n2 else 0,packets=n2,logical_slots=142080,
                sessions=m._pair_sessions,session_count=len(m._pair_sessions),dropped=0,
                confidence_filter=False,all_original_source_RX=list(d.RX),formal_test_changed=False,
                strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in sorted(m._pair_strata.items())]),
            parent_history_difference=self.c['parent_history_difference'])
        result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
        d.write(self.out/'result.json',result)
        return result

def train(c):
    o=Observer(c);segment_train(c,design=d,model_api=model,observer=o);return o.finish()
