from pathlib import Path
from experiments.cvs_raw_anchor20 import design as old
from experiments.cvs_white_observation20 import design as white
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-origin-jitter-RX1-e20-s2026092701-m1-r01'
RELEASE='cvs_F_origin_jitter_RX1_e20_20261016_r01'
OWNER='codex/root/F-origin-jitter-RX1-E20-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=old.SEED;UPDATES=4440;STEPS=222;CONTROL=white.CONTROL
ORIGIN=dict(kind='CAUSAL_REFLECT_ORIGIN_JITTER',seed=20261016321,delay_min=0,delay_max=24,distribution='DiscreteUniform0..24',
    formula='G[n]=x[abs(n-d)]; I/Q share d; exact causalreflect delta[d]',per_packet=True,probability=1.,
    schedule='fixed all20epochs, refresh step/view',labels_used=False,hardware_effects=False,post_normalization=False,
    all_other_operators=False,synthetic_prefix_max=24,real_observation_min=232,before_satellite=False)
def rows():return [dict(arm='OJ24',held_RX=1,row_id=f'OJ24-RX1-s{SEED}')]
def parent_config():return old.parent_config()
def control_config():return white.control_config()
def config(row):
    if row not in rows():raise ValueError('One fixed OJ24 source row only')
    c=old.config(old.rows()[2])
    return dict(c,schema='F_origin_jitter_RX1_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
        method='bounded_origin_jitter',baseline_reference=CONTROL.as_posix(),origin_jitter=ORIGIN,
        training_recipe='F scratch20=13L+7U/oldU/defaultcos; .5RD(raw)+.5RD(OJ24), no wholeordinary/As',
        training_mechanism=dict(c['training_mechanism'],L_view_kind=ORIGIN['kind'],second_pair='origin_jitter',ordinary_stream_seed=ORIGIN['seed'],
            no_original_ordinary_operators=True,ordinary_policy='Legacy use_aug switch invokes only OJ24 factory, never original circularshift or otherordinary'),
        source_falsification=dict(vs_raw_accuracy_gain_pp_min=1.,vs_raw_macroF1_gain_pp_min=0.,seen_V_mean_drop_pp_max=.5,
            all_actual_L_U_sequence_equal=True,finite_complete_updates=True,full_TX_RX_day_coverage=True,
            on_fail='CloseOJ24,no delayrange/probability/seed/RX rescue',on_pass='Only separate fixed remaining4sourceRX, no target/joint autoextension'),
        cost_claim='New scratch20/4440updates/2273280Lforwards; matching completed F A0 metrics/traces reused,0weights;U/EMA/eval separate')
def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed OJ24 configuration differs')
def make_args(c,device='cuda:0'):
    validate(c);a=old.make_args(old.config(old.rows()[2]),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    for key in ('dac','pa','amp_scale','phase_rot','cfo','phase_noise','awgn','multipath','dc_offset','bandedge_taper'):setattr(a,'aug_p_'+key,0.)
    a.aug_enable_pa_normal=False;a.aug_enable_class_signature=False;a.aug_p_time_shift=1.;a.aug_max_time_shift=24
    return a
def prerequisite():
    result=white.prerequisite();p=Path(PROJECT)/'runs/20261016-diagnostic-F-phase-origin-source-RX1-s2026092701-m2-r02'
    proof=read(p/'completion.json')
    if proof['status']!='SOURCE_F_PHASE_ORIGIN_COMPLETE' or proof['packet_forwards']!=47884 or proof['target_access'] or proof['optimizer_updates']:
        raise ValueError('Completed source observation attribution required')
    return result
qualify=white.qualify
