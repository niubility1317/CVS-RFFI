from pathlib import Path
from experiments.cvs_raw_anchor20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-white-observation-RX1-e20-s2026092701-m1-r01'
RELEASE='cvs_F_white_observation_RX1_e20_20261016_r01'
OWNER='codex/root/F-white-observation-RX1-E20-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=old.SEED;UPDATES=4440;STEPS=222
CONTROL=Path(PROJECT)/'runs'/old.RUN/f'A0-RX1-s{SEED}'/'source'
NOISE=dict(kind='WHITE_FIXED_ENERGY',signal_to_added_noise_db=30.,added_energy_ratio=.001,
    seed=20261016231,per_packet=True,probability=1.,strength_schedule='fixed all20 epochs; no amplitude sweep',
    formula='n=z*sqrt(.001*sum(x^2)/sum(z^2)); G=x+n; original model normalization unchanged',
    distribution='Gaussian IID direction normalized to a sphere; isotropic fixed energy, not exact Gaussian AWGN',
    zero_power_policy='zero n, preserve packet, count separately',labels_used=False,hardware_effects=False,
    before_satellite=False,boundary_change=False,total_SNR_claim=False)
def rows():return [dict(arm='W30',held_RX=1,row_id=f'W30-RX1-s{SEED}')]
def parent_config():return old.parent_config()
def control_config():return old.config(old.rows()[0])
def config(row):
    if row not in rows():raise ValueError('One fixed white observation source row only')
    c=old.config(old.rows()[2])
    return dict(c,schema='F_white_observation_RX1_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='bounded_white_observation',
        baseline_reference=CONTROL.as_posix(),observation_noise=NOISE,
        training_recipe='F scratch20=13L+7U/oldU/defaultcos; .5 RD(raw)+.5 RD(white30); no wholeordinary or As',
        training_mechanism=dict(c['training_mechanism'],L_view_kind=NOISE['kind'],
            ordinary_policy='ordinary_L is legacy use_aug factory switch only; actual factory solely fixed-energy white30',
            second_pair='white_observation',ordinary_stream_seed=NOISE['seed'],no_original_ordinary_operators=True),
        source_falsification=dict(vs_raw_accuracy_gain_pp_min=1.,vs_raw_macroF1_gain_pp_min=0.,seen_V_mean_drop_pp_max=.5,
            all_actual_L_U_sequence_equal=True,finite_complete_updates=True,full_TX_RX_day_coverage=True,
            on_fail='Close white30 candidate, no SNR/probability/seed/RX rescue',
            on_pass='Only separately register fixed remaining4 sourceRX same F/white30; no target or joint automatic extension'),
        cost_claim='One new scratch20/4440updates/2273280Lforwards; matched completed F A0 result reused, no weight loaded; U/EMA/eval separately counted')
def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed white30 source configuration differs')
def make_args(c,device='cuda:0'):
    validate(c);a=old.make_args(old.config(old.rows()[2]),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    for key in ('dac','pa','time_shift','amp_scale','phase_rot','cfo','phase_noise','multipath','dc_offset','bandedge_taper'):
        setattr(a,'aug_p_'+key,0.)
    a.aug_enable_pa_normal=False;a.aug_enable_class_signature=False
    a.aug_p_awgn=1.;a.aug_snr_min_db=30.;a.aug_snr_max_db=30.
    return a
def prerequisite():
    root=CONTROL.parent.parent;done=read(root/'completion.json');q=read(root/'source_decision.json')
    result=read(CONTROL/'result.json');origin=read(CONTROL/'completion.json')
    if (done['status']!='ARTIFACTS_COMPLETE' or done['target_access'] or q['status']!='FAIL'
        or origin['config']!=control_config() or origin['optimizer_steps']!=4440 or origin['initialization']!='scratch'
        or origin['checkpoint_sources'] or result['L_forward_packets']!=2273280 or result['target_access']):
        raise ValueError('Completed exact F raw/raw control required; prior raw+wholeG remains closed')
    return result
def qualify(candidate,control):
    if any(v['target_access'] or v['U_truth_read'] or v['head_fit_updates'] or v['L_forward_packets']!=2273280 or v['held_V']['count']!=5400 for v in (candidate,control)):
        raise ValueError('Legal matched F source results required')
    h,b=candidate['held_V'],control['held_V'];acc=100*(h['accuracy']-b['accuracy']);f1=100*(h['macro_f1']-b['macro_f1'])
    drop=100*(control['seen_V']['source_val_accuracy']-candidate['seen_V']['source_val_accuracy'])
    checks=dict(held_accuracy=acc>=1.-1e-9,held_macroF1=f1>=-1e-9,seen_health=drop<=.5+1e-9,
        complete_health=bool(candidate['complete_health'] and control['complete_health']))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,accuracy_gain_pp=acc,macroF1_gain_pp=f1,
        seen_V_mean_drop_pp=drop,source_only=True,target_access=False,automatic_extension=False,goal_completed=False)
