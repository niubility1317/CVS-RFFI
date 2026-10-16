from pathlib import Path
from experiments.cvs_ordinary_fold20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-raw-anchor-ordinary-RX1-e20-s2026092701-m3-r01'
RELEASE='cvs_F_raw_anchor_ordinary_RX1_e20_20261016_r01'
OWNER='codex/root/F-raw-anchor-ordinary-RX1-E20-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=old.SEED;UPDATES=4440;STEPS=222
ARMS=('A0','A1','A2')
def rows():return [dict(arm=a,held_RX=1,row_id=f'{a}-RX1-s{SEED}') for a in ARMS]
def parent_config():return old.parent_config()
def config(row):
    if row not in rows():raise ValueError('Fixed three source RX1 arms only')
    enabled=row['arm']!='A0';c=old.config(next(r for r in old.rows() if r['arm']==('N1' if enabled else 'N0')))
    return dict(c,schema='F_raw_anchor_matched_fourforward_RX1_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='cvs_raw_anchor_source',
        training_recipe='User-fixed F: SIG_TIME_X3/CENTER_25K/CENTER_UNITS/RDrop/defaultcos scratch20=13L+7U; twoRDpairs/fourLforwards, '+row['arm'],
        baseline_reference='Fresh F A0/A1; old RAW_HZ and E50 F results are historical, no inherited weights',
        combination=dict(SIG_TIME='SIG_TIME_X3',Sinc_coordinates='CENTER_25K',Sinc_optimizer='CENTER_UNITS',s=25000.,Fs=25000000.,
            default_head_joint=True,MEAN_VAR=False,COV=False,ensemble=False),
        F_implementation=dict(source_commit='09554c3a06c976c53efc4dbc61f5c501a1e7b06e',
            source_package='experiments/cvs_rdrop_sinc_units',modules=['coordinate.py','optimizer.py','x3_features.py','descriptors.py'],
            scope='Exact F feature/Sinc math reused; scratch E20 schedule and source fold, no E50 initialization'),
        sinc_optimizer_units=dict(mode='CENTER_UNITS',sample_rate_Hz=25000000.,center_scale_Hz=25000.,
            lr='alpha Fs/s',epsilon='epsilon s/Fs',weight_decay='q -= alpha lambda (q+p0/s); AdamW q decay0',
            gradient_clipping='physical Hz globalclip5',extra_optimizer_steps=0),
        training_mechanism=dict(c['training_mechanism'],ordinary_L=enabled,L_forwards_per_sample=4,expected_L_forward_packets=2273280,
            sinc_coordinates='CENTER_25K',
            first_pair='ordinary' if row['arm']=='A1' else 'raw',second_pair='ordinary' if enabled else 'raw',
            pair_weights=[.5,.5],two_RD_pairs=True,cross_view_consistency=False,actual_ordinary_draws_per_L_step=int(enabled),
            ordinary_policy='unchanged authorized old recipe as research view, no per-operator identity-preserving claim'),
        source_falsification=dict(A2_vs_A0_accuracy_gain_pp_min=1.,A2_vs_A0_macroF1_gain_min=0.,seen_V_mean_drop_pp_max=.5,
            A2_vs_A1_accuracy_drop_pp_max=.5,A2_vs_A1_macroF1_drop_pp_max=.5,
            all_three_real_L_U_sequence_equal=True,all_four_forward_counts=True,finite_complete_updates=True,full_TX_RX_day_coverage=True,
            on_fail='End raw-anchor candidate; no alternate primary/strength/seed/RX rescue',
            on_pass='Only separately registered remaining4 sourceRX same three arms; no target/joint/seed auto-extension'),
        cost_claim='Each new scratch20/4440updates/2273280Lforwards, fourforward matched all3; U/EMA/eval/smoke separately counted;0inheritedweights')
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed raw-anchor configuration differs')
def make_args(c,device='cuda:0'):
    validate(c);a=old.make_args(old.config(next(r for r in old.rows() if r['arm']==('N1' if c['training_mechanism']['ordinary_L'] else 'N0'))),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def prerequisite():
    root=Path(PROJECT)/'runs/20261016-phase1-ordinary-noPA-RX1-e20-s2026092701-m1-r01'
    done=read(root/'completion.json');q=read(root/'source_decision.json')
    if done['status']!='ARTIFACTS_COMPLETE' or done['target_access'] or q['status']!='FAIL':raise ValueError('Completed prior source falsification differs')
    return dict(prior_NoPA_source_status='FAIL',prior_weights_loaded=False,target_access=False)
def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('All matched source controls required')
    if any(v['held_V']['count']!=5400 or v['target_access'] or v['U_truth_read'] or v['head_fit_updates'] or v['L_forward_packets']!=2273280 for v in results.values()):
        raise ValueError('Complete legal fourforward source results required')
    a,b,z=(results[k] for k in ('A2','A0','A1'));h=a['held_V']
    checks=dict(vs_A0_accuracy=100*(h['accuracy']-b['held_V']['accuracy'])>=1.-1e-9,
        vs_A0_macroF1=h['macro_f1']>=b['held_V']['macro_f1']-1e-12,
        seen_health=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy'])<=.5+1e-9,
        A1_accuracy_retained=100*(z['held_V']['accuracy']-h['accuracy'])<=.5+1e-9,
        A1_macroF1_retained=100*(z['held_V']['macro_f1']-h['macro_f1'])<=.5+1e-9,
        complete_health=all(v['complete_health'] for v in results.values()))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        A2_vs_A0_accuracy_gain_pp=100*(h['accuracy']-b['held_V']['accuracy']),A2_vs_A1_accuracy_gain_pp=100*(h['accuracy']-z['held_V']['accuracy']),
        A2_vs_A0_macroF1_gain_pp=100*(h['macro_f1']-b['held_V']['macro_f1']),A2_vs_A1_macroF1_gain_pp=100*(h['macro_f1']-z['held_V']['macro_f1']),
        source_only=True,target_access=False,formal_target_qualified=False,automatic_extension=False,goal_completed=False)
