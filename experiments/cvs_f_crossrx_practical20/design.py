from pathlib import Path
from experiments.cvs_raw_anchor20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_practical_source_RX1_e20_20261016_r01'
OWNER='codex/root/FcrossRX-practical-source-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;STEPS=222;UPDATES=4440;ARMS=('C0','C1')
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
TRAIN_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CHANNEL_SEEDS=dict(train=20261016421,validation=20261016422,receiver=20261016423)
SUPCON=dict(mode='CROSS_RX',levels=['time_down','t2.pool'],temperature=.2,max_weight=.1,warm_epochs=5,ramp_epochs=10,
    positive='same true L TX, different physical RX',negative='different true L TX',exclude='self and sameTX sameRX',
    descriptor='channel mean/populationstd; current L batch detached standardization floor1e-4, sampleL2 eps1e-4',
    forwards='Only the first ordinary-clean RD pair, same IQ twice; no feature constraint on satellite; zero extra forwards',
    source_commit='4df7b94d0fd8ce29056bd8236364e45afa92e07f',new_parameters=0,U_truth=False)

def rows():return [dict(arm=a,held_RX=1,row_id=f'{a}-RX1-s{SEED}') for a in ARMS]
def parent_config():return old.parent_config()

def config(row):
    if row not in rows():raise ValueError('Fixed FcrossRX two-row source matrix only')
    c=old.config(old.rows()[1]);sat=row['arm']=='C1'
    c.update(schema='FcrossRX_practical_source_E20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='FcrossRX_paired_practical',epochs=20,label_epochs=13,pseudo_epochs=7,updates=4440,
        baseline_reference='Fresh C0 of same FcrossRX/four L forwards/data/scratch seed; no historical weights or result as control',
        baseline_reuse=False,supcon=dict(SUPCON),new_Ag=False,As=sat,
        source_falsification=dict(urban_mean_gain_pp_min=3.,low_urban_gain_pp_min=5.,urban_macroF1_gain_pp_min=0.,
            held_clean_drop_pp_max=.5,seen_V_drop_pp_max=.5,suburban_gap_pp_exploratory=2.,
            on_fail='Close fixed paired-As recipe, no loss/strength/scene/seed/RX rescue',on_pass='Separately preregister source confirmation; no automatic target or Ag combination'),
        channel=dict(train_scenes=list(TRAIN_SCENES),scene_distribution='uniform prior; no rejection or quality selection',
            evaluation_scenes=list(SCENES),seeds=dict(CHANNEL_SEEDS),route='original residual/post_sync',EQ=False,G='1',reference_snr_db=30.,boundary='reflect',
            input='original normalized source IQ, never output of ordinary augmentation',ordinary_before_satellite=False,output_post_RMS=False),
        training_recipe='User fixed F+CROSS_RX SupCon; scratchE20=13L+7U; .5RD(old ordinary clean)+.5RD(same clean or original practical(raw)); oldU unchanged',
        budget_authorization='Active Ag+As goal/latest F+CROSS_RX baseline, E20 per model; two source-only firstfold rows, no old E50 weights',
        cost_claim='2 new scratch E20,8880updates,4546560Lnetwork packets; ownU/EMA/eval/source6practical measured separately;0 inherited weights',
        training_mechanism=dict(c['training_mechanism'],ordinary_L=True,L_forwards_per_sample=4,expected_L_forward_packets=2273280,
            first_pair='old ordinary clean',second_pair='practical(original IQ)' if sat else 'same old ordinary clean',
            pair_weights=[.5,.5],same_IQ_within_RDpair=True,new_U_augmentation=False,new_crossview_loss=False,
            SupCon_only_first_RDpair=True,raw_definition='Original physical IQ before common old ordinary; second As view bypasses ordinary',
            ordinary_policy='Existing F ordinary is common condition, not validated new Ag; first clean supervision weight .5 fixed in both rows'))
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed E20/FcrossRX/source/As configuration differs')

def make_args(c,device='cuda:0'):
    validate(c);a=old.make_args(old.config(old.rows()[1]),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)!=(20,13,7,.01) or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('ScratchE20 required')
    # The second supervised view is owned by our IQ adapter, not native concatenation.
    if a.use_concat_sat_channel_aug:raise ValueError('No duplicate native satellite route')
    return a

def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('Both source rows required')
    b,a=(results[k] for k in ARMS)
    for r in results.values():
        if r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] or r['optimizer_updates']!=4440:raise ValueError('Illegal/incomplete source result')
    urban=SCENES[3:];mean=lambda r,k:sum(r['practical'][s][k] for s in urban)/3
    gain=100*(mean(a,'accuracy')-mean(b,'accuracy'));low=100*(a['practical'][SCENES[-1]]['accuracy']-b['practical'][SCENES[-1]]['accuracy'])
    f1=100*(mean(a,'macro_f1')-mean(b,'macro_f1'));drop=100*(b['held_V']['accuracy']-a['held_V']['accuracy'])
    seen=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy'])
    checks=dict(urban=gain>=3.-1e-9,low=low>=5.-1e-9,urban_F1=f1>=-1e-9,clean=drop<=.5+1e-9,seen=seen<=.5+1e-9,
        complete_health=all(v['complete_health'] for v in results.values()))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,urban_gain_pp=gain,low_gain_pp=low,urban_macroF1_gain_pp=f1,
        clean_drop_pp=drop,seen_drop_pp=seen,suburban_gaps_pp={s:100*(a['held_V']['accuracy']-a['practical'][s]['accuracy']) for s in SCENES[:3]},
        source_only=True,target_access=False,automatic_extension=False,goal_completed=False)
