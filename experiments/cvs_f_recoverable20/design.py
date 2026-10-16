from copy import deepcopy
from pathlib import Path
from experiments.cvs_f_crossrx_folds20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,SEED,UPDATES,SCENES,TRAIN_SCENES,CHANNEL_SEEDS,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','SEED','UPDATES','SCENES','TRAIN_SCENES','CHANNEL_SEEDS','read','write'))
RUN='20261016-phase1-FcrossRX-recoverable-source-RX6-e20-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_recoverable_source_RX6_E20_20261016_r01'
OWNER='codex/root/FcrossRX-recoverable-source-20261016'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE=Path(PROJECT)/'runs'/prior.RUN
COURSE_SEED=20261016651
def rows():return [dict(arm='C2',held_RX=6,row_id=f'C2-RX6-s{SEED}')]
def parent_config():return prior.parent_config()
def original(arm):return prior.config(next(r for r in prior.rows() if r['arm']==arm and r['held_RX']==6))
def config(row):
    if row not in rows():raise ValueError('One fixed source RX6 curriculum row')
    c=deepcopy(original('C1'))
    c.update(schema='FcrossRX_recoverable_state_E20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='FcrossRX_recoverable_state_curriculum',
        baseline_reference='Reuse matching heldRX6 C0/C1 FINAL20 results and metadata only; zero inherited weights',
        new_Ag=False,As=True,curriculum=dict(seed=COURSE_SEED,eta_initial=0.,eta_max=.5,warm_epochs=5,eta_increment=.1,
            mixture='(1-eta)*original_nominal + eta*finite_recoverable_proposal_law; nominal lower bound .5',
            hard_scenes=['practical_mid_urban','practical_low_urban'],locks=['locked','degraded'],signal_to_added_noise_db=[3.,15.],
            signal_to_added_noise_is_total_SNR=False,extra_proposals_max=4,
            finite_law='First generate exact C1 nominal draw. Hard branch accepts it if eligible; else up to4 independent complete urban draws; on failure return original nominal draw.',
            progress='Previous epoch fitL trueTX x physicalRX x day primary ordinary accuracy/margin/CE, plus existing native gradient telemetry; no heldV/target or U truth',
            progress_rule='AfterE5 increase .1 only if accuracy>=.7,p10cellaccuracy>=.6,p10cellmargin/30>=.02,CE<=1,mean gradient<=5 and<=2x previous3median; meanCE<=1.05x previous3median,p10margin>=previous3median-.01. Else hold.',
            never='No CE gate, parameter weakening, partial impairment removal, target feedback or auxiliary classifier'),
        training_mechanism=dict(c['training_mechanism'],second_pair='full practical(raw) sampled by bounded recoverable-state curriculum; all selected labeled views supervised'),
        source_falsification=dict(clean_drop_vs_C0_pp_max=.5,seen_drop_vs_C0_pp_max=.5,urban_gain_vs_C1_pp_min=0.,low_gain_vs_C1_pp_min=0.,urban_F1_gain_vs_C1_pp_min=0.,
            actual_curriculum_activation=True,complete_health=True,on_fail='Close fixed curriculum; no eta/range/warm/proposals/ratio/RX/seed/threshold rescue',
            on_pass='Only separately register other source-fold confirmation, no automatic target/seed/Ag promotion'),
        budget_authorization='Active Ag+As goal/latest F+CROSS_RX baseline; one new scratchE20 sourceRX6 falsification, original U fixed',
        cost_claim='One newE20/4440updates,4Lforwards/oneordinary draw,3579018 expectednetworkpackets; proposal draws/time measured separately. Reused C0/C1 eachE20,no parentweights.')
    c.pop('confirmation',None);return c
def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed recoverable curriculum/source/architecture/budget differs')
def make_args(c,device='cuda:0'):
    validate(c);a=prior.make_args(original('C1'),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def reference_results():
    result={}
    for arm in ('C0','C1'):
        p=REFERENCE/f'{arm}-RX6-s{SEED}'/'source';done=read(p/'completion.json');init=read(p/'initialization.json');freeze=read(p/'source_frozen.json');r=read(p/'result.json');contract=read(p/'source_contract.json')
        if (done['config']!=original(arm) or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] or not r['default_head_joint_training'] or r['optimizer_updates']!=4440
            or contract['held_RX']!=6 or contract['source_receivers']!=[1,3,4,8] or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):raise ValueError('Matching legal sourceRX6 references required, no weights')
        result[arm]=r
    return result
def qualify(result,reference=None):
    refs=reference_results() if reference is None else reference;b,a=refs['C0'],refs['C1'];c=result
    for r in (b,a,c):
        if r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] or r['optimizer_updates']!=4440 or r['held_RX']!=6 or r['fit_RX']!=[1,3,4,8]:raise ValueError('Complete legal sourceRX6 rows required')
    mean=lambda r,k:sum(r['practical'][s][k] for s in SCENES[3:])/3
    values=dict(clean_drop_vs_C0_pp=100*(b['held_V']['accuracy']-c['held_V']['accuracy']),seen_drop_vs_C0_pp=100*(b['seen_V']['source_val_accuracy']-c['seen_V']['source_val_accuracy']),
        urban_gain_vs_C1_pp=100*(mean(c,'accuracy')-mean(a,'accuracy')),low_gain_vs_C1_pp=100*(c['practical'][SCENES[-1]]['accuracy']-a['practical'][SCENES[-1]]['accuracy']),
        urban_F1_gain_vs_C1_pp=100*(mean(c,'macro_f1')-mean(a,'macro_f1')))
    checks=dict(clean=values['clean_drop_vs_C0_pp']<=.5+1e-9,seen=values['seen_drop_vs_C0_pp']<=.5+1e-9,urban=values['urban_gain_vs_C1_pp']>=-1e-9,
        low=values['low_gain_vs_C1_pp']>=-1e-9,urban_F1=values['urban_F1_gain_vs_C1_pp']>=-1e-9,activation=c['curriculum']['max_eta']>0 and c['training_practical']['hard_requested']>0,
        complete_health=all(r['complete_health'] for r in (b,a,c)))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,**values,source_only=True,target_access=False,model_seed_count=1,source_fold_count=1,automatic_extension=False,goal_completed=False)
