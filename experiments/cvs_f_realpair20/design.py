from copy import deepcopy
from pathlib import Path
from experiments.cvs_f_crossrx_practical20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,SEED,UPDATES,SCENES,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','SEED','UPDATES','SCENES','read','write'))
RUN='20261016-phase1-FcrossRX-realpair-source-RX1-e20-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_realpair_source_RX1_E20_20261016_r01'
OWNER='codex/root/FcrossRX-realpair-source-20261016'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('SAME_RX','CROSS_RX');PAIR_SEED=20261016711;SECOND_AUG_SEED=20261016712
REFERENCE=Path(PROJECT)/'runs'/prior.RUN/'C0-RX1-s2026092701/source'

def rows():return [dict(arm=a,held_RX=1,row_id=f'{a}-RX1-s{SEED}') for a in ARMS]
def parent_config():return prior.parent_config()
def config(row):
    if row not in rows():raise ValueError('Two fixed real-observation source rows only')
    c=deepcopy(prior.config(prior.rows()[0]))
    c.update(schema='FcrossRX_real_observation_pair_E20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='FcrossRX_real_observation_pair',As=False,new_Ag='UNPROVEN real sameTX observation pairing',
        baseline_reference=REFERENCE.as_posix(),baseline_reuse=False,channel=None,
        real_pair=dict(mode=row['arm'],bank_role='Only actual filtered training L5040; exact held/source roles unchanged',
            same_true_TX=True,same_day=True,different_physical_packet=True,
            RX='same source RX' if row['arm']=='SAME_RX' else 'uniform among other3 fit RX',
            packet_selection='uniform among original labeled cell members, excluding the primary physical packet',
            pair_seed=PAIR_SEED,secondary_ordinary_seed=SECOND_AUG_SEED,bank_IQ_shape=[5040,2,256],
            normalization='Original native dataset packet RMS once, no new bank or paired-view RMS',
            U_used=False,V_used=False,held_RX_used=False,synthetic_hardware=False,IQ_mixing=False,
            scope='Dataset-level true-label observation augmentation, not a digital transform of the identical physical packet or a fitted RX substitute'),
        training_recipe='F+CROSS_RX unchanged; .5RD(original ordinary primary)+.5RD(independently ordinary real sameTX/day partner); scratch20=13L+7U, original ownEMA/oldU/cos',
        training_mechanism=dict(c['training_mechanism'],first_pair='old ordinary primary',second_pair='independent ordinary real sameTX/day '+row['arm'],
            ordinary_draws_per_update=2,unique_physical_packets_per_anchor=2,matched_primary_stream=True,
            pair_weights=[.5,.5],same_IQ_within_RDpair=True,new_crossview_loss=False,SupCon_only_first_RDpair=True,
            ordinary_policy='Common old recipe; partner independent ordinary draw, identical policy/seeds across SAME_RX and CROSS_RX, no added physical distortion'),
        source_falsification=dict(cross_vs_C0_accuracy_gain_pp_min=1.,cross_vs_same_accuracy_gain_pp_min=.5,macroF1_gain_pp_min=0.,seen_V_drop_pp_max=.5,
            final_only=True,on_fail='Close crossRX-specific real-pair hypothesis; no ratio/strength/RX/seed/control rescue',
            on_pass='Separately preregister remaining source folds; no automatic target or Ag+As promotion'),
        budget_authorization='Active mixed Ag+As goal, newest F+CROSS_RX/E20 baseline; two source-only scratchE20 training rows',
        cost_claim='2newE20/8880updates; each2273280Lnetwork packets, same4 forwards as C0. Bank5040 nativeIQ reads once;568320 secondary physical exposures and ordinary draws per row. Reference C0 E20 separate, no inherited weights.')
    return c
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source-only real-pair design differs')
def make_args(c,device='cuda:0'):
    validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    return a
def reference_result():
    expected=prior.config(prior.rows()[0]);done=read(REFERENCE/'completion.json');init=read(REFERENCE/'initialization.json')
    contract=read(REFERENCE/'source_contract.json');freeze=read(REFERENCE/'source_frozen.json');result=read(REFERENCE/'result.json')
    if (done['config']!=expected or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
        or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
        or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
        or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
        or result['target_access'] or result['U_truth_read'] or not result['default_head_joint_training']
        or contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8]
        or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):
        raise ValueError('Matching current C0 reference provenance differs')
    return result
def qualify(results,reference=None):
    if set(results)!=set(ARMS):raise ValueError('Both registered rows required')
    baseline=reference_result() if reference is None else reference
    for r in [baseline,*results.values()]:
        if r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] or r['optimizer_updates']!=4440 or r['held_RX']!=1 or r['fit_RX']!=[3,4,6,8]:raise ValueError('Complete matching legal source results required')
    same,cross=results['SAME_RX'],results['CROSS_RX'];pairs={}
    for name,b in [('C0',baseline),('SAME_RX',same)]:
        pairs[name]=dict(accuracy_gain_pp=100*(cross['held_V']['accuracy']-b['held_V']['accuracy']),
            macroF1_gain_pp=100*(cross['held_V']['macro_f1']-b['held_V']['macro_f1']),
            seen_drop_pp=100*(b['seen_V']['source_val_accuracy']-cross['seen_V']['source_val_accuracy']))
    checks=dict(vs_C0=pairs['C0']['accuracy_gain_pp']>=1.-1e-9,vs_same=pairs['SAME_RX']['accuracy_gain_pp']>=.5-1e-9,
        F1=all(v['macroF1_gain_pp']>=-1e-9 for v in pairs.values()),seen=all(v['seen_drop_pp']<=.5+1e-9 for v in pairs.values()),
        complete_health=all(r['complete_health'] for r in [baseline,*results.values()]))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,paired_comparisons=pairs,
        source_only=True,target_access=False,model_seeds=1,automatic_extension=False,goal_completed=False)
