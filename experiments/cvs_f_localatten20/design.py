from copy import deepcopy
from pathlib import Path
from experiments.cvs_f_crossrx_practical20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,SEED,UPDATES,SCENES,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','SEED','UPDATES','SCENES','read','write'))
RUN='20261016-phase1-FcrossRX-localatten-source-RX1-e20-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_localatten_source_RX1_E20_20261016_r01'
OWNER='codex/root/FcrossRX-localatten-source-20261016'
BASE=Path(PROJECT)/'runs'/RUN
LOCAL_SEED=20261017111
REFERENCE=Path(PROJECT)/'runs'/prior.RUN/'C0-RX1-s2026092701/source'

def rows():return [dict(arm='LA16',held_RX=1,row_id=f'LA16-RX1-s{SEED}')]
def parent_config():return prior.parent_config()
def config(row):
    if row not in rows():raise ValueError('One fixed local-attenuation source row only')
    c=deepcopy(prior.config(prior.rows()[0]))
    c.update(schema='FcrossRX_localatten_E20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='FcrossRX_bounded_local_observation',As=False,
        new_Ag='UNPROVEN bounded local observation attenuation; requires actual heldRX clean benefit',
        baseline_reference=REFERENCE.as_posix(),baseline_reuse=False,channel=None,
        local_attenuation=dict(seed=LOCAL_SEED,width=16,start_min=0,start_max=240,selection='Uniform complete window,private stream keyed by seed and optimizerstep; independent of labels,RX,loss and quality',
            window='sin^2(pi*(k+.5)/16),k=0..15; zero elsewhere',depth_max=.5,relative_L2_max=.1,
            depth='min(.5,.1*(1-2e-6)*norm(x)/norm(window*x)); when weighted energy0 depth.5,zero packet retained',
            IQ='Same positive real mask on I/Q; unchanged time indices and all240 outside samples; no new RMS/AGC',
            refresh='Each epoch/step/view; one generated masked IQ reused within its RD pair',
            excluded=['TX hardware inversion','CFO','new noise','new multipath','extra PA/DAC','crop','circular join','padding','signal permutation'],
            scope='Digital partial-observation regularization,not a physical LEO model or fitted receiver replacement; label preservation is unproven'),
        training_recipe='F+CROSS_RX unchanged;.5RD(old ordinary IQ)+.5RD(localatten(same generated ordinary IQ));scratch20=13L+7U,original ownEMA/oldU/cos',
        training_mechanism=dict(c['training_mechanism'],first_pair='old ordinary clean',second_pair='bounded local attenuation of same actual ordinary IQ',
            pair_weights=[.5,.5],actual_ordinary_draws_per_L_step=1,unique_physical_packets_per_anchor=1,
            same_IQ_within_RDpair=True,new_crossview_loss=False,SupCon_only_first_RDpair=True,
            ordinary_policy='Old ordinary including native endpoint Tukey retained; new short smooth window has random location and per-packet norm bound'),
        source_falsification=dict(held_clean_accuracy_gain_pp_min=1.,macroF1_gain_pp_min=0.,seen_V_drop_pp_max=.5,each_practical_drop_pp_max=.5,
            final_only=True,on_fail='Close fixed localatten hypothesis;no width/depth/L2bound/probability/ratio/RX/seed rescue',
            on_pass='Only separately preregister source confirmation;not automatic target or Ag+As promotion'),
        budget_authorization='Active Ag+As goal,latest F+CROSS_RX/E20;one independent source-only scratchE20;no extra head/module/loss',
        cost_claim='One newE20/4440updates/2273280Lnetwork packets,expected3579018 totalNN from actual counters;same4forwards/oneordinary and primaryL-U as C0.568320 local windows. ReferenceC0 E20 separate,no inherited state.')
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source-only localatten design differs')

def make_args(c,device='cuda:0'):
    validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

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
def qualify(result,reference=None):
    baseline=reference_result() if reference is None else reference
    for r in [baseline,result]:
        if r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] or r['optimizer_updates']!=4440 or r['held_RX']!=1 or r['fit_RX']!=[3,4,6,8] or set(r['practical'])!=set(SCENES):raise ValueError('Complete matching legal source results required')
    delta=100*(result['held_V']['accuracy']-baseline['held_V']['accuracy'])
    f1=100*(result['held_V']['macro_f1']-baseline['held_V']['macro_f1'])
    seen=100*(baseline['seen_V']['source_val_accuracy']-result['seen_V']['source_val_accuracy'])
    practical={s:100*(result['practical'][s]['accuracy']-baseline['practical'][s]['accuracy']) for s in SCENES}
    checks=dict(clean=delta>=1.-1e-9,F1=f1>=-1e-9,seen=seen<=.5+1e-9,practical=all(v>=-.5-1e-9 for v in practical.values()),complete_health=result['complete_health'] and baseline['complete_health'])
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,clean_gain_pp=delta,macroF1_gain_pp=f1,seen_drop_pp=seen,practical_gain_pp=practical,
        source_only=True,target_access=False,model_seeds=1,automatic_extension=False,goal_completed=False)
