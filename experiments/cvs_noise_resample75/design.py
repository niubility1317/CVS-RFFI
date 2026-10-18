"""A fixed six-L CE comparison, not another noise-strength or teacher search."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_full_channel_pair68 import design as old

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=old.PROJECT,old.SOURCE
RUN='20261019-phase1-FcrossRX-noiseResample6-e30-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_noise_resample6_E30_20261019_r01'
OWNER='codex/root/noise-resample75'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE_RUN,SAME_RUN,FULL_PAIR_RUN=old.REFERENCE_RUN,old.SAME_RUN,old.RUN
SEED,STEPS,UPDATES=2026092701,222,2220
RX=FIT=(1,3,4,6,8)
ARMS=('C01','N01')
LOCAL_NN,SMOKE_NN=0,1540
FULL_FP32_POLICY,CLASSES,SCENES,VIEWS=old.FULL_FP32_POLICY,old.CLASSES,old.SCENES,old.VIEWS
FORMAL_VIEWS,TRUTH,PANEL,SOURCE_CACHE=old.FORMAL_VIEWS,old.TRUTH,old.PANEL,old.SOURCE_CACHE
TRAIN_SCENES=old.TRAIN_SCENES
read,write=old.read,old.write

def rows():
    return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]

def reference_config(arm='AS_CE'):
    if arm=='INDEP_CE6':return old.config(old.rows()[0])
    return old.reference_config(arm)

def old_config(arm=None):
    if arm is not None and arm not in ARMS:raise ValueError('Only the two fixed noise-pair rows')
    return reference_config('AS_CE')

def config(row,segment=30):
    if row not in rows() or segment!=30:raise ValueError('Only common AS_CE20 plus ten epochs')
    inherited=old_config();c=deepcopy(inherited)
    method='SAME_NOISE_CE6' if row['arm']=='C01' else 'INDEP_NOISE_CE6'
    c.update(schema='FcrossRX_noise_resample75_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=str(BASE/row['row_id']/'source'),parent_source=inherited['parent_source'],
        checkpoint_sources=[inherited['parent_source']+'/final_ssdg.pth'],
        method=method,stage='FcrossRX_noise_resample75',satellite_mode='AS_CE',
        model_initialization='Strict same SAT18 AS_CE scratch20 student/ownEMA; fresh AdamW/cos10/CENTER_UNITS',
        training_recipe='Original sixL768/CE1.25/ordinary-raw KL/ORD128 SupCon/nativeU;two As64 CE slots,only noise reuse differs',
        baseline_reference='New same-parent SAME_NOISE_CE6 control; original P43 NoAs30/As30 and P68 INDEP_CE6 reused descriptively',
        baseline_reuse=False,
        noise_pair=dict(method=method,source='Same raw L IQ,not first received As IQ',
            parent20_As_KL_off=True,first_namespace='Original FcrossRX/train/e{epoch}/step{step}/L_S',
            second_noise_suffix='/view1-noise-independent',channel_seed=20261016421,receiver_seed=20261016423,
            same_physical_seed=True,same_geometry_H_state_phase_IQ_EQ=True,
            second_noise='same received IQ' if row['arm']=='C01' else 'Replace only instance latent.rngs[noise] with an independent named Generator',
            same_hardware_object=True,session='RX/day/eq,never TX',
            AGC='Per-view original native recomputation; not frozen',
            noise_energy='Native random energy; no exact-RMS calibration or output white-noise addition',
            As_KL=False,original_dropout=True,ordinary_raw_KL=True,two_separate_CE_mean=True,
            total_As_CE_coefficient=.25,per_pass_As_CE_coefficient=.125,
            L_NN_per_step=768,As_forward_NN_per_step=128,
            first_generations_per_step=64,second_generations_per_step=0 if row['arm']=='C01' else 64,
            unique_nonnoise_latents_per_step=64,extra_NN=0,extra_parameters=0,extra_losses=0,
            all_labeled_As_CE=True,quality_rejection=False,scene_prior_changed=False,
            U_changed=False,formal_test_changed=False),
        history_boundary='Old heldRX1 ten-L INDEP/ANTI supervised two-noise views but had no SAME_NOISE control; ANTI vs INDEP FAIL is not this isolated comparison',
        parent_history_difference='Both new rows inherit SAT18 AS_CE20 with As KL already off; not original P43 As20, no attribution of first20 differences to new noise resampling',
        budget_authorization='Two commonAS_CE20+10 tails,total30/6660 each; no new seed; later totalE50 only after source qualification with matched controls',
        cost_claim='Each new10 has 2220updates/1704960LNN;C142080,N284160nativeAs calls;NN unchanged,physical CPU/wall recorded',
        goal=dict(fullV_clean_N_vs_NoAs_min_pp=-.1,
            pooled_midlow_N_vs_max_C_originalAs_min_pp=.5,
            source_FAIL_still_all7=True,E50_only_after_source_PASS=True,
            no_noise_strength_or_seed_rescue=True,target_feedback=False))
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed same-parent/native-noise-only/sixL contract differs')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=old.old.make_args(old_config(),device)
    a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
    a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth'
    if a.teacher_ckpt or a.pseudo_temporal_mode!='batch_neighbor' or not a.pseudo_temporal_gate:
        raise ValueError('Original ownEMA/native U rules required')
    return a
