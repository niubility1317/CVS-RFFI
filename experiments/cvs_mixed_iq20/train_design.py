"""Single-seed fixed factorial; no target-dependent row or budget selection."""
from pathlib import Path
from . import design as proposal
from . import probe_design as source_probe
from experiments.cvs_sigtime_rdrop_legacy import design as legacy

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY=(getattr(legacy,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY'))
RUN='20261015-phase1-sigtime-rdrop-mixedIQ-e20-manysig-s2026092701-m5-r01'
RELEASE='cvs_sigtime_rdrop_mixedIQ_e20_20261015_r01'
OWNER='codex/root/mixed-IQ-E20-20261015'
BASE=Path(PROJECT)/'runs'/RUN
FORMAL_VIEWS=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
SCENES=proposal.SCENES;VIEWS=('clean',*SCENES);SEEDS=(2026092701,);ARMS=proposal.ARMS
EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,UPDATES=20,13,7,4440
read,write=source_probe.read,source_probe.write


def rows():
    return [dict(arm=a,model_seed=2026092701,row_id=a+'-s2026092701') for a in ARMS]


def parent_recipe(seed):
    return legacy.config(next(r for r in legacy.rows() if r['row_id']==f'E20_RDROP-s{seed}'))


def config(row):
    if row not in rows():raise ValueError('Unregistered fixed E20 mixed IQ row')
    p=proposal.plan();r=next(x for x in p['rows'] if x['arm']==row['arm'])
    return dict(parent_recipe(row['model_seed']),schema='SIGTIME_RDrop_mixedIQ_factorial_v1',
        run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
        baseline_reuse=False,initialization='scratch',parent_checkpoint=None,checkpoint_sources=[],
        parent_training_epochs=0,incremental_epochs=20,total_epochs=20,epochs=20,
        source_roles=dict(L_s=6300,U_s=56700,V=27000),source_RX=[1,3,4,6,8],source_days=[1,2,3],
        sample_rate_hz=25_000_000,IQ_length=256,equalized=1,
        label_epochs=13,pseudo_epochs=7,updates=4440,baseline_reference='fresh R_LEGACY, same full source contract',
        training_recipe='SIG_TIME RAW_HZ/defaultcos/NoExplicitCFO; scratch20=13L+7U oldgate/ownEMA; '+row['arm'],
        test_views=list(VIEWS),readouts=['NATIVE_HEAD'],goal=dict(primary='B11',**p['exploration_targets'],target_feedback=False),
        cost_claim='five scratchE20; matched factorial6 L forwards, legacy2; no parent weights',
        training_mechanism=dict(A_g=r['A_g'],A_s=r['A_s'],legacy_L_augmentation=r['legacy_L_augmentation'],
            L_forwards_per_sample=2 if row['arm']=='R_LEGACY' else 6,
            expected_L_forward_packets=r['labeled_forward_packets'],physical_L_exposures=568320,
            objective=r['objective'],slot_weights=[1.] if row['arm']=='R_LEGACY' else [.25,.25,.5],
            RDrop_alpha=1.,label_smoothing=.01,same_generated_IQ_within_each_RDrop_pair=True,
            compose_A_s_A_g=False,extra_parameters=0,new_U_augmentation=False,
            generalization_seed=20261015031,practical_seed=20261015032,receiver_seed=2026092701,
            scene_probabilities=p['training_scene_probabilities'],response_radius=.06,
            source_panel_seed=20261015033,source_panel_role='V diagnostic subset720, not a new role',
            actual_U_rule='native oldgate+ownEMA unchanged; TX hidden'),
        source_falsification=p['source_falsification'],checkpoint_selection='final_only',
        separate_head_training=False,source_safety_run=source_probe.RUN)


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed E20 mixed IQ configuration differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c);a=legacy.make_args(parent_recipe(c['model_seed']),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    a.use_aug=c['training_mechanism']['legacy_L_augmentation']
    if a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled or a.label_smoothing!=.01 or a.baseline_ckpt or a.teacher_ckpt:
        raise ValueError('Native old-U/scratch/default-head recipe required')
    return a


def source_safety():
    complete=read(source_probe.BASE/'complete.json');result=read(source_probe.BASE/'results.json')
    if (complete.get('status')!='ANALYZED' or complete.get('forward_packets')!=135002
            or result.get('source_gate',{}).get('status')!='PASS' or result.get('query_read')
            or not result.get('state_unchanged') or result.get('optimizer_updates')
            or result.get('config')!=source_probe.config()):
        raise ValueError('Declared zero-update source safety evidence must have passed unchanged')
    return result
