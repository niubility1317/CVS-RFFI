from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from experiments.cvs_rx_u_increment20 import design as fold
from .mechanism import ARMS
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-SIGTIME-rdrop-curvature-completion-sourcefold-s2026092702-m4-r01'
RELEASE='cvs_SIGTIME_rdrop_curvature_completion_source20_20261016_r01'
OWNER='codex/root/SIGTIME-RDrop-curvature-completion-20261016'
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
DEPENDENCY_RELEASE,DEPENDENCY_COMMIT,FOLD_RELEASE=(getattr(original,k) for k in ('DEPENDENCY_RELEASE','DEPENDENCY_COMMIT','FOLD_RELEASE'))
BASE=Path(PROJECT)/'runs'/RUN
SEED,FIT,HELD,UPDATES=2026092702,(3,4,6,8),1,4440

def native_config():
    return original.config(next(r for r in original.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Fixed four source completion rows required')
    return dict(native_config(),schema='SIGTIME_RDrop_curvature_completion_v1',run_id=RUN,**row,
        launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='cvs_rdrop_curvature_completion',
        stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],fold=fold.config()['fold'],updates=UPDATES,source_falsification=None,
        training_recipe='Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint; scratch20 13L+7oldU,true RX1 exclusion,222steps',
        training_mechanism=dict(arm=row['arm'],alpha=1.,label_smoothing=.01,
            FULL='Exact original symmetric KL,full gradient',
            SPAN='FULL + .5*mean(stop(trace_Fisher)*squared span-projected normalized joint160 difference/rank)',
            ISOTROPIC='FULL + .5*mean(stop(trace_Fisher)*squared normalized joint160 difference/160)',
            NULL='FULL + .5*mean(stop(trace_Fisher)*squared null-projected normalized joint160 difference/(160-rank))',
            geometry='A=normalize(default W,eps1e-4),C=A-rowmean(A),B=right singular vectors of C with sigma>sigma_max*1e-6;P=B B^T;rank<=5;all detached',
            Fisher_trace='t_i=30^2*(sum_k pbar_ik*norm(A_k)^2 - norm(pbar_i A)^2);pbar=(softmax(logits1)+softmax(logits2))/2,detached',
            trace_budget='Each added metric has total trace t_i;not equal realized losses,gradient norms,AdamW steps or strengths across independent trajectories',
            objective='Original paired smoothed CE + original full symmetric KL + chosen complement quadratic + original U loss',
            default_head_joint=True,KL_default_W_joint=True,auxiliary_default_W_direct_gradient=False,
            original_hidden_U_recipe_unchanged=True,augmentation_redraws=0,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=568320,
            interpretation='Null refers to centered cosine classifier in normalized feature coordinates,not a raw-feature or encoder/AdamW invariance guarantee;no cross-RX invariance or newTX proof'),
        source_selection=dict(primary='NULL',controls=['FULL','SPAN','ISOTROPIC'],held_RX=HELD,
            readout='Fixed final20 default joint head,1260 held source L',final_L_CE_max=.5,minimum_aux_positive_call_fraction=.9,
            errors_strictly_fewer=True,mean_CE_no_higher=True,
            on_pass='Separately preregister fixed RX3-excluded source confirmation,same4arms/seed/E20;no main target before confirmation',
            on_fail='Close fixed curvature completion recipe;no alpha,rank,layer,seed,temperature,epoch rescue',target_access=False),
        baseline_reference='All4 own scratch from same original construction;no trained ancestor. FULL native plus3same-trace direction controls',
        budget_authorization='Only SIG_TIME+RDrop;new complementary feature-signal hypothesis;fixed next source seed02,4scratch20',
        cost_claim='4scratch20/80model-epoch/17760updates;each568320extraL/396368hiddenU observations/1260heldL/144unusedL anchors;0target',
        goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source completion matrix differs')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=original.make_args(native_config(),device)
    a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
    a.source_fold_held_rx,a.source_fold_training_receivers=HELD,','.join(map(str,FIT))
    if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Native source scratch recipe required')
    return a

source_validation=fold.source_validation

def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('All4 complete source results required')
    for r in results.values():
        h=r['exports']['FINAL']
        if h['count']!=1260 or not math.isfinite(h['mean_CE']):raise ValueError('Complete finite source readout required')
    main=results['NULL']['exports']['FINAL'];healthy=all(r['source_health'] for r in results.values())
    active=results['NULL']['evidence_audit']['aux_positive_calls']>=.9*UPDATES
    contrasts={a:dict(error_difference=main['errors']-results[a]['exports']['FINAL']['errors'],CE_difference=main['mean_CE']-results[a]['exports']['FINAL']['mean_CE']) for a in ARMS[:-1]}
    return dict(qualified=healthy and active and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrasts.values()),all_source_healthy=healthy,signal_active=active,contrasts=contrasts,target_access=False,goal_completed=False)

def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True)
