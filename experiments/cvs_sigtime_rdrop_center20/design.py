"""Fixed RX3-excluded source falsification; native KL always retained."""
from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from .mechanism import ARMS
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(original,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-SIGTIME-rdrop-batch-center-RX3-sourcefold-s2026092702-m4-r01'
RELEASE='cvs_SIGTIME_rdrop_batch_center_RX3_source20_20261016_r01'
OWNER='codex/root/SIGTIME-RDrop-batch-center-20261016'
MODEL_RELEASE=PROJECT+'/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
DEPENDENCY_RELEASE,DEPENDENCY_COMMIT,FOLD_RELEASE=(getattr(original,k) for k in ('DEPENDENCY_RELEASE','DEPENDENCY_COMMIT','FOLD_RELEASE'))
BASE=Path(PROJECT)/'runs'/RUN
SEED,FIT,HELD,RX,STEPS,UPDATES=2026092702,(1,4,6,8),3,(1,3,4,6,8),222,4440

def native_config():
    return original.config(next(r for r in original.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))

def rows():return [dict(row_id=a+'-RX3-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]

def config(row):
    if row not in rows():raise ValueError('Fixed four RX3 source rows required')
    return dict(native_config(),schema='SIGTIME_RDrop_batch_center_v1',run_id=RUN,**row,
        launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='cvs_rdrop_batch_center',
        stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],updates=UPDATES,source_falsification=None,
        fold=dict(held_RX=HELD,training_RX=list(FIT),effective_roles=dict(L_s=5040,U_s=45360,V=21600),held_L_evaluation=1260,
            carrier_source_roles='Original VALIDATED_ONCE L6300/U56700/V27000;metadata verified before masking',
            all_model_teacher_training_calibration_and_feature_cache_inputs_exclude_RX3=True,
            raw_source_carrier_and_static_domain_ID_map_retained=True,trained_checkpoint_reuse_allowed_for_all5RX=False,
            steps_per_epoch=222,U_observations_per_pseudo_epoch=56624),
        training_recipe='Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint;scratch20 13L+7oldU,true RX3 exclusion,222steps',
        training_mechanism=dict(arm=row['arm'],alpha=1.,label_smoothing=.01,
            FULL='Exact original symmetric KL,full gradient',
            POINT='FULL + .5*mean(stop(T)*squared(normalize(z1)-normalize(z2))/160);old ISOTROPIC positive control',
            CENTER='FULL + .5*mean(stop(T)*squared(normalize(z1-mu)-normalize(z2-mu))/160);mu=mean((z1+z2)/2),differentiable',
            NORM_MATCH='FULL + stop(norm(g_CENTER)/norm(g_POINT))*POINT_aux;gradients only at current actual joint160 pair',
            Fisher_trace='T=900*(sum_k pbar_k*norm(normalize(W)_k)^2 - norm(pbar@normalize(W))^2);pbar=mean softmax(logits1,logits2),detached,double calculation',
            normalization_eps=1e-4,mean_stop_gradient=False,norm_ratio_stop_gradient=True,norm_match_space='Concatenated raw joint160 features z1,z2;not model parameters/clip/AdamW',
            norm_match_tolerance=2e-5,common_translation_gradient_ratio_tolerance=1e-5,
            objective='Original paired smoothed CE + original full symmetric KL + selected feature auxiliary + original U',
            default_head_joint=True,KL_default_W_joint=True,auxiliary_default_W_direct_gradient=False,
            original_hidden_U_recipe_unchanged=True,augmentation_redraws=0,extra_parameters=0,extra_optimizer_steps=0,extra_L_identity_forwards=568320,
            boundary_autograd_traversals_per_step=0 if row['arm']=='FULL' else 2,
            interpretation='Only CENTER auxiliary with fixed detached T is invariant to shared batch translation;not full objective,encoder-center,optimizer or RX invariance guarantee'),
        source_selection=dict(primary='CENTER',controls=['FULL','POINT','NORM_MATCH'],held_RX=HELD,
            readout='Fixed final20 default joint head,1260 held source L',final_L_CE_max=.5,minimum_aux_positive_call_fraction=.9,
            errors_strictly_fewer=True,mean_CE_no_higher=True,
            on_pass='Separately preregister fixed RX1-excluded same4arms/seed/E20 confirmation;no target before confirmation',
            on_fail='Close fixed CENTER recipe;no alpha,seed,temperature,epoch,layer rescue',target_access=False),
        positive_route_confirmation=dict(candidate='POINT',control='FULL',held_RX=HELD,
            criterion='Strictly fewer held errors and mean CE no higher;all source healthy,POINT positive auxiliary>=90%calls',
            purpose='Predeclared separate confirmation of prior RX1 whole-feature positive evidence;does not replace CENTER primary qualification',
            on_pass='Retain POINT as independently confirmed across two development source folds;future separate preregistration required,no target in this run',
            on_fail='Retain prior RX1 evidence with failed RX3 confirmation;no rescue or target escalation'),
        baseline_reference='Four own scratch trajectories,same original construction;no trained ancestor',
        budget_authorization='Only SIG_TIME+RDrop;centered direction hypothesis and explicit separate old whole-feature positive-route confirmation;fixed source seed02,4scratch20',
        cost_claim='4scratch20/80model-epoch/17760updates;each568320extraL/396368hiddenU observations/1260heldL/144unusedL anchors;0target',
        goal=dict(clean_three_seed_mean=.84,source_only=True,target_access=False,global_novelty_claim=False,goal_completed=False))

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed RX3 source centering matrix differs')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=original.make_args(native_config(),device)
    a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
    a.source_fold_held_rx,a.source_fold_training_receivers=HELD,','.join(map(str,FIT))
    if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Native source scratch recipe required')
    return a

def source_validation(model,loader,device):
    import torch
    model.eval();hits=total=0;groups={}
    with torch.no_grad():
        for x,y,domain,meta in loader:
            correct=(model(x.to(device)).argmax(1).cpu()==y)
            for rx in meta['rx_i'].unique().tolist():
                mask=meta['rx_i']==rx;v=groups.setdefault(str(rx),[0,0]);v[0]+=int(correct[mask].sum());v[1]+=int(mask.sum())
            hits+=int(correct.sum());total+=len(y)
    if total!=21600 or set(groups)!=set(map(str,FIT)):raise ValueError('Source health V must exclude held RX3')
    rates={k:v[0]/v[1] for k,v in groups.items()}
    return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()),held_RX_excluded=True)

def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('All four complete source results required')
    for r in results.values():
        h=r['exports']['FINAL']
        if h['count']!=1260 or not math.isfinite(h['mean_CE']):raise ValueError('Complete finite source readout required')
    healthy=all(r['source_health'] for r in results.values())
    def contrast(candidate,control):
        a,b=(results[k]['exports']['FINAL'] for k in (candidate,control))
        return dict(error_difference=a['errors']-b['errors'],CE_difference=a['mean_CE']-b['mean_CE'])
    contrasts={a:contrast('CENTER',a) for a in ('FULL','POINT','NORM_MATCH')}
    active=results['CENTER']['evidence_audit']['aux_positive_calls']>=.9*UPDATES
    point=contrast('POINT','FULL');point_active=results['POINT']['evidence_audit']['aux_positive_calls']>=.9*UPDATES
    wins=lambda v:v['error_difference']<0 and v['CE_difference']<=0
    return dict(qualified=healthy and active and all(wins(v) for v in contrasts.values()),all_source_healthy=healthy,signal_active=active,contrasts=contrasts,
        separate_POINT_confirmation=dict(qualified=healthy and point_active and wins(point),contrast=point,signal_active=point_active,not_primary_substitution=True),
        target_access=False,goal_completed=False)

def request():return dict(run_id=RUN,model_release=MODEL_RELEASE,model_code_commit=MODEL_COMMIT,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],source_only=True)
