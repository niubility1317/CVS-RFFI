"""One genuine RX1-held-out scratch20 SSL fold; four optimizer diagnostics."""
from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-RX1-U-increment-sourcefold-s2026092701-r01'
RELEASE='cvs_SIGTIME_RX1_U_increment_sourcefold_20261015_r01'
DEPENDENCY_RELEASE=PROJECT+'/releases/'+parent.RELEASE
SEED=2026092701;HELD=1;RX=(1,3,4,6,8);FIT=(3,4,6,8)
EPOCHS=20;STEPS=222;UPDATES=4440;PROBES=(2896,3344,3792,4432)
MODES=('RAW','L_ONLY_F','SHRINK','CONE')
OUTPUT=PROJECT+'/runs/'+RUN+'/source'
GATE=dict(minimum_qualified_probes=2,final_training_L_CE_max=.5,U_loss_min=1e-8,raw_U_F_to_full_F_step_min=.01,
    retained_U_F_norm_min=.1,CONE_RAW_cosine_max=.95,
    ACTUAL_and_PRE_W_CE_improvement_min=1e-5,errors_must_not_increase=True,
    actual_stored_norm_relative_tolerance=1e-3,rounded_cone_violation_relative_max=1e-3,
    identity_controls=['RAW','SHRINK','L_ONLY_F'],
    promotion='Only a full controlled source-fold training matrix, never direct target/main qualification')

def parent_config():return parent.config(next(r for r in parent.rows() if r['arm']=='BASE_SSL' and r['model_seed']==SEED))
def config():
    c=parent_config()
    return dict(c,schema='phase1_RX_U_optimizer_increment_sourcefold_v1',run_id=RUN,row_id='RAW_SSL-RX1-s2026092701',
        arm='RAW_SSL_SOURCE_FOLD',method='cvs_source_RX_U_optimizer_increment',stage='source_RX_fold_diagnostic',
        test_views=[],readouts=['ACTUAL','PRE_W'],
        budget_authorization='Active Phase1 source-falsification research: one genuine source-RX1 held-out scratch20 fold and four temporary update probes; no target or full candidate trajectory',
        baseline_reference='Own unchanged RAW SSL fold trajectory; not reused as a full-five-RX performance control',
        launch_owner='codex/root/RX1-U-increment-20261015',output_root=OUTPUT,
        training_mechanism=dict(actual='Unchanged raw L+U student AdamW; diagnostic never corrects it',
            probes=list(PROBES),modes=list(MODES),feature_scope='71 encoder tensors including three feature gains; excludes default class W',
            L_counterfactual='Separate real backward of current identity_loss, same pre-state optimizer, clip5 once',
            RAW_reference='Actual already-clipped gradients copied into temporary original AdamW; no second clip',
            source_risk='Current default-head CE of fixed144 training-L anchors, one normalized gradient per training RX',
            projection='min ||p-u||^2 with g_RX dot p<=0; u=RAW_after_F-L_only_after_F',
            norm_control='SHRINK follows raw U displacement and matches rounded CONE FP32 displacement norm',
            evaluation='1260 original visible L packets of RX1, never trained in this fold',
            head_fit_updates=0,actual_candidate_corrections=0,temporary_AdamW_steps=8,
            additional_source_L_IQ_reads=1404,diagnostic_packet_forwards_including_initial_smoke=21998),
        source_falsification=dict(GATE),fold=dict(held_RX=HELD,training_RX=list(FIT),
            carrier_source_roles='Original unchanged VALIDATED_ONCE L6300/U56700/V27000; metadata verified before masking',
            effective_roles=dict(L_s=5040,U_s=45360,V=21600),held_L_evaluation=1260,
            all_model_teacher_training_calibration_and_feature_cache_inputs_exclude_RX1=True,
            raw_source_carrier_and_static_domain_ID_map_retained=True,trained_checkpoint_reuse_allowed_for_all5RX=False,
            steps_per_epoch=222,U_observations_per_pseudo_epoch=56624),
        training_recipe='SIG_TIME+CENTER25K/default cosine; scratch20 LS0; 13L+7U original oldgate/ownEMA; four-RX filtered loaders with original222 step budget',
        cost_claim='One source-only20-epoch baseline/4440 actual steps, no target; 8 temporary AdamW updates/4risk probes/21998 diagnostic forwards; no full candidate training yet',
        goal=dict(clean_three_seed_mean=.84,goal_completed=False,source_diagnostic_only=True))
def validate(c):
    if c!=config():raise ValueError('Fixed RX1 U-increment source fold differs')
    return c
def make_args(c,device='cuda:0'):
    validate(c);a=parent.make_args(parent_config(),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=OUTPUT
    a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
    return a

def gate(records):
    passed=[]
    for r in records:
        v=r['geometry'];pairs=r['paired'];good=(r['U_loss']>GATE['U_loss_min']
            and v['raw_U_F_to_full_F_step']>=GATE['raw_U_F_to_full_F_step_min']
            and v['retained_U_F_norm_ratio']>=GATE['retained_U_F_norm_min']
            and v['CONE_RAW_cosine'] is not None and v['CONE_RAW_cosine']<=GATE['CONE_RAW_cosine_max']
            and v['stored_norm_control']['norm_matched']
            and v['rounded_cone_violation_relative']<=GATE['rounded_cone_violation_relative_max'])
        for control in GATE['identity_controls']:
            for readout in ('ACTUAL','PRE_W'):
                good=good and pairs[control][readout+'_CE_difference']<=-GATE['ACTUAL_and_PRE_W_CE_improvement_min']
                good=good and pairs[control][readout+'_error_difference']<=0
        if good:passed.append(r['step'])
    return dict(qualified_steps=passed,qualified_count=len(passed),total_probes=len(records),
        full_source_training_qualification=len(records)==4 and len(passed)>=GATE['minimum_qualified_probes'],main_run_qualification=False)

def source_validation(model,loader,device):
    import torch
    model.eval();hits=0;total=0;groups={}
    with torch.no_grad():
        for x,y,domain,meta in loader:
            correct=(model(x.to(device)).argmax(1).cpu()==y)
            for rx in meta['rx_i'].unique().tolist():
                mask=meta['rx_i']==rx;v=groups.setdefault(str(rx),[0,0]);v[0]+=int(correct[mask].sum());v[1]+=int(mask.sum())
            hits+=int(correct.sum());total+=len(y)
    if total!=21600 or set(groups)!=set(map(str,FIT)):raise ValueError('Source health V must exclude held RX1')
    rates={key:value[0]/value[1] for key,value in groups.items()}
    return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()),held_RX_excluded=True)
