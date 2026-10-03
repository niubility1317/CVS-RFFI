"""Preregister two cross-path relation architectures, four own-scratch seeds."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_crosspath_relation_identity.model import build,VARIANTS,readout_contract
from experiments.cvs_crosspath_relation_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN,ANCHOR_CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-crosspath-relation-identity-manysig-m8-r01'
RELEASE='cvs_crosspath_relation_identity_20261003_r01'
CONFIRM_RUN='20261003-phase1-cvs-crosspath-relation-clean-manysig-m48-r01'
OLD_CLEAN_RUN='20261003-phase1-cvs-response-fusion-clean-manysig-m44-r01'

def main():
    configs=ROOT/'experiments/cvs_crosspath_relation_identity/configs'
    if configs.exists():raise FileExistsError('Preserve existing cross-path registration/configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-crosspath-relation-identity-ce',
        display_name='CVS跨路径关系：整体RMS Gram与逐投影通道RMS coherence×四seed',
        description='保留浅卷积残差骨干及两路原320维skip；仅在behavior skip加入低秩复跨路径关系，比较两种投影后RMS归一化。',
        authorization='2026-10-03用户授权源域架构优化实验；保持原单CE、原数据与训练策略。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN,ANCHOR_CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','crosspath_relation','source_selection'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='f929f4d1116640474d324844e2f0824f97ca7dd0')
    spec['permissions']['claim_scope']='Architecture-only cross-path relation; projected-feature gain property is not physical channel removal or arbitraryRX invariance. Historically exposed benchmark; no target-derived design/ranking, hardware recovery, or first-blind-test claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],
        selection_rule='Own scratch fixedE200;max four-seed mean(0.5V+0.5worstsourceRX). Shallow and current source winner Anchor controls are source metadata only; no inherited weights or target scores.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-crosspath-relation-identity-20261003',
        runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],
        source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],
        remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_crosspath_relation_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Available capacity only;at most two total training jobs/GPU;>=12GB free;no healthy task interference')
    spec['metrics_plan'].update(primary='CompleteE200/10000steps CE,V/worstRX,fixedsource ranking;actual relation gradients/output/per-lag norms/floor fractions;90sourceTXRXday cells and resources',
        later_test='Only source-selected new variant:4new predictions+44immutable controls;48rows clean. If control retained,reuse its already completed test.')
    spec['notes']=[
        'Only architecture changes. Same original single cross_entropy/E200/AdamW/cosine/data/loader;no addedloss,augmentation,teacher,EMA,weighting,sampling,curriculum,stagedtraining or targetadaptation.',
        'Design uses source-only evidence and code. The cross-path relation and its projected-feature normalization are hypotheses; no causal physical-channel separation is asserted.',
        'Both variants from own scratch neural_residual_shallow, never Anchor initialization. Retain frequencybranch,160embedding,cosine30classifier and both original320InvariantReadout skips. Shallow and current source winner Anchor are fixed metadata-only controls.',
        'Time and behavior last maps [2,32,64] each receive learned complex32to8 pointwise projection. For lags0/4/8, use common behavior t=8..63 paired with time t-lag, exactly56 pairs each; compute uncentered complex cross moments. No padding or circular lag pairing.',
        'crosspath_gram uses whole projected branch RMS; crosspath_coherence uses each projected channel RMS, measured within the same lag window. Powerfloor1e-6; each lag is divided by8. Feature gain properties apply after learned projections above floors, not to arbitrary received channels.',
        'Flatten3lags×complex8×8=384; biasfreeLinear384to16(random) then biasfreeLinear16to320(zero), no activation, explicit relation rank<=16; add only to behavior320skip before existing projection. Time skip and all later projections/head remain original. No branch stop-gradient or orthogonality constraint.',
        'Both variants total233275 parameters, new12288=1024 complex projections+6144 compression+5120 zero exit over220987base. NewRNG isolated; initialfunction equals own scratchshallow. Hidden firststepzerograd is expected zero-exit behavior. NoRX/TX/role identifiers,queryclasscounts,crosspacket statistics ormutableteststate.',
        'Same L6300,U56700unused,V27000,RX1/3/4/6/8,day1/2/3,split392005,4modelseeds,batch128/nodrop,200x50steps,AdamW2e-4,wd1e-4,cosine1e-6,fullFP32/TF32False.',
        'Fixed16source records=8new+4shallowcontrols+4Anchorcontrols;maxmean0.5V+0.5worstRX,exactperformance ties then costs. No bestepoch,earlystop,targetreorder or selective rerun.',
        'SourceV uses seen RX;history/source docs may contain appended test results. Design inputs are explicit source-only CSV/JSON and code;incidental oldtarget exposure disclosed,not used. Finalconfirmation remains historicalbenchmark,not firstblind.',
        'Newcandidate winner only:4newclean+44frozen=48rows/384ALL+RXscores,same168000physicalquery/6TX/7RX. Allpredictionsfixed thenindependenttruthlast. NoLEO/support/SFT/newclasses.',
        'Detailed step/epoch/compactJSONL/CSV/text reports measuredCE/LR/gradients/relationoutputs/norms/floors/sourceV/time/memory. Lasttrainingbatch28 telemetry is not fullsource or physicalchannel causality evidence. MAC conv/matmul only,not totalFLOPs. All negativeoutcomes preserved.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='Only new source winner; otherwise reuse selected fixed control test with zero newquery',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=44,total_rows=48,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run=OLD_CLEAN_RUN,baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',
        candidate_universe=list(CANDIDATES),target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],
        target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json',
        'source_physical_diagnostics.json','source_selection.json','conditional independent clean test']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/cvs_crosspath_relation_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_crosspath_relation_identity',variant=variant,model_seed=seed,
                source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,
                augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                split_seed=392005,device='cuda:0',readout=readout_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_crosspath_relation_candidate',gpu=None,
                config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Onlymodel/loaderRNGvaries;fixedphysicalsplit;noaugmentation/support/evaluationRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_crosspath_relation_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),
                      ('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=8,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
