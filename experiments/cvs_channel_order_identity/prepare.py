"""Preregister four architecture-only channel/order candidates, four scratch seeds."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_channel_order_identity.model import build,VARIANTS,channel_contract
from experiments.cvs_channel_order_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-channel-order-identity-manysig-m16-r01'
RELEASE='cvs_channel_order_identity_20261003_r01'
CONFIRM_RUN='20261003-phase1-cvs-channel-order-clean-manysig-m40-r01'
OLD_CLEAN_RUN='20261002-phase1-cvs-neural-residual-clean-manysig-m36-r01'

def main():
    configs=ROOT/'experiments/cvs_channel_order_identity/configs'
    if configs.exists():raise FileExistsError('Preserve existing channel registration/configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-channel-order-identity-ce',
        display_name='CVS信道结构优化：容量、补偿、双路与顺序差值×四seed',
        description='固定浅层控制骨干；比较匹配容量、逐包补偿、原始/补偿双路和非线性顺序差值结构。',
        authorization='2026-10-03用户明确要求“根据创新完成优化设计落地实现，跑实验”；遵循此前纯神经网络结构、原CE训练方案及无额外损失的边界。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','channel_order','packet_compensation','source_selection'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='f929f4d1116640474d324844e2f0824f97ca7dd0')
    spec['permissions']['claim_scope']='Architecture-only channel/order network; historically exposed benchmark; no target-derived design/ranking, arbitraryRX invariance, hardware recovery, or first-blind-test claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],
        selection_rule='Own scratch fixedE200;max four-seed mean(0.5V+0.5worstsourceRX). Current shallow control is source metadata only; no inherited weights or target scores.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-channel-order-identity-20261003',
        runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],
        source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],
        remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_channel_order_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Available capacity only;at most two total training jobs/GPU;>=12GB free;no healthy task interference')
    spec['metrics_plan'].update(primary='CompleteE200/10000steps CE,V/worstRX,fixedsource ranking;actual channel gradients/G magnitude/order-difference outputs;90sourceTXRXday cells and resources',
        later_test='Only source-selected new variant:4new predictions+36immutable controls;40rows clean. If control retained,reuse its already completed test.')
    spec['notes']=[
        'Only architecture changes. Same single cross_entropy/E200/AdamW/cosine/data/loader;no addedloss,augmentation,teacher,EMA,weighting,sampling,curriculum,stagedtraining or targetadaptation.',
        'All models scratch. Four structural comparisons:channel_capacity/channel_compensated/channel_dual/channel_order. No inherited checkpoint;immutable shallow control is source metadata only.',
        'Channel contract records actual operators/parameter counts. G is a learned bounded packet-specific compensator,not calibrated physical TX/RX/channel recovery. Order difference tests noncommutativity;no arbitrarychannel guarantee.',
        'NoRX/TX/role identifiers,queryclasscounts,batchstatistics or mutableteststate. Per-packet receivedIQ only;preserve full registered classifier competition.',
        'Same L6300,U56700unused,V27000,RX1/3/4/6/8,day1/2/3,split392005,4modelseeds,batch128/nodrop,200x50steps,AdamW2e-4,wd1e-4,cosine1e-6,fullFP32/TF32False.',
        'Fixed20source records=16new+4shallowcontrols;maxmean0.5V+0.5worstRX,exactperformance ties then costs. Capacity candidate can win. No bestepoch,earlystop,targetreorder or selective rerun.',
        'SourceV uses seenRX. Independent clean benchmark is historically exposed,not firstblind. No target score enters design or source ranking.',
        'Newcandidate winner only:4newclean+36frozen=40rows/320ALL+RXscores,same168000physicalquery/6TX/7RX. Allpredictionsfixed thenindependenttruthlast. NoLEO/support/SFT/newclasses.',
        'Detailed step/epoch/compactJSONL/CSV/text reports measuredCE/LR/gradients/channelG/orderD/sourceV/time/memory. InactiveG/D metrics are N/A,never fabricated0. MAC conv/matmul only,not totalFLOPs. Negativeoutcomes preserved.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='Only new source winner; otherwise reuse current shallow control test with zero newquery',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=36,total_rows=40,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run=OLD_CLEAN_RUN,baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',
        candidate_universe=list(CANDIDATES),target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],
        target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json',
        'source_physical_diagnostics.json','source_selection.json','conditional independent clean test']
    actual_counts={}
    for variant in VARIANTS:
        model=build(variant);actual=model.contract()
        if actual!=channel_contract(variant):raise ValueError('Actual model differs from preregistered channel contract')
        count=sum(p.numel() for p in model.parameters())
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count!=actual['total_parameters'] or trainable!=actual['total_trainable_parameters']:raise ValueError('Actual parameter accounting mismatch')
        actual_counts[variant]=dict(total_parameters=count,trainable_parameters=trainable,new_trainable_parameters=sum(p.numel() for p in model.channel_parameters() if p.requires_grad))
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/cvs_channel_order_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_channel_order_identity',variant=variant,model_seed=seed,
                source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,
                augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                split_seed=392005,device='cuda:0',channel=channel_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_channel_order_candidate',gpu=None,
                config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Onlymodel/loaderRNGvaries;fixedphysicalsplit;noaugmentation/support/evaluationRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_channel_order_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),
                      ('local_parameter_counts.json',actual_counts)]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=16,source_control_rows=4,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
