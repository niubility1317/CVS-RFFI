"""Preregister two architecture-only readout candidates, four scratch seeds."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_neural_readout_identity.model import build,VARIANTS,readout_contract
from experiments.cvs_neural_readout_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN,ANCHOR_CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01'
RELEASE='cvs_neural_readout_identity_20261003_r01'
CONFIRM_RUN='20261003-phase1-cvs-neural-readout-clean-manysig-m48-r01'
OLD_CLEAN_RUN='20261003-phase1-cvs-response-fusion-clean-manysig-m44-r01'

def main():
    configs=ROOT/'experiments/cvs_neural_readout_identity/configs'
    if configs.exists():raise FileExistsError('Preserve existing readout registration/configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-neural-readout-identity-ce',
        display_name='CVS纯网络读出优化：邻域注意力与可学习复混合×四seed',
        description='固定浅卷积残差骨干及原320维统计skip；两路加入4头邻域注意力读出，比较是否增加复通道混合。',
        authorization='2026-10-03用户要求继续优化完成目标；此前明确纯神经网络能力，不增加训练策略或额外损失。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN,ANCHOR_CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','neural_readout','context_attention','source_selection'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='f929f4d1116640474d324844e2f0824f97ca7dd0')
    spec['permissions']['claim_scope']='Architecture-only learned readout; historically exposed benchmark; no target-derived design/ranking, arbitraryRX invariance, hardware recovery, or first-blind-test claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],
        selection_rule='Own scratch fixedE200;max four-seed mean(0.5V+0.5worstsourceRX). Shallow and current source winner Anchor controls are source metadata only; no inherited weights or target scores.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-neural-readout-identity-20261003',
        runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],
        source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],
        remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_neural_readout_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Available capacity only;at most two total training jobs/GPU;>=12GB free;no healthy task interference')
    spec['metrics_plan'].update(primary='CompleteE200/10000steps CE,V/worstRX,fixedsource ranking;actual readout gradients/output/attention;90sourceTXRXday cells and resources',
        later_test='Only source-selected new variant:4new predictions+44immutable controls;48rows clean. If control retained,reuse its already completed test.')
    spec['notes']=[
        'Only architecture changes. Same original single cross_entropy/E200/AdamW/cosine/data/loader;no addedloss,augmentation,teacher,EMA,weighting,sampling,curriculum,stagedtraining or targetadaptation.',
        'Design derives from exact source CSV/selection evidence and architecture code: deeperconv lowers trainingCE but raises sourceVCE;fixed channelwise statistics may discard learnable channel/time information. The bottleneck remains a hypothesis.',
        'All models from scratch;retain neural_residual_shallow topology,frequencybranch,160embedding and cosine30classifier. Shallow is the matched direct backbone control; Anchor is the current source winner. Both are fixed metadata-only controls, chosen without target scores.',
        'Per time/behavior path,retain original320InvariantReadout as skip. Newlogpower32→Conv1d32to16k3pad1→GELU→Conv1d16to4k1→softmax over currentpacket64positions;4 weighted32values→flatten128→biasfreeLinear320zeroexit→addskip.',
        'Secondcandidate adds learned32to32complexpointwise mixing beforelogpower, initializedcomplexidentity. Bothscorestructures identical;two paths independent. NewRNG isolated,initialfunction exactly ownscratchshallow;hidden firststepzerograd expected,secondstep verified.',
        'Parameters306147/310243;new85160/89256 over220987base. NoRX/TX/role identifiers,queryclasscounts,batchstatistics ormutableteststate. Globalcommonphase property doesnot imply arbitraryRX/CFO invariance or TXhardware recovery.',
        'Same L6300,U56700unused,V27000,RX1/3/4/6/8,day1/2/3,split392005,4modelseeds,batch128/nodrop,200x50steps,AdamW2e-4,wd1e-4,cosine1e-6,fullFP32/TF32False.',
        'Fixed16source records=8new+4shallowcontrols+4Anchorcontrols;maxmean0.5V+0.5worstRX,exactperformance ties then costs. No bestepoch,earlystop,targetreorder or selective rerun.',
        'SourceV uses seen RX;history/source docs may contain appended test results. Design inputs are explicit source-only CSV/JSON and code;incidental oldtarget exposure disclosed,not used. Finalconfirmation remains historicalbenchmark,not firstblind.',
        'Newcandidate winner only:4newclean+44frozen=48rows/384ALL+RXscores,same168000physicalquery/6TX/7RX. Allpredictionsfixed thenindependenttruthlast. NoLEO/support/SFT/newclasses.',
        'Detailed step/epoch/compactJSONL/CSV/text reports measuredCE/LR/gradients/readoutattention/sourceV/time/memory. MAC conv/matmul only,not totalFLOPs. All negativeoutcomes preserved.'
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
            ref='experiments/cvs_neural_readout_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_neural_readout_identity',variant=variant,model_seed=seed,
                source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,
                augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                split_seed=392005,device='cuda:0',readout=readout_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_learned_readout_candidate',gpu=None,
                config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Onlymodel/loaderRNGvaries;fixedphysicalsplit;noaugmentation/support/evaluationRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_neural_readout_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),
                      ('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=8,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
