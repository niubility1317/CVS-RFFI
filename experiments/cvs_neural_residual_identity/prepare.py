"""Two learned residual depths x four scratch CE source seeds."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_neural_residual_identity.model import build,VARIANTS,neural_contract
from experiments.cvs_neural_residual_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-neural-residual-identity-manysig-m8-r01'
RELEASE='cvs_neural_residual_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-neural-residual-clean-manysig-m36-r01'

def main():
    configs=ROOT/'experiments/cvs_neural_residual_identity/configs'
    if configs.exists():raise FileExistsError('Preserve registered feature-curvature configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-feature-neural-residual-identity-ce',
        display_name='CVS可学习复卷积残差：浅深两容量×四seed纯CE',
        description='保留现有adaptive骨干，两支路末端增加1或2个32→64→32复卷积残差模块；入口/时域/出口全部可学习，仅使用原CE。',
        authorization='2026-10-02用户明确只靠神经网络能力，不借助其他训练策略或额外损失；继续完成性能优化目标。沿用原数据/训练方案/固定源选模，仅clean测试。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','neural_residual','learned_convolutions','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='a31fd318996be20f030aeccd7ed6ed614ce7048d')
    spec['data']['leo_config_ref']=None
    spec['permissions']['claim_scope']='Source-selected learned neural residual capacity; no additional training strategy/loss, target feedback, first-blind-test or arbitraryRX invariance claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],selection_rule='Own scratch E200; four-seed mean(0.5V+0.5worstsourceRX). Adaptive control source metadata only; no checkpoint inheritance or target score reading.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-neural-residual-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_neural_residual_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Free capacity only; at most two total training jobs/GPU and >=12GB free; preserve healthy tasks')
    spec['metrics_plan'].update(primary='CompleteE200/10000steps singleCE,V/worstRX,fixedsource ranking; measured neural gradients and outputs; final90sourceTXRXday cells; original frozenpublic diagnostics and resources',
        later_test='Selected new candidate:4new frozen clean+32fixed controls='+CONFIRM_RUN+'. Otherwise reuse already completed adaptive clean.')
    spec['notes']=[
        'User explicitly requires neural architecture only: single unchanged cross_entropy; no new loss, augmentation, weighting, sampling, teacher, contrastive objective, curriculum, fine-tuning or domain training.',
        'Design docs/CVS_NEURAL_RESIDUAL_20261002.md; diagnosed from full source curves and frozen-source attribution only. Historical target results are not inputs to architecture, hyperparameters, source selection or reruns.',
        'Retain adaptive_volterra_lag4 architecture but initialize every model from scratch. No inherited checkpoint. Append within each third block one(shallow) or two(deep) learned complex residual modules.',
        'Each module:32→64 complexpointwise/sharedRMS/radialgate→64 depthwisecomplex k5/sharedRMS/radialgate→32 complexpointwise zeroexit→identitysum. No scalar mixing gate or handcrafted new polynomial.',
        'Perpacket sharedRMS preserves relativechannelenergy at normalization; behavior temporal leftpadding causal, time symmetric. Same160embedding/readout/cosine30head. NoRX/class/roleinput or mutableinferencefitting state.',
        'Shallow220987(+18432),deep239419(+36864) parameters. Additional initialization is RNG-isolated; both initially reproduce ownscratch adaptive output exactly. FirstCEstep hiddenresidualgradient zero is expected with zeroexit; after exitupdate innergradients must be measured.',
        'SameE200x50steps,L6300,V27000,U56700unused,split392005,4modelseeds,b128/nodrop,AdamW2e-4,wd1e-4,cosine1e-6,fullFP32/TF32False. No addedoptimizerstrategy; source label use onlyCE.',
        '12 source ranking records=8new+4immutable adaptive metadata controls. Highest four-seed mean0.5V+0.5worstRX; exact performance ties then costs; fixedE200. Never pick bestepoch or use target feedback.',
        'Selectednewcandidate only:4newclean predictions+32existing frozencontrols=36rows/288ALL+RXrecords;168000samephysicalquery/6TX/7RX; fixedpredictions thenindependenttruthlast. Ifcontrolretained,reuse completedclean and no newquery. NoLEO/support/SFT/newclasses.',
        'Fullstep,epoch,compactJSONL/CSV and text record measuredCE/weight/LR/gradients/neuralnorms/outputchanges/sourcevalidation/time/memory. ResourceMAC includes actual aten.convolution and matrix multiplies; excludes FFT, normalization, gating, pooling and elementwise arithmetic. Actualwalltime/CUDApeak measured.',
        'SourceVsharesfiveRXwithsourceL. This study doesnotclaim unseenRXgeneralization until frozenindependentclean evaluation. Negativeoutcomespreserved.']
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='new_candidate_selected; otherwise reuse current adaptive verified test, zero new query',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=32,total_rows=36,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run='20261002-phase1-cvs-adaptive-volterra-clean-manysig-m32-r01',
        baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',candidate_universe=list(CANDIDATES),
        target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],
        target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json','source_physical_diagnostics.json','source_selection.json','conditional independent clean test']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_neural_residual_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_neural_residual_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',neural=neural_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_learned_neural_residual_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Onlymodel/loaderRNGvaries;fixedphysicalsplit;deterministicpublicdiagnostics;noaug/support/evalRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_neural_residual_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
