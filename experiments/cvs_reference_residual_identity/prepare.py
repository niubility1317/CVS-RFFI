"""Fixed architecture matrix and unconditional final clean tests, before training."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_reference_residual_identity.model import VARIANTS
from experiments.cvs_reference_residual_identity.contracts import architecture_contract,PRECISION
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-reference-residual-identity-manysig-m8-r01'
RELEASE='cvs_reference_residual_identity_20261003_r01'

def main():
    folder=ROOT/'experiments/cvs_reference_residual_identity/source_configs'
    if folder.exists():raise FileExistsError('Preserve existing registered source matrix')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_validdual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-reference-residual-fixed-ce',display_name='CVS参考约束残差：标量/逐频补偿固定八行对照',
        description='原始IQ/PA身份路径加可学习正则响应残差；仅原单CE，两臂各四seed从零E200，全部clean测试。',
        status='PLANNED',rows=[],parent_run_ids=['20261003-diagnostic-cvs-reference-residual-public-m2-r01'],
        tags=['cvs','reference_residual','ce_only','clean_only','fixed_comparison'])
    for key in ('registered_at','outcome'):spec.pop(key,None)
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    spec['permissions']['claim_scope']='Historically exposed clean benchmark; fixed architecture comparison, no target-driven selection, tuning or selective rerun; no blind confirmation claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],selection_rule='All8 own scratch fixed E200; no elimination; two preregistered matched-parameter arms')
    runtime=dict(run_id=RUN,launch_owner='codex/root/reference-residual-20261003',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],numerical_policy=FULL_FP32_POLICY)
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='local_artifacts/'+RELEASE,launch_command='python -m experiments.cvs_reference_residual_identity.publish --output local_artifacts/'+RELEASE)
    spec['metrics_plan']=dict(primary='Final clean accuracy,macroF1,perRX,perTX,four-seed mean/SD and paired inverse-scalar differences; source accuracy is not final test',
        training='E200x50; singleCE,LR,all gradient norms,ridge/branch gradients,reference fit,inverse magnitude,projection norm,sourceV,elapsed,peak memory; complete step/epoch logs and compact JSONL/CSV',
        mechanism='Complete sourceV/90TXRXday aggregate feature geometry; previous public cascade limitations preserved',
        resources='Actual parameters,state and nonpersistent buffer bytes,inference/training time,peak memory; explicitly partial MAC counts; physical solve FP64/complex128')
    spec['notes']=[
        'Source/public diagnostic motivated constrained compensation; original residual_fusion implementation is a fixed code anchor, not a test-selected trained checkpoint.',
        'Public probe does not support universal inverse compensation: IQ drift improved in some conditions while PA/memory drift worsened. Both arms retained as fixed research comparison; no promise of test gain.',
        'Both189562 parameters,new25337. Nine learned ridge values regularize9x9 apparent-response solve. Residual80 samples encoded and zero-exit added to raw160D identity. Original core/dropout RNG preserved.',
        'Original6300L,56700unusedU,27000V; oneCE only, no new loss/augmentation,teacher,EMA,optimizer change,resampling or checkpoint inheritance.',
        'Explicit mixed precision: trainable parameters/raw backbone FP32,physical branch FP64/complex128,output castFP32. No TF32,other numerical flags matched.',
        'All8 fixedE200 checkpoints unconditionally clean-tested on168000sameIDs; prediction fixed before independent truth-last scoring. No target feedback or selective rerun.',
        'No Phase2support,SFT,newclasses or satellite views; K and adaptation metrics N/A. Claimed real benefit requires test evidence.'
    ]
    spec['test_completion_plan'].update(condition='All8 source checkpoints E200 complete and frozen; unconditional fixed comparison',
        candidate_universe=list(VARIANTS),selection='fixed_E200_all_arms',target_feedback=False,
        output_root=PROJECT+'/runs/20261003-phase1-cvs-reference-residual-clean-manysig-m8-r01',
        historical_report_comparators=['native','residual_fusion'],
        comparison_scope='Same physical IDs/four seeds; old predictions reused for reporting only, not architecture or candidate choice')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json','frozen_source_matrix.json','clean_test results']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=f'{variant}-s{seed}';out=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/cvs_reference_residual_identity/source_configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            config=dict(method='cvs_reference_residual_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',response=architecture_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,config)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='fixed_comparison_arm',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',
                data_overrides={},seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Model and loader RNG fixed per model seed; same physical split; no augmentation/support/evaluation RNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300L/epoch;original singleCE',
                output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_reference_residual_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(folder/'experiment_spec.json',spec);write(folder/'launch_spec.json',runtime)
    print(RUN)

if __name__=='__main__':main()
