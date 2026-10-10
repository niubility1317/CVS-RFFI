"""Register fixed full-width architecture comparison before any target access."""
import copy
from experiments.standard_ce_baselines.common import *
from experiments.standard_ce_baselines.model import VARIANTS,build


def main():
    old=read(ROOT/'experiments/cvs_clean_design/configs/experiment_spec.json')
    previous=read(ROOT/'experiments/cvs_sixscene_eval/configs/launch_spec.json')
    spec=copy.deepcopy(old)
    spec.update(run_id=RUN,group_id='phase1-standard-fullwidth-ce-baselines',kind='comparison',
        display_name='常见大容量1D/2D网络纯CE：十二基准与两种CVS四seed逐行自动clean测试',
        description='Twelve common/full-width architecture controls plus native and frozen residual CVS; scratch CE only; fixed E200; each row predicts clean before separate scoring.',
        authorization='2026-10-10用户要求更多常见、更大容量网络并实际跑对比；沿用纯CE无额外训练策略要求及逐行自动测试协议。',
        stage='Phase1',status='PLANNED',parent_run_ids=[],replaces_run_id=None,rows=[],
        tags=['ce_only','no_augmentation','standard_architecture','full_width','four_seeds','automatic_row_test'])
    spec['data']['leo_config_ref']=None
    spec['data']['validation_ref']=previous['capsule']+'/manifest.json (VALIDATED_ONCE unchanged clean IQ)'
    spec['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',
        selection_rule='Every fixed comparator uses its own E200; no candidate ranking or target feedback')
    spec['permissions']=dict(regime='Phase1 source-only training then frozen inference',
        query_use='per-sample all-six-class clean inference; fixed prediction before independent scorer reads truth',
        external_method_exception=None,claim_scope='Previously exposed proxy benchmark, matched fixed-budget architecture comparison')
    spec['code']=dict(commit='release_commit.txt',checkout=str(ROOT),environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=PROJECT+'/releases/'+RELEASE)
    runtime=dict(run_id=RUN,launch_owner=OWNER,runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,
        capsule=previous['capsule'],truth=previous['truth'],rows=[])
    spec['execution']=dict(host='N607',launch_owner=OWNER,gpu_policy='At most2 total workers/GPU; >=12GB free; no intervention in other runs',
        remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.standard_ce_baselines.publish --output local_artifacts/'+RELEASE,
        stop_rule='Own row only on nonfinite, provenance mismatch, output collision, nonzero exit or incomplete prediction; retain artifacts, no automatic retry; low scores do not stop training')
    spec['metrics_plan']=dict(metric_names=['accuracy','macro_f1','worst_receiver_accuracy','parameters','gradient_used_parameters','training_time','inference_time','peak_memory'],
        dimensions=['architecture','model_seed','view','receiver','transmitter'],primary='all56 fixed rows: clean, 4-seed mean/sample SD and paired differences to both CVS controls',
        prediction_ref=runtime['runtime_root']+'/<row>/prediction/predictions.npz',scorer_ref='experiments/standard_ce_baselines/score.py')
    spec['expected_artifacts']=['pipeline_state.json','each source/last.pt','each source/initialization.json','each source/source_contract.json',
        'each source/resolved_config.json','each source/step_metrics.jsonl','each source/epoch_metrics.jsonl','each source/epoch_metrics.csv',
        'each source/resource_profile.json','each source/completion.json','each source/freeze.json',
        'each prediction/predictions.npz','each prediction/complete.json','each scores/scores.json','each scores/report.md','summary.json','report.md']
    spec['notes']=[
        'Only labeled L_s6300 used; U_s56700 unused; source V27000; physical split identical; no input augmentation, auxiliary loss, PL, EMA, distillation or pretraining.',
        'All56 scratch E200x50updates; AdamW2e-4/wd1e-4/cosine1e-6, batch128, full FP32, no clipping; fixed last checkpoint; no source or target ranking.',
        'CNNs preserve canonical depths and widths, replace 2D spatial operators with1D; params differ from ImageNet2D originals. VGG adaptive pool7 and two4096 classifier layers. ConvNeXt stochastic depth0, layer scale1e-6.',
        'Transformer6: d512/8heads/FFN2048/ReLU/postnorm/dropout0.1; BiLSTM3: hidden512 each direction/dropout0.1. Both concatenate four contiguous IQ pairs into64 tokens; no sample discarded.',
        'CNN standard Linear heads; CVS native cosine scale30/no label margin. Architecture-native dropout retained and disclosed.',
        'Complex CNN controls are complex ResNet18 IQ adaptations: published real-to-complex ResNet conversion approach, canonical2/2/2/2 blocks and64/128/256/512 complex widths; split real-imag BN/CReLU/pooling; classify packed1024 features. Not a width256 three-layer capacity control, nor an exact torchcvnn covariance-BN reproduction.',
        '2D CVCNN and ResNet18/50 use deterministic row-major IQ2x16x16 without interpolation or synthetic image conversion. 2D convolutions use full square kernels; widths/depths retained.',
        'model seed also controls shuffle/dropout; split392005 fixed; data/support/augmentation/evaluation seeds null because no such sampled role and clean fixed IDs.',
        'Each row independently completes train->E200 freeze->clean prediction->independent truth-last scoring->report. Scoring results never consumed by trainers or other row configuration.',
        'User explicitly specified automatic clean testing in this turn; no practical or LEO views scheduled.',
        'No smaller-parameter filter. Parameter count is measured, not a selection criterion. This benchmark was previously exposed and is not a new blind confirmation set.']
    counts={}
    for variant in VARIANTS:
        model=build(variant);counts[variant]=sum(p.numel() for p in model.parameters());del model
    write(ROOT/'experiments/standard_ce_baselines/configs/parameter_counts.json',counts)
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);source=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/standard_ce_baselines/configs/'+rid+'.json'
            config=dict(method='standard_ce_baselines',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],
                dataset=spec['data']['dataset'],output_root=source,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,
                weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],
                selection='fixed_last_epoch',split_seed=392005,device='cuda:0')
            write(ROOT/ref,config)
            row=dict(row_id=rid,method=variant,purpose='fixed_comparator',gpu=None,config_ref=ref,resolved_config_ref=source+'/resolved_config.json',
                data_overrides={},seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Model seed controls initialization/dropout/shuffle; no training augmentation/support; clean fixed physical IDs.',
                k=None,scenario=','.join(VIEWS),optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='200x50updates',
                output_root=runtime['runtime_root']+'/'+rid,log_path=runtime['log_root']+'/'+rid+'.source.log',
                command='python -m experiments.standard_ce_baselines.source --config '+PROJECT+'/releases/'+RELEASE+'/'+ref,
                expected_artifacts=spec['expected_artifacts'][1:-2])
            spec['rows'].append(row)
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=PROJECT+'/releases/'+RELEASE+'/'+ref,
                source_output=source,prediction_output=runtime['runtime_root']+'/'+rid+'/prediction',score_output=runtime['runtime_root']+'/'+rid+'/scores',
                source_contract=spec['data']['contract_ref']))
    write(ROOT/'experiments/standard_ce_baselines/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/standard_ce_baselines/configs/launch_spec.json',runtime)
    print(counts)

if __name__=='__main__':main()
