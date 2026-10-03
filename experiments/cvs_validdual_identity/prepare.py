"""Fixed static/dynamic architectural comparison, all eight receive final clean test."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_validdual_identity.model import VARIANTS,dual_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-validdual-identity-manysig-m8-r01'
RELEASE='cvs_validdual_identity_20261003_r01'

def main():
    configs=ROOT/'experiments/cvs_validdual_identity/configs'
    if configs.exists():raise FileExistsError('Preserve existing experiment')
    old=ROOT/'experiments/cvs_frontfilter_identity/configs/experiment_spec.json'
    spec=copy.deepcopy(json.loads(old.read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-valid-history-dual-ce',display_name='CVS有效历史33点滤波与原始指纹双路径：静态/动态固定对照',
        description='原始IQ与有效历史有界滤波共享同一主干，逐特征融合；仅原单CE。静态和动态各四seed从零E200，全部冻结后clean测试。',
        status='PLANNED',rows=[],parent_run_ids=['20261003-diagnostic-cvs-mirror-history-public-m8-r01'],
        tags=['cvs','ce_only','clean_only','valid_history','dual_path','fixed_comparison'])
    spec.pop('source_controls',None)
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    spec['permissions']['claim_scope']='Fixed architecture comparison, historically exposed clean benchmark; no target-driven model choice or retraining; no blind-confirmation claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],selection_rule='Every row own scratch fixed E200; no candidate elimination; static/dynamic are preregistered fixed comparison arms')
    runtime=dict(run_id=RUN,launch_owner='codex/root/validdual-20261003',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],numerical_policy=FULL_FP32_POLICY)
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_validdual_identity.publish --output local_artifacts/'+RELEASE)
    spec['metrics_plan']=dict(primary='Final clean accuracy,MacroF1,perRX,perTX,four-seed mean/SD and paired static/dynamic differences; no source-score replacement',
        training='E200x50;singleCE,LR,all parameter gradients,gate/kernel,input change,sourceV,worstsourceRX,time,memory; full logs and compact JSONL/CSV',
        mechanism='Public fixed-kernel capability and complete27000sourceV/90TXRXday coefficient cells; no target feedback',
        resources='Actual parameter count,state bytes,training/inference times,MACs,peak memory; two shared-backbone paths increase work')
    spec['notes']=[
        'Design premise is previous public FIR-history intervention, not target test ranking. Existing shallow backbone retained as architectural control; no historical weights loaded.',
        'G(x)=x+.5*mask*C_k(x),33causal taps with complex L1<=1. Only224 fully observed windows are used; first32IQ untouched; first16valid corrections raised cosine ramp.',
        'Raw and corrected IQ feed one weight-shared backbone call with2B internal paths. z=zraw+sigmoid(gate160)*(zcorr-zraw). No extra labels,samples,loss,augmentation,teacher,EMA,source resampling or optimizer schedule change.',
        'Static coefficient66parameters; dynamic66to64GELU64to66 current-packet generator. Both add160fusion logits. Static/dynamic capacities differ and are disclosed; no pure capacity-controlled causality claim.',
        'Exact scratch equality in eval at zero filter exit. Original dropout remains independent in two train arms; training stochastic equivalence is not claimed. Report extra compute.',
        'Fixed-kernel correction operator norm<=.5; not dynamic-map injectivity,identified channel inverse,TX/RX disentanglement,or general FIR invariance.',
        'All8E200checkpoints receive preregistered clean test regardless of source score. Predictions fixed before independent truth-last scoring. No target-driven rerank or selective rerun.',
        'No Phase2support/SFT/newclass/LEO in this run. K and adaptation metrics N/A. Original6300L/56700unusedU/27000V source roles and4modelseeds retained.'
    ]
    spec['test_completion_plan']=dict(condition='Unconditional fixed static/dynamic comparison after all8 source checkpoints complete and freeze',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=8,reused_rows=0,total_rows=8,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],
        candidate_universe=list(VARIANTS),target_feedback=False,selection='fixed_E200_all_arms',
        output_root=runtime['runtime_root']+'/clean_test',expected_artifacts=['8 clean_predictions.npz','clean_scored_results.json','clean_summary.csv','independent_recount.json'],
        historical_report_comparators=['native','residual_fusion','neural_residual_shallow','frontfilter_static','frontfilter_dynamic'],
        comparison_scope='Same recorded physical IDs/seeds; historic scores for reporting only, no new target-based selection')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json','source_physical_diagnostics.json','frozen_source_matrix.json','clean_test results']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=f'{variant}-s{seed}';out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_validdual_identity/configs/'+rid+'.json'
            remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_validdual_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                split_seed=392005,device='cuda:0',validdual=dual_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='fixed_comparison_arm',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Model and loader RNG varied; same fixed physical split; no augmentation/support/evaluation RNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300L/epoch;shared backbone two paths',output_root=out,
                log_path=runtime['log_root']+'/'+rid+'.log',command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_validdual_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(configs/'experiment_spec.json',spec);write(configs/'launch_spec.json',runtime)
    print(RUN)
if __name__=='__main__':main()
