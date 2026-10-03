"""Fixed source experiment: static versus packet-conditioned full-input FIR."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_frontfilter_identity.model import build,VARIANTS,filter_contract
from experiments.cvs_frontfilter_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN,ANCHOR_CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-frontfilter-identity-manysig-m8-r01'
RELEASE='cvs_frontfilter_identity_20261003_r01'
CONFIRM_RUN='20261003-phase1-cvs-frontfilter-clean-manysig-m48-r01'
OLD_CLEAN_RUN='20261003-phase1-cvs-response-fusion-clean-manysig-m44-r01'

def main():
    configs=ROOT/'experiments/cvs_frontfilter_identity/configs'
    if configs.exists():raise FileExistsError('Preserve existing frontfilter configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-frontfilter-identity-ce',
        display_name='CVS全主干有界复FIR前置：静态与逐包动态×四seed',
        description='所有身份通路仅消费前置有界滤波后的IQ；比较全局静态与逐包神经预测的复系数。无原IQ身份旁路。',
        authorization='2026-10-03用户要求依据创新落地架构并跑实验，继续纯架构优化；保持原单CE及训练策略。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN,ANCHOR_CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','frontfilter','source_selection'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='7c9b77e1d22d6adb484c5932ef39abe3767a2fcd')
    spec['permissions']['claim_scope']='Architecture-only learned received-IQ front filter, not a recovered physical inverse or proven channel invariance. Benchmark historically exposed; no target-driven design/ranking or first-blind-test claim.'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],
        selection_rule='Own scratch fixed E200; max four-seed mean(0.5V+0.5worstsourceRX). Shallow and current source winner Anchor source metadata only; no inherited weights.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-frontfilter-identity-20261003',
        runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],
        remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_frontfilter_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Available capacity only; at most two total training jobs/GPU; >=12GB free; no healthy task interference')
    spec['metrics_plan'].update(primary='Complete E200/10000 steps CE,V/worstRX,source ranking; frontend coefficient/kernel/input change and gradients; 90sourceTXRXday cells and resources',
        later_test='Only a new source winner:4new+44immutable controls,48rows clean. Otherwise verify/reuse selected control test.')
    spec['notes']=[
        'Architecture-only change. Same original single cross_entropy/E200/AdamW/cosine/data/loader; no extra losses,augmentation,teacher,EMA,weighting,sampling,curriculum,staged training,target adaptation.',
        'Design evidence: prior source channel-attribution showed identity-G removed only0.00185pp and changed7/108000 predictions with an unfiltered main bypass; crosspath relation variants both lose all4seeds. These motivate a topology test, not proof of the cause.',
        'Both variants own scratch neural_residual_shallow. One frontend G_x is applied exactly once to input; time,Sinc,frequency,behavior and all identity statistics see G_x x. Original x only enters the coefficient context or explicitly labeled raw-input diagnostics. No identity bypass,auxiliary branch or postfilter RMS.',
        'G_x=I+.25 B_a, with4 learned complex5tap bases and complex4 coefficients. Each basis complex-modulusL1<=1; coefficientL1<=1. Same zero padding as priorG. Dynamic uses original8 context summaries,8to16 GELU16to8,zero last layer; static uses globalzero[2,4] coefficients. Both begin exactly at own scratchShallow and share basis initialization.',
        'Static adds48parameters, dynamic adds320. Their different coefficient generators are not parameter matched. Four common kernels,rho,taps,backbone and training fixed. No claim that any dynamic benefit is isolated from parameterization/capacity.',
        'For fixed coefficients the finite-window linear operator has singular values in[.75,1.25]. This is not global injectivity/Lipschitz of the dynamic network, TX information preservation, physical H inverse or nonlinear RX separation. Context may also carry identity; no causal disentanglement claim.',
        'SameL6300,U56700unused,V27000,RX1/3/4/6/8,days1/2/3,split392005,fourmodelseeds,batch128nodrop,E200x50,AdamW2e-4wd1e-4cosine1e-6,FP32TF32False.',
        'Fixed16source records=8new+4Shallow+4Anchor metadata controls. Max fourseed(.5V+.5worstRX), exactperformance ties V/worst thencost. No bestepoch/earlystop/targetreorder/seed replacement/selective rerun.',
        'Newsourcewinner only:4newclean+44frozen=48rows/384ALL+RXscores, same168000physicalquery/6TX/7RX; fixed predictions before independent truth-last. NoLEO/support/SFT/newclasses. Unselected candidates never query.',
        'Existing sourceV uses seenRX; source scores cannot prove unseenchannel/RX generalization. equalized1,center256,unitRMS data unchanged. Historicaltarget exposure disclosed, excluded from design/ranking. PurePhase1 paper/performance goal unproven.',
        'Detailed measured CE/LR/gradients/frontend bounds/output/sourceV/time/memory retained in fullstep,epoch,compactJSONL,CSV,text. Lastbatch28diagnostics labeled; original rawCFO summaries not interpreted as afterfilter compensation. Allnegative outcomes preserved.',
        'Final frozen all27000V forward captures the actual single frontend call and stores90TX/RX/day cells of input changes,norm ratios,coefficient/kernel L1 and8D coefficient means/trace variances; no extra fitting or source selection use.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='Only new source winner; otherwise reuse selected control with zero newquery',
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
            ref='experiments/cvs_frontfilter_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_frontfilter_identity',variant=variant,model_seed=seed,
                source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,
                augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                split_seed=392005,device='cuda:0',frontfilter=filter_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_full_backbone_frontfilter_candidate',gpu=None,
                config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Onlymodel/loaderRNGvaries; fixedphysicalsplit; noaugmentation/support/evaluationRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_frontfilter_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),
                      ('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=8,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
