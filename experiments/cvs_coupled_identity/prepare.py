"""Two fixed causal envelope lags x four seeds; four fixed source controls."""
import copy
import json
from pathlib import Path
from experiments.cvs_coupled_identity.model import build,VARIANTS,coupled_contract
from experiments.cvs_coupled_identity.dispatch import control_rows,CANDIDATES
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-coupled-identity-manysig-m8-r01'
RELEASE='cvs_coupled_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-coupled-clean-manysig-m28-r01'


def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-coupled-envelope-identity-ce',
        display_name='CVS 因果包络耦合：固定12项输入与原读出，lag1/lag4×四seed纯CE',
        description='共享能量核心中以当前和滞后功率的固定均值构造三阶/五阶输入，保留原位置功率读出；lag1/4无新增参数，原数据划分、CE唯一、无增强、从零训练、源选后仅clean测试。',
        authorization='用户授权持续优化基础网络，性能第一，真正 RFF physics aware；普通 CE、无增强、clean-only，每次发布默认完成测试。',
        status='PLANNED',rows=[],parent_run_ids=['20261002-phase1-cvs-energy-identity-manysig-m8-r01'],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','causal_envelope_memory','fixed_parameter_budget','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='a3c6fdcaa1e70d7195250191f04d05b292622486')
    spec['data']['leo_config_ref']=None
    spec['permissions']['claim_scope']='Source-selected coupled-envelope architecture, raw received IQ and unchanged clean benchmark; no target-driven tuning, no first-blind-test or identified hardware claim'
    spec['checkpoint'].update(sources=[],selection_rule='Own scratch E200. Four-seed mean(0.5V+0.5worstsourceRX); energy sourcecontrol metadata only. No target input or inherited weights.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-coupled-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_coupled_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training jobs/GPU and >=12GB free; preserve others')
    spec['metrics_plan'].update(primary='Complete200epoch sourceCE/V/worstRX and fixedE200 performancefirst ranking; finalTXRXday/embedding geometry; actual coupled input, fixed public TX/RX cascade and whole-phase diagnosis',
        later_test='New source-selected candidate automatically receives4clean predictions+24frozen controls in '+CONFIRM_RUN+'. If current energy winner wins, retain original verified test; unselected new model testN/A.')
    spec['notes']=[
        'Design docs/CVS_COUPLED_ENVELOPE_HYPOTHESIS_20261002.md; algebra and public RF memory literature only; no target score used.',
        'Only behavior input changes: z[n-m]((P[n-m]+P[n-m-lag])/8)^q, m0..3/q0..2, lag1 or4; causal zero padding. Same12complex inputs, original position-power readout, depth/width/embedding/head and202553parameters; own scratch only.',
        'Both variants retain raw received IQ exactly, fixed alignment alpha0; no frequency correction, learned alignment, feature conditioner or extra loss.',
        'Cross-memory terms enlarge the input lift linear span relative to aligned terms; not a proof that the old full nonlinear network cannot approximate them. No fullGMP, DPD or hardware coefficient recovery.',
        '25MHz lag1/4 correspond to40ns/160ns design delays, not measured device time constants. Received envelope also includes RX/channel/noise effects; no uniqueTX or TX/RX causal identification.',
        'All6complex time/behavior blocks use one gain perpacket across channels/time. Relative channel energy fractions preserved by normalization only; learned scales/gates can change them.',
        '8scratch E200x50=80000CEupdates/1600epochs;L6300/V27000/U56700unused;split392005/batch128/AdamW2e-4/wd1e-4/cosine1e-6;no clipping/checkpoint/teacher/EMA/augmentation/domain/extra loss.',
        'FullFP32 cuDNNTF32False,matmulFalse/benchmarkFalse/deterministicFalse/highest. Same unchanged VALIDATED_ONCE data; no repeated validation. ConvLinear MAC excludes new elementwise lift cost; actual resources measured.',
        '12source records=8new+4currentenergy;four-seed E200 mean0.5V+0.5worstRX highest;exacttiesV/worstRX/cost/fixedorder. Public physics numerical tolerances not ranking/stopping gates.',
        'Detailedtext+stepJSONL+compactepochJSONLCSV; CE/LR/allparametergradient/rawalpha and scalargradN/A; actual coupled conv-input measurements and6block energy ratios, full90sourcecells and resources.',
        'Ifnewselected freeze4newclean+24oldcontrols=28truthlastrows;sourcecontrolwinner retains original verifiedtest, no newquery;unselectedN/A. No targetfeedback/selectivererun/LEO/SFT/newclasses.']
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='new_candidate_selected;otherwise retain current energy winner test without new/unselected query',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=24,total_rows=28,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run='20261002-phase1-cvs-energy-clean-manysig-m24-r01',baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',
        candidate_universe=list(CANDIDATES),target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],
        target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json','source_physical_diagnostics.json','source_selection.json','conditional independent clean test']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_coupled_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_coupled_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',coupled=coupled_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_coupled_envelope_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Model/loader RNG varies only; fixed physical split; fixed public diagnostic no RNG; no augmentation/support/target randomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_coupled_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_coupled_identity/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_coupled_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_coupled_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))


if __name__=='__main__':main()
