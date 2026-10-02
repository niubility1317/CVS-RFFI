"""Two fixed mixed-delay feature operators x four scratch CE source seeds."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_phase_curvature_identity.model import build,VARIANTS,curvature_contract
from experiments.cvs_phase_curvature_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-phase-curvature-identity-manysig-m8-r01'
RELEASE='cvs_phase_curvature_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-phase-curvature-clean-manysig-m36-r01'

def main():
    configs=ROOT/'experiments/cvs_phase_curvature_identity/configs'
    if configs.exists():raise FileExistsError('Preserve registered feature-curvature configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-feature-phase-curvature-identity-ce',
        display_name='CVS 六层相位曲率残差：混合延迟14/24×四seed纯CE',
        description='在六个复数FIR输出后、共享归一化前加入有界三阶延迟平衡差分，每块一个零初始化全局系数。增加6参数，保留原时域/行为宽深和读出、输入相位记忆控制。',
        authorization='用户持续优化CVS基础网络：数学/通信/物理/RFF结构分析；性能优先，CE唯一、无增强、原划分、仅clean测试。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','phase_curvature','six_extra_parameters','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='a31fd318996be20f030aeccd7ed6ed614ce7048d')
    spec['data']['leo_config_ref']=None
    spec['permissions']['claim_scope']='Source-selected received-feature phase-curvature inductive bias; no target feedback, first-blind-test, whole CFO invariance or unique TX hardware identification claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],selection_rule='Own scratch E200; four-seed mean(0.5V+0.5worstsourceRX). Adaptive control source metadata only; no checkpoint inheritance or target score reading.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-phase-curvature-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_phase_curvature_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Free capacity only; at most two total training jobs/GPU and >=12GB free; preserve healthy tasks')
    spec['metrics_plan'].update(primary='Complete200epochs/10000steps CE/V/worstRX and fixed E200 ranking; eight global gate values/gradients, actual six FIR curvature outputs, original12term adaptive input, six shared normalizations, final90sourceTXRXday cells, frozen public cascade and measured resources',
        later_test='If new source candidate selected:4new frozen clean predictions+32fixed controls='+CONFIRM_RUN+'. Otherwise reuse current adaptive completed clean; unselected candidates query N/A.')
    spec['notes']=[
        'Design docs/CVS_FEATURE_PHASE_CURVATURE_20261002.md. Mixed-lag feature interaction differs from previous crossphase readout and original input memory. Source/preamble diagnostics motivate received temporal field structure; no target score feedback.',
        'Each six ComplexConv outputs y uses v=y*min(1,2/sqrt(abs(y)^2+1e-6)); delta=(v[t-a]v[t-b]conj(v[t-a-b])-v[t]abs(v[t])^2)/4, prefix t<a+b masked; output y+tanh(beta)*delta.',
        'Two predeclared delay pairs(1,4) and(2,4); no tuning scan. Six scalar beta initialized0. Exact own-scratch adaptive_lag4 initial function/state under matching RNG; no new random draws or historical weights. Original alpha input gates2 retained.',
        'Constant-amplitude affine-phase delta null, local operator affine-phase covariance, boundabsdelta<=4 proved/tested. This is a classification inductive bias, not hardware coefficient recovery or whole-network CFO invariance. Original noncausal FIR padding retained.',
        '202561 parameters(+6 vs202555 control); unchanged channel/depth/readout/160embedding/cosine30head. Feature grid strides[2,4,4,2,4,4] original samples; no role/class/RX labels, batch pooling, future/wraparound in added operator or persistent fitting state.',
        '8own-scratch E200x50=80000CEupdates/1600epochs;L6300,V27000,U56700unused;split392005,b128/no drop,AdamW2e-4,wd1e-4,cosine1e-6,no clipping/EMA/teacher/augmentation/domain/extra loss.',
        'FullFP32:TF32False/benchmarkFalse/deterministicFalse/highest. Detailed text plus fullstepJSONL/epochJSONL/compactJSONL/CSV. Measured six memory scalar signed and absolute gradients, raw/tanhbeforeafter and actual output delta/formula/prefix; retained2 input gates and all202561 CE parameter gradients.',
        '12 source ranking records=8new+4current adaptive metadata controls. Highest four-seed mean0.5V+0.5worstRX; exact tiesV/worstRX,thenMAC/parameters/fixedorder. Fixed E200 only; public numeric responses are descriptive, never selection/stopping gates.',
        'Selected candidate only:4newclean+32immutable current controls=36rows/288ALL+RXrecords,168000samephysicalID/6TX/7RX; complete predictions before separatetruthlast. No new selected =>reuse completed adaptive controlclean, newcandidatecleanN/A. No LEO/SFT/support/newclasses or targetfeedback/selectivererun.',
        'Same physical data roles/VALIDATED_ONCE capsule; no repeated data validation. ConvLinear MAC excludes newelementwise mixedlag products/clipping; measure actual time/memory/state and report this limit.']
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
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_phase_curvature_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_phase_curvature_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',curvature=curvature_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_feature_phase_curvature_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Onlymodel/loaderRNGvaries;fixedphysicalsplit;deterministicpublicdiagnostics;noaug/support/evalRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_phase_curvature_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
