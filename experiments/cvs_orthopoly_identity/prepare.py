"""Fixed instantaneous/memory envelope coordinates x four scratch seeds."""
import copy,json
from pathlib import Path
from experiments.cvs_orthopoly_identity.model import build,VARIANTS,orthopoly_contract
from experiments.cvs_orthopoly_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-orthopoly-identity-manysig-m8-r01'
RELEASE='cvs_orthopoly_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-orthopoly-clean-manysig-m36-r01'

def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-packet-orthogonal-envelope-identity-ce',
        display_name='CVS 包内正交包络输入：原 IQ 保留，瞬时/记忆4×四seed纯CE',
        description='仅改变首个行为卷积的12项输入，用包内加权矩正交化三阶/五阶包络项，保留原始IQ和高阶能量；无新增参数，原物理划分、CE唯一、无增强、从零训练、源选后仅clean测试。',
        authorization='用户持续优化CVS基础网络：性能第一，数学/通信/物理/RFF结构分析，CE唯一，无增强，原数据划分，测试只clean。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','packet_orthogonal_envelope','fixed_parameter_budget','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='7084d694ee940d7f235c1f3b210e8a471fce1308')
    spec['data']['leo_config_ref']=None
    spec['permissions']['claim_scope']='Source-selected received-envelope coordinate hypothesis; no target-driven tuning, first-blind-test, whole-CFO invariance or unique TX/RX hardware identification claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],selection_rule='Own scratch E200; four-seed mean(0.5V+0.5worstsourceRX). Current adaptive source winner metadata only; no weight inheritance or test score reading.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-orthopoly-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_orthopoly_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training jobs/GPU and >=12GB free; preserve healthy tasks')
    spec['metrics_plan'].update(primary='Full200epoch CE/V/worstRX and fixedE200 performance ranking; final90 TX/RX/day cells; actual input Gram/reconstruction/order energy; public TX/RX confounds, phase response and resources',
        later_test='If new source candidate selected,4new frozen clean predictions+32fixed controls='+CONFIRM_RUN+'. Otherwise retain existing adaptive clean; unselected models never access query.')
    spec['notes']=[
        'Design docs/CVS_PACKET_ORTHOGONAL_ENVELOPE_20261002.md. Own closed-form derivation and primary PA orthogonal-model literature motivate conditioning only, no target score used.',
        'Two fixed variants: p=abs(z)^2/4 or p=(abs(z)^2+delay4(abs(z)^2))/8; each lag m0..3 independently uses weightabs(z)^2,mu,var,skew/max(var,1e-8). Inputs z;4z(p-mu);16z((p-mu)^2-a(p-mu)-var).',
        'Raw clipped IQ retained exactly. Twelve complex terms, fixed scales1/4/16, same conv/head/position-power readout and202553 parameters (current control202555). No per-order energy whitening or learned projection/gate.',
        'Moments depend only on the same received256point packet; no labels/RX/TX/roles, batch pooling, persistent state, optimizer or adaptation objective. Full-packet moments are not streaming causal even though envelope delays are causal.',
        'Eligible same-packet same-lag order terms orthogonal; degeneracy guarded. Original aligned polynomial terms exactly reconstructed by packet-specific triangular formulas. No claim of whole-network function equivalence or orthogonality across12lag terms/classes.',
        'Constant phase charge1 and delay-dependent affine phase charge1 of input only; whole affine-phase/CFO/RX/LTI invariance and unique TX hardware parameter recovery unproven. Received nonlinear terms can originate in RX/channel as well as TX.',
        '8scratch E200x50=80000CEupdates/1600epochs;L6300,V27000,U56700unused;split392005,b128,AdamW2e-4,wd1e-4,cosine1e-6,no clipping/checkpoint/EMA/teacher/augmentation/domain/extra loss.',
        'FullFP32 TF32False/benchmarkFalse/deterministicFalse/highest; existing data roles unchanged, no repeated data validation. ConvLinear MAC does not count packet moments/elementwise operations; actual time/memory measured.',
        '12source records=8new+4current adaptive source controls. Highest four-seed mean0.5V+0.5worstRX; exact ties V/worstRX/MAC/parameters/fixedorder. Public numeric outcomes reported, not performance selection/stopping gates.',
        'Detailed text+all stepJSONL+epochJSONL/compactJSONL/CSV; actual CE/LR/fullparametergradients,alpha0; actual last28source input Gram/raw/reconstruction/orderenergy, six normalization blocks; final90sourceV cells, public30cascade cases.',
        'If selected,4new clean+32immutable old control predictions=36rows, independent truth-last scoring after all fixed. Otherwise current control test reused and new/unselected queryN/A. No LEO/SFT/support/newclass/target feedback/selective rerun.']
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
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_orthopoly_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_orthopoly_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',orthopoly=orthopoly_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_packet_orthogonal_envelope_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Model/loader RNG varies only; fixed physical split; public deterministic diagnostics; no augmentation/support/evaluation RNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_orthopoly_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:
        write(ROOT/'experiments/cvs_orthopoly_identity/configs'/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
