"""Learned instant/memory packet residuals x four scratch seeds; adaptive source control metadata only."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_moment_residual_identity.model import build,VARIANTS,moment_contract
from experiments.cvs_moment_residual_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-moment-residual-identity-manysig-m8-r01'
RELEASE='cvs_moment_residual_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-moment-residual-clean-manysig-m36-r01'

def main():
    if (ROOT/'experiments/cvs_moment_residual_identity/configs').exists():raise FileExistsError('Preserve registered moment residual configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-packet-moment-residual-identity-ce',
        display_name='CVS 可学习矩残差：保留相位记忆控制，瞬时/记忆4×四seed纯CE',
        description='在原adaptive_lag4输入上增加两个零初始化全局tanh系数，由源CE学习三阶/五阶包内矩修正；保留原坐标参照及同种子初始函数，仅新增2参数，深宽/读出/物理划分固定，源选后仅clean测试。',
        authorization='用户持续优化CVS基础网络：性能第一，数学/通信/物理/RFF结构分析，CE唯一，无增强，原数据划分，测试只clean。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','packet_moment_residual','two_extra_parameters','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),architecture_base_commit='6a79d2cfc7bfe84f0d8bb8e69729b2c85bfc36d4')
    spec['data']['leo_config_ref']=None
    spec['permissions']['claim_scope']='Source-selected received-envelope coordinate hypothesis; no target-driven tuning, first-blind-test, whole-CFO invariance or unique TX/RX hardware identification claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],selection_rule='Own scratch E200; four-seed mean(0.5V+0.5worstsourceRX). Current adaptive source winner metadata only; no weight inheritance or test score reading.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-moment-residual-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_moment_residual_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training jobs/GPU and >=12GB free; preserve healthy tasks')
    spec['metrics_plan'].update(primary='Full200epoch/10000step CE/V/worstRX and fixedE200 ranking; allfour gate values/gradients/coefficients, actual12term moment residual conv-input, six sharednormalization blocks, final90sourceTXRXday cells, publicTXRX confounds/phase response/resources',
        later_test='If new source candidate selected,4new frozen clean predictions+32fixed controls='+CONFIRM_RUN+'. Otherwise retain existing adaptive clean; unselected models never access query.')
    spec['notes']=[
        'Design docs/CVS_LEARNED_MOMENT_RESIDUAL_20261002.md. New hypothesis follows source negative orthopoly result and public coordinate inverse; no target score feedback or selection.',
        'Current adaptive lag4 scratch baseline A=B+tanh(a)*(Q-B), two phase-memory gates. Add two zero gates beta to A+tanh(beta)*(O-B), independently for order3/5; O is unscaled same-packet centered order1/3/5 with envelope lag0 or4. B always coupled_lag4. No fixed coefficient scan.',
        'Twelve complex inputs, unchanged kernels/width/depth/position-power readout/160embedding/cosine30head and202557 parameters(+2 vs202555control). Four totalglobalgates. Shared own-scratch initialization and matching train-mode dropout RNG preserve initial function; no historical checkpoint load.',
        'Existing raw order1 and native time/frequency paths retained. Moment residual scale1,1/4,1/16 undoes previous orthopoly scales4/16; original polynomial coordinates remain explicit reference. Learned nonzero beta changes the function; no guarantee of better conditioning or performance.',
        'Packet moments use only current received256IQ/causal envelope delays; differentiable, no TX/RX/class/role labels, crossbatch pooling, persistent state, fitting, adapter or target input. Fullpacket moment evaluation is not streaming causal.',
        'Same lag affine-phase covariance and constantphase property only. No arbitraryCFO/RX/LTI invariance, completeVolterra, recoveredPA or uniqueTXhardware claim; existing public TX/RX samewaveform confounds retained.',
        '8scratch E200x50=80000CEupdates/1600epochs;L6300,V27000,U56700unused;split392005,b128,AdamW2e-4,wd1e-4,cosine1e-6,no clipping/checkpoint/EMA/teacher/augmentation/domain/extra loss.',
        'FullFP32 TF32False/benchmarkFalse/deterministicFalse/highest. Detailedtext,allstepJSONL,epochJSONL/compactJSONL/CSV contain measured fourgate gradients/raw/tanhbeforeafter,CE/LR/all202557gradparams,alpha0 and actualmethod flags; no target validation.',
        '12source records=8new+4current adaptive source controls. Highest four-seed mean0.5V+0.5worstRX; exact tiesV/worstRX thenMAC/parameters/fixedorder. E200 only; physical numeric outcomes reportonly, not ranking or stopping.',
        'If selected,4newclean+32original immutable control predictions=36rows/288ALL+RXrecords,168000samephysicalID/6TX/7RX; all predictions fixed before independenttruthlast. Otherwise reuse verified adaptive controlclean and unselected newqueryN/A. No LEO/SFT/support/newclasses/targetfeedback/selectivererun.',
        'Existing data roles unchanged, no repeated data validation. ConvLinear MAC excludes packet moments/elementwise operations; actual time/memory/state measured, no equal totalcompute claim.']
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
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_moment_residual_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_moment_residual_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',moment=moment_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_learned_packet_moment_residual_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Model/loader RNG varies only; fixed physical split; public deterministic diagnostics; no augmentation/support/evaluation RNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_moment_residual_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:
        write(ROOT/'experiments/cvs_moment_residual_identity/configs'/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
