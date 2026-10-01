"""One whole-identity phase-equivariant hypothesis x4 seeds, with4 controls."""
import copy
import json
from pathlib import Path
from experiments.cvs_equivariant_identity.model import build,VARIANTS,equivariant_contract
from experiments.cvs_equivariant_identity.dispatch import control_rows,CANDIDATES
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-equivariant-identity-manysig-m4-r01'
RELEASE='cvs_equivariant_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-equivariant-clean-manysig-m24-r01'


def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-equivariant-memory-ce',display_name='CVS 全路径复相位等变记忆网络：纯 CE 四 seed 源域研发',
        description='Sinc 后的时间路径与 received 记忆路径保持复相位等变，以功率和复相关读出；保留原频谱/融合/余弦头。所有身份路径公共相位不变，普通 CE、无增强、scratch，四 seed 与原残差源控制比较，选中后默认 clean。',
        authorization='用户授权持续优化基础网络，性能第一，真正 RFF physics aware；普通 CE、无增强、clean-only，每次发布默认完成测试。',
        status='PLANNED',rows=[],parent_run_ids=['20261001-phase1-cvs-residual-identity-manysig-m8-r01','20261002-phase1-cvs-reference-identity-manysig-m4-r01'],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','phase_equivariant','complex_memory','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE)
    spec['checkpoint'].update(sources=[],selection_rule='Own scratch E200. Fourseed 0.5V+0.5worstsourceRX; original residual sourcecontrol metrics only. No target input or inherited weights.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-equivariant-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows())
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_equivariant_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training jobs/GPU and >=12GB free; preserve others')
    spec['metrics_plan'].update(primary='Complete200epoch sourceCE/V/worstRX and fixedE200 performancefirst ranking; finalTXRXday/embedding geometry; fixed controlled TX/RX cascade and whole-phase diagnosis',
        later_test='New source-selected candidate automatically receives4clean predictions+20frozen controls in '+CONFIRM_RUN+'. If original residual wins, retain original verified test; unselected new model testN/A.')
    spec['notes']=[
        'Architecture hypothesis docs/CVS_EQUIVARIANT_IDENTITY_HYPOTHESIS_20261002.md; source-only previous full representation phase sensitivity motivated all-path constraint. No target-score input.',
        'Complex tied-sign bias-free filters, per-packet isotropic RMS, real radial gate; final power/lag1/lag2 Hermitian statistics and four positional power bins. No hard phase reference/delay route.',
        'Time Sinc24 -> complex24/32/32, behavior 12 MP terms ->complex24/32/32; native phase-invariant rawFFT/fusion/residual cosine head retained. No unconstrainedraw identity bypass.',
        'Only constant complexphase invariance; CFO, LTI and RX image/nonlinear remain. ReceivedMP orders1/3/5 and delays0..3, radialclip4, scaling2^(order-1); sharedfilters are not recoveredTXcoefficients.',
        'Behavior basis and filters are causal, but per-packet RMS/readout use the entire packet; whole classifier is not claimed online-causal or a physical TX simulator.',
        'ScratchE200x50,6300LperEpoch,27000Vreadonly,batch128;AdamW2e-4/wd1e-4/cosine1e-6;FP32,noclip;CEweight1;noaugmentation/domain/teacher/EMA/margin/Uuse.',
        'Exactly8source records:4new+4immutableoriginal sourcecontrol;maximum fourseed0.5V+0.5worstRX;exacttiesV,worstRX,ConvLinearMACs,parameters,baseline/equivariant order. No cost tolerance band.',
        'Full logs measured CE/LR/gradient/actual equivariant status/sourceV/time; complete stepJSONL and compactepochJSONL/CSV. Final27000V/90TXRXday groups.',
        'Frozen30TXRX cascade and3globalphase tests are explanatory only, noaugmentation/selection/stopping. TX/RXsamewaveform confounds retained; not exactWiSig equalizer/noise/onset.',
        'Cost secondary; report actual realtiedConvMACs,FFT,fullparameters,train/inferenceelapsed/peak. All sourceparameters CEgrads are measured, not guessed.',
        'Selectednewonly4query;original20predictions reused readonly;all24 scored truth-last afterpredictioncomplete. Exposedclean6classproxybenchmark, noLEO/SFT/newclasses. No test feedback/selective retry.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='new_candidate_selected;otherwise retain original residual test without new/unselected query',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=20,total_rows=24,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run='20261001-phase1-cvs-selected-clean-manysig-m20-r01',baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',
        candidate_universe=list(CANDIDATES),target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],
        target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_metrics.csv','source_final_diagnostics.json','source_physical_diagnostics.json','source_selection.json','conditional independent clean test']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_equivariant_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_equivariant_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',equivariant=equivariant_contract(variant))
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_whole_phase_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Model/loader RNG varies only; fixedphysicalsplit; fixedsyntheticdiagnostic noRNG, noaugmentation/support/targetrandomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_equivariant_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_equivariant_identity/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_equivariant_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_equivariant_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,new_rows=4,source_control_rows=4,candidate_universe=list(CANDIDATES))))


if __name__=='__main__':main()
