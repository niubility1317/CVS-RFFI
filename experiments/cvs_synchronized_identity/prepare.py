"""One whole-identity phase-synchronized hypothesis x4 seeds, with4 controls."""
import copy
import json
from pathlib import Path
from experiments.cvs_synchronized_identity.model import build,VARIANTS,synchronized_contract
from experiments.cvs_synchronized_identity.dispatch import control_rows,CANDIDATES
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-synchronized-identity-manysig-m8-r01'
RELEASE='cvs_synchronized_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-synchronized-clean-manysig-m24-r01'


def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-synchronized-identity-ce',display_name='CVS 全路径逐包频偏同步：两种身份核心×四seed纯CE源研发',
        description='固定源公共及完整V敏感性支持逐包同步；全部身份路径前的20点重复相关估计相对CFO，保留样点幅度/非仿射相位，复等变或相位规范化核心从零训练。普通CE无增强/同划分，四seed性能最高优先，选中后默认clean。',
        authorization='用户授权持续优化基础网络，性能第一，真正 RFF physics aware；普通 CE、无增强、clean-only，每次发布默认完成测试。',
        status='PLANNED',rows=[],parent_run_ids=['20261001-phase1-cvs-residual-identity-manysig-m8-r01','20261002-diagnostic-cvs-received-sensitivity-source-manysig-m56-r01'],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','packet_synchronization','relative_cfo','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE)
    spec['checkpoint'].update(sources=[],selection_rule='Own scratch E200. Fourseed 0.5V+0.5worstsourceRX; original residual sourcecontrol metrics only. No target input or inherited weights.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-synchronized-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_synchronized_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training jobs/GPU and >=12GB free; preserve others')
    spec['metrics_plan'].update(primary='Complete200epoch sourceCE/V/worstRX and fixedE200 performancefirst ranking; finalTXRXday/embedding geometry; fixed controlled TX/RX cascade and whole-phase diagnosis',
        later_test='New source-selected candidate automatically receives4clean predictions+20frozen controls in '+CONFIRM_RUN+'. If original residual wins, retain original verified test; unselected new model testN/A.')
    spec['notes']=['Architecture hypothesis docs/CVS_SYNCHRONIZED_IDENTITY_HYPOTHESIS_20261002.md. Only source/public and fullV56 sensitivity motivate candidate;no target-score or breakdown input.', 'All256received samples rotate by a per-packet estimated linear phase before Sinc/time/frequency/behavior paths. Deterministic architectural representation, not random training/channel augmentation.', '80:160=4x20 repeated field;60 adjacent-cycle pairs;omega=arg(C)/20. Relative-CFO principal interval+-625kHz, modulo1.25MHz. Coherence1e-6/energy1e-12 fallback:no correction;no affine claim for degenerate or branch-crossing cases.', '2cores:equivariant_memory and gauge_coherent;sample amplitudes preserved, nonlinearphase differences up to an affine coordinate change;zero newlearned parameters. Total202553/164225. No rawbypass around synchronizer.', 'Not TX oscillator/PA/IQ recovery or universal RX/channel invariance;received CFO maycarryidentity information and removal can hurt, so legal sourceperformance decides.', '8new scratch xE200x50=80000CEupdates/1600epochs;L6300/V27000;U56700 unused;batch128;AdamW2e-4/wd1e-4/cosine1e-6;no clipping/extra loss/domain/teacher/EMA/checkpoint inheritance.', 'All newmodels fullFP32:cuDNNTF32False,matmulFalse/benchmarkFalse/deterministicFalse/highest. Original residual control retains historical precision;comparison ismethod performance,not single-factor precision-causal attribution.', 'Exactly12 source records:8 new+4 immutable original residual controls;fourseed max0.5V+0.5worstRX, exact ties V,worstRX,MAC,parameters,fixedcandidateorder. No costwindow or physicsreport threshold selection.', 'Complete stepJSONL/detailedtext/compactepochJSONLCSV;actualsynchronizerstatus/CFO/coherence/fallback/gradient/sourceV/resource. FullV90TXRXday group synchronization estimates remain relative,not truehardware labels.', 'Frozen existing public5TXx6RX chain plus boundedphase/CFOaudit;TX/RX image/cubic confounds retained. Mathematical and numericscope reported;no synthetic training/no target/no exactWiSig equalizer.', 'Selectednewonly4clean predictions+20old fixed controls=24 truthlast scored rows. Baselinewinner retains verifiedhistoricaltest;unselectedmodels get no query. No target feedback/selective rerun;D92/LEO/SFTnewclassesN/A.']
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
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_synchronized_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_synchronized_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',synchronized=synchronized_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_whole_phase_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Model/loader RNG varies only; fixedphysicalsplit; fixedsyntheticdiagnostic noRNG, noaugmentation/support/targetrandomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_synchronized_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_synchronized_identity/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_synchronized_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_synchronized_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))


if __name__=='__main__':main()
