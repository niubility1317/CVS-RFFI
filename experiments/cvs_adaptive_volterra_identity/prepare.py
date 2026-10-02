"""Two phase-memory lags, four seeds, current source winner metadata only."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_adaptive_volterra_identity.model import VARIANTS,build,adaptive_contract
from experiments.cvs_adaptive_volterra_identity.dispatch import control_rows,CANDIDATES
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-adaptive-volterra-identity-manysig-m8-r01'
RELEASE='cvs_adaptive_volterra_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-adaptive-volterra-clean-manysig-m32-r01'


def main():
    folder=ROOT/'experiments/cvs_adaptive_volterra_identity/configs'
    if folder.exists():raise FileExistsError('Preserve registered Volterra configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-adaptive-delay-balanced-volterra-identity-ce',
        display_name='CVS 可学习相位记忆残差：两零初始化系数，lag1/4×四seed纯CE',
        description='保留lag4包络控制初始化，以两个全局tanh系数学习复三阶及五阶残差；新增2参数，深宽/读出及物理划分固定，源选后仅clean。',
        authorization='用户持续授权优化基础神经网络：性能第一，数学/通信/物理/RFF原理；CE唯一、无信道增强、clean-only，默认测试收尾。',
        status='PLANNED',rows=[],parent_run_ids=['20261002-phase1-cvs-coupled-identity-manysig-m8-r01'],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','complex_volterra_memory','two_extra_parameters','source_selection','rff_physics'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),architecture_base_commit='f58c02be4f192b996ed5a3dc4f6fa96e93a7a39a')
    spec['data']['leo_config_ref']=None
    spec['permissions'].update(query_use='No target paths in source training; after fixed source freeze selected4 clean inference then independent truth-last score',
        claim_scope='Received complex Volterra identity input; no fitted PA kernels, uniqueTX or arbitraryRX/channel invariance; not a new blind benchmark')
    spec['checkpoint'].update(sources=[],selection_rule='Own scratch E200; fourseed mean0.5sourceV+0.5worstsourceRX maximum; current coupled_lag4 source metadata only, no inherited weights or target metrics.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-adaptive-volterra-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_adaptive_volterra_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training jobs/GPU and >=12GB free; preserve unrelated jobs')
    spec['metrics_plan'].update(primary='FullE200 sourceCE/V/worstRX, fixedperformance ranking, sourceTXRXday/geometry, actual complex cubic conv-input, sixblock relativeenergy, publicTX->channel->RX, measuredresources',
        later_test='New source-selected model4clean +28originalcontrols in '+CONFIRM_RUN+'; retained current sourcewinner reuses alreadyverified clean, unselected newmodelsN/A')
    spec['notes']=[
        'Source/public design only, not based on historical target scores. Current comparator is coupled_lag4 solely because the fixed source rule selected it.',
        'At t=n-m,p=(P[t]+P[t-4])/8,C=z[t-l]^2conj(z[t-2l])/4;inputs z,zp+tanh(a3)(C-zp),zp^2+tanh(a5)(Cp-zp^2),m0..3,l1or4. Two global gates initialized exactly zero and learned only by CE; no hyperparameter coefficient scan.',
        'Same12complex inputs, original sixcomplex blocks/positionpower readout/depth/width/160embedding/cosine30head and202555parameters(+2 vs202553control); same-seed shared scratch state and initial logits exactly match current coupled control; new gates are zero, no checkpoint load.',
        'Affine phase input covariance follows l+l-2l=0; all12 slots match z[n-m]. Entire network claims constantphase property only; native finite convolution and FFT still react to frequency. Raw received input/alpha0 exactly.',
        'Restricted third/fifth complex Volterra-inspired identity basis, not completeVolterra, DPD, uniquely identified TX or PA coefficients. RX/channel and unknownexcitation can yield the same received waveform.',
        'Lags1/4 at25MHz are40ns/160ns, largest new cubic delay2l=80ns/320ns; design intervals, not measured hardware timeconstants. All causalpadding, no circularhistory.',
        'CE only,scratch,8xE200x50=80000updates,1600epochs,L6300/V27000/U56700unused;split392005/batch128/AdamW2e-4/wd1e-4/cosine1e-6;no extra loss,augmentation,domain,clipping/teacher/EMA.',
        'FullFP32:cuDNN/matmulTF32False,benchmarkFalse,deterministicFalse,highest. Detailedtext+allstepJSONL+compactepochJSONL/CSV,actual12term conv input/all6 normalization andallstep two gate raw/coefficient/gradient measurements. Original dataVALIDATED_ONCE unchanged.',
        '12source records=8new+4currentcoupled source metadata; highest fourseed mean0.5V+0.5worstRX; exacttiesV/worstRX thenMAC/parameters/fixedorder. Physical tests are direct correctness, public numeric tolerance is reporting only.',
        'If source-selected new:4newclean+28originalfrozencontrols=32rows/256scoredALL+RXrecords,168000physicalID/6TX/7RX; all predictions fixed before independenttruthlast. No unselectedquery,LEO,SFT/support/newclass or testfeedback.',
        'Conv/Linear MAC excludes fixed new complex products andelementwise operations; actual latency,memory/statecosts measured, no constanttotalcompute claim.']
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='new_candidate_selected;otherwise retain current coupled sourcewinner original clean without rerun',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=28,total_rows=32,
        scored_records=256,capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run='20261002-phase1-cvs-coupled-clean-manysig-m28-r01',baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',
        candidate_universe=list(CANDIDATES),target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],
        target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json','source_physical_diagnostics.json','source_selection.json','conditional independent clean test']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_adaptive_volterra_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_adaptive_volterra_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',adaptive=adaptive_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_delay_balanced_complex_Volterra',gpu=None,
                config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Only model/loader RNG varies; fixed physicalsplit; publicdiagnostics fixed; no augmentation/support/targetrandomness',k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_adaptive_volterra_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(folder/'experiment_spec.json',spec);write(folder/'launch_spec.json',runtime)
    write(folder/'local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=4,candidate_universe=list(CANDIDATES))))


if __name__=='__main__':main()
