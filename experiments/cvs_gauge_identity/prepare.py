"""Preregister paired source-only hypotheses and a frozen clean completion plan."""
import copy,json
from pathlib import Path
from experiments.cvs_gauge_identity.model import VARIANTS,build,gauge_contract
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
from experiments.cvs_gauge_identity.dispatch import control_rows,CANDIDATES
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-gauge-identity-manysig-m8-r01'
RELEASE='cvs_gauge_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-gauge-clean-manysig-m24-r01'
PREDECESSOR='20261001-phase1-cvs-residual-identity-manysig-m8-r01'

def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-gauge-identity-ce',display_name='CVS物理研发：保留完整IQ的逐包相位规范化',
        description='两种固定相位参考各四seed从零训练，保留原残差身份网络和全部256点IQ，仅消除公共相位；源域性能与四个历史残差源指标对照，选定后默认clean测试。',
        authorization='用户要求继续优化CVS，性能优先；2026-10-02明确要求真正的RFF physics aware。沿用CE唯一、身份骨干、无增强、clean-only，每次发布默认最后测试。',
        parent_run_ids=[PREDECESSOR],status='PLANNED',rows=[],tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','whole_identity_gauge','source_selection'])
    spec['checkpoint'].update(sources=[],selection_rule='Every new row scratch fixed E200. Rank two new gauges and immutable residual_fusion source control by four-seed source V/worst RX only. No inherited weights/state, no clean-score input.')
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='c315cc239eae8e2a517a4b6e0ea36e3a577e45ae')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-gauge-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=control_rows())
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_gauge_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training processes/GPU and >=12GB free; preserve all other processes')
    spec['metrics_plan'].update(primary='Full 200-epoch source CE/V/worstRX; select via performance first: maximum0.5 mean V+0.5 mean worstRX;costs only after exactly equal performance',
        later_test='If a new gauge is source-selected, automatically register '+CONFIRM_RUN+': selected4 new clean predictions plus read-only old20 fixed predictions, independent truth-last scoring. If residual control wins, retain its verified historical clean artifacts and do not query unselected gauges.')
    spec['notes']=[
        'Prospective hypothesis docs/CVS_GAUGE_IDENTITY_HYPOTHESIS_20261002.md;physical equations plus full33300sourceL/V periodicity evidence;no target-score input to structure, hyperparameters or source ranking.',
        'Fixed phase gauge precedes every identity path;all256 receivedIQ samples multiplied by one packet complex scalar. No raw bypass, sample differencing or waveform substitution.',
        'Both whole models global-phase invariant in real arithmetic;CFO/time-varying phase retained. No arbitrary channel/RX invariance or identified TX parameter claim. Floating point errors/tie boundary caveats reported.',
        'Peak reference: first within1e-5 relative power of packet maximum. Coherent reference:80:160 four20-sample cycles, phase-align only the reference average;degenerate reference fallback to raw peak. eps1e-6 fixed,not tuned.',
        'Native time/Sinc/frequency/received-PA-proxy paths and residual fusion unchanged;160embeddings,ordinarycosine30;164225parameters with0 newlearned frontendparameters. Samecoreinitialization byseed;no inherited checkpoint.',
        'Only ordinaryCE;scratchE200x50,batch128,FP32/noclip,AdamW2e-4/wd1e-4/cosine1e-6;fixedsourcephysicalsplit;Uunused;no augmentation/domain/teacher/EMA/margin.',
        'Fullsourcecurves plus finalTXRXday/embeddingdispersion report;sourceV never updates model/normalization;diagnostic aggregates not selectioninput.',
        'Threearchitecture ranking uses12 source records:8newtraining+4immutablebaseline. Highest0.5meanV+0.5meanworstRX,exact tiesV thenworstRX thenMACs thenparameters,then residual/peak/coherent order. No0.2pp cost band.',
        'Default selectednew4clean+old20;baselinewinner retains priorverifiedtest and marksnewcandidate testN/A. No unselected query,selective re-run or target feedback.',
        'Conv1d/matrixMACs;FFT/gaugeelementwise excluded from MACs but included actual elapsed/peak;costsecondary to performance.',
        'Exposedcleanclosedsetproxybenchmark,notnewblind/LEO/support/novelTXtest;allnegative outcomes retained;preserveotherhealthyprocesses.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,views=['clean'],query_count=168000,registered_classes=6,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',previous_cvs_run='20261001-phase1-cvs-selected-clean-manysig-m20-r01',
        new_prediction_rows=4,reused_rows=20,total_rows=24,selection='source_only',target_feedback=False)
    spec['test_completion_plan'].update(condition='new_candidate_selected;else retain prior residual clean test, new/unselected predictions N/A',
        candidate_universe=list(CANDIDATES),retained_baseline_test_run='20261001-phase1-cvs-selected-clean-manysig-m20-r01',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'])
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv','step_metrics.jsonl','source_final_diagnostics.json','source_physical_diagnostics.json','source_selection.json','later independently scored clean confirmation']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_gauge_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            c=dict(method='cvs_gauge_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],
                output_root=out,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',gauge=gauge_contract(variant))
            write(ROOT/ref,c)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_whole_gauge_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Only model/loader RNG varies; fixed physical source split; no augmentation/support/evaluation randomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='200x50 updates;6300L/epoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_gauge_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_gauge_identity/configs/experiment_spec.json',spec);write(ROOT/'experiments/cvs_gauge_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_gauge_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,rows=len(runtime['rows']))))

if __name__=='__main__':main()
