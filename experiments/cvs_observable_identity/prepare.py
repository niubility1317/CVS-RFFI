"""Preregister paired source-only hypotheses and a frozen clean completion plan."""
import copy,json
from pathlib import Path
from experiments.cvs_observable_identity.model import VARIANTS,build,observable_contract
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-observable-identity-manysig-m8-r01'
RELEASE='cvs_observable_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-observable-clean-manysig-m24-r01'
PREDECESSOR='20261001-phase1-cvs-residual-identity-manysig-m8-r01'

def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-observable-identity-ce',display_name='CVS继续研发：整体物理观测身份表征',
        description='两个源域前瞻相位/仿射相位不变表征假设各四seed从零训练；相同source物理划分和200x50步；仅CE/身份骨干/无增强。性能优先源选定一个候选后默认clean测试。',
        authorization='用户要求继续优化CVS，性能优先；2026-10-02明确要求真正的RFF physics aware。沿用CE唯一、身份骨干、无增强、clean-only，每次发布默认最后测试。',
        parent_run_ids=[PREDECESSOR],status='PLANNED',rows=[],tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','whole_identity_observables','source_selection'])
    spec['checkpoint'].update(sources=[],selection_rule='Every row scratch fixed E200. Two candidates chosen by four-seed source V/worst RX only; clean scores never consumed. Parent source is an analysis reference, no inherited state.')
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='dd02ce1862b6eafcf7e0d1717bbe48be8d0975f8')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-observable-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[])
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_observable_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training processes/GPU and >=12GB free; preserve all other processes')
    spec['metrics_plan'].update(primary='Full 200-epoch source CE/V/worstRX; select via performance first: maximum0.5 mean V+0.5 mean worstRX;costs only after exactly equal performance',
        later_test='Automatically register '+CONFIRM_RUN+' after source-only freeze: selected4 new clean predictions, read-only reuse old20 fixed predictions; independently score all24; no target feedback')
    spec['notes']=[
        'Prospective hypothesis docs/CVS_OBSERVABLE_IDENTITY_HYPOTHESIS_20261002.md;based on physical equations and all33300sourceL/V observable statistics atdd02ce186;no historical or futuretargetscore input.',
        'Allidentitypaths share fixed physicalfrontend:no rawIQ bypass.A/logpower differences;lags1/2/5/20;regularizedincrements/closures eps1e-6;phase21channels/affine13channels.',
        'Wholemodelglobalphaseinvariant;affine variant also exp(j(theta+omega*t)) invariant onprovidedIQ. No universalchannel/RX invariance or TXparameteridentification.',
        'Threepaths:complete256time48channels;source-supported80:1604x20grid32/48Conv2d;FFTvariationofrealobservables32/48Conv1d. All160embeddings,perpacketGN/LN,boundedadditivefusion,ordinarycosine30.',
        'Pairedseedcomplete21columninitializationthenaffine13columnspruned;commoncolumns anddownstreamweightsidentical;no checkpointinheritance.',
        'Only ordinaryCE;scratchE200x50,batch128,FP32/noclip,AdamW2e-4/wd1e-4/cosine1e-6;fixedsourcephysicalsplit;Uunused;no augmentation/domain/teacher/EMA/margin.',
        'Fullsourcecurves plus finalTXRXday/embeddingdispersion report;sourceV never updates model/normalization;diagnostic aggregates not selectioninput.',
        'Performancefirstfour-seedsourceV/worstRXscore;exactties thencosts. Selectonecandidate;default4newclean+old20fixedpredictions;independenttruthlast;no targetfeedback or unselectedquery.',
        'Conv1d/Conv2d/matrixMACs;FFT/physicalelementwise excluded from MACs but included actual elapsed/peak;costsecondary to performance.',
        'Exposedcleanclosedsetproxybenchmark,notnewblind/LEO/support/novelTXtest;allnegative outcomes retained;preserveotherhealthyprocesses.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,views=['clean'],query_count=168000,registered_classes=6,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',previous_cvs_run='20261001-phase1-cvs-selected-clean-manysig-m20-r01',
        new_prediction_rows=4,reused_rows=20,total_rows=24,selection='source_only',target_feedback=False)
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv','step_metrics.jsonl','source_final_diagnostics.json','source_physical_diagnostics.json','source_selection.json','later independently scored clean confirmation']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_observable_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            c=dict(method='cvs_observable_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],
                output_root=out,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',observables=observable_contract(variant))
            write(ROOT/ref,c)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_whole_observable_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Only model/loader RNG varies; fixed physical source split; no augmentation/support/evaluation randomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='200x50 updates;6300L/epoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_observable_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_observable_identity/configs/experiment_spec.json',spec);write(ROOT/'experiments/cvs_observable_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_observable_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,rows=len(runtime['rows']))))

if __name__=='__main__':main()
