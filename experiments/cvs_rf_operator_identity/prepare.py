"""Preregister paired source-only hypotheses and a frozen clean completion plan."""
import copy,json
from pathlib import Path
from experiments.cvs_rf_operator_identity.model import VARIANTS,build,operator_contract
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-rf-operator-identity-manysig-m8-r01'
RELEASE='cvs_rf_operator_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01'
PREDECESSOR='20261001-phase1-cvs-residual-identity-manysig-m8-r01'

def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-rf-operator-identity-ce',display_name='CVS继续研发：受约束复数射频行为算子',
        description='两个源域前瞻MP/GMP行为算子假设各四seed从零训练；相同source物理划分和200x50步；仅CE/身份骨干/无增强。性能优先源选定一个候选后默认clean测试。',
        authorization='用户要求继续优化CVS，性能优先；2026-10-02明确要求真正的RFF physics aware。沿用CE唯一、身份骨干、无增强、clean-only，每次发布默认最后测试。',
        parent_run_ids=[PREDECESSOR],status='PLANNED',rows=[],tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','rf_behavior_operator','source_selection'])
    spec['checkpoint'].update(sources=[],selection_rule='Every row scratch fixed E200. Two candidates chosen by four-seed source V/worst RX only; clean scores never consumed. Parent source is an analysis reference, no inherited state.')
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='bae489879323c7c36d12f8678917d1f8ca917104')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-rf-operator-identity-20261002',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[])
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_rf_operator_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training processes/GPU and >=12GB free; preserve all other processes')
    spec['metrics_plan'].update(primary='Full 200-epoch source CE/V/worstRX; select via performance first: maximum0.5 mean V+0.5 mean worstRX;costs only after exactly equal performance',
        later_test='Automatically register '+CONFIRM_RUN+' after source-only freeze: selected4 new clean predictions, read-only reuse old20 fixed predictions; independently score all24; no target feedback')
    spec['notes']=[
        'Hypothesis fixed in docs/CVS_RF_OPERATOR_HYPOTHESIS_20261002.md at bae489879 before simplex clean release. Design uses current input pipeline and RF equations, not any target score.',
        'Preserve residual_fusion time/frequency/mean pools/dense fusion/ordinary learned cosine30 classifier;replace only PA lift/gate and first convolution input shape. No phase supplement, DSQ or simplex head.',
        'Causal odd orders1/3/5, aligned delays0..3;GMP additionally envelope lags1/2 for orders3/5. Eight complex direct and conjugate filters,radial clip4,eps1e-6,order scale2^(p-1),32 invariant observation channels. No complex biases or real mixing before charge-zero observations.',
        'All weights scratch. Both variants draw full28-term RF bank and retain12/28;common terms and all shared tensors initialized identically for same seed;GMP adds512 coefficients,not extra CNN depth/width.',
        'Operator represents truncated received-IQ behavior; coefficients are shared source-learned filters,not identified TX PA/IQ parameters. Global-phase invariance holds for PA observations only;no full-network,CFO,multipath,RX invariance or TX/RX identifiability claim.',
        'Formal training only ordinary CE,200x50 updates,batch128,FP32/no clip,AdamW2e-4/wd1e-4/cosine1e-6,fixed physical split,U_s unused,no augmentation/domain/teacher/EMA/resume. Synthetic physics tests are separate correctness checks,not training augmentation.',
        'Performance first four-seed E200 mean sourceV/worstRX score,exact performance ties then costs;never target-based selection. Select one candidate,default selected4 clean predictions plus immutable old20 controls,all24 fixed before independent truth scoring.',
        'Report full source curves/logs and RF coefficient/gradient diagnostics;real cost/latency/memory measured on disposable clone. RF contractions included in Conv MACs;radial protection,powers,observations excluded from MACs but included in elapsed/peak.',
        'Existing exposed clean closed-set benchmark only;no blind/LEO/support/novel class claims;retain all negative outcomes;one launch owner;preserve other healthy runs.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,views=['clean'],query_count=168000,registered_classes=6,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',previous_cvs_run='20261001-phase1-cvs-selected-clean-manysig-m20-r01',
        new_prediction_rows=4,reused_rows=20,total_rows=24,selection='source_only',target_feedback=False)
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv','step_metrics.jsonl','source_selection.json','later independently scored clean confirmation']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_rf_operator_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            c=dict(method='cvs_rf_operator_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],
                output_root=out,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0',rf_operator=operator_contract(variant))
            write(ROOT/ref,c)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_RF_behavior_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Only model/loader RNG varies; fixed physical source split; no augmentation/support/evaluation randomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='200x50 updates;6300L/epoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_rf_operator_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_rf_operator_identity/configs/experiment_spec.json',spec);write(ROOT/'experiments/cvs_rf_operator_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_rf_operator_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,rows=len(runtime['rows']))))

if __name__=='__main__':main()
