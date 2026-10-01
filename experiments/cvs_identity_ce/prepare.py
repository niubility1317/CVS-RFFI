"""Write the user-requested fixed five-seed identity-only CE preregistration."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=Path('E:/type10-7')
RUN='20261001-phase1-cvs-identity-ce-practical-manysig-m5-r02'
RELEASE='cvs_identity_ce_20261001_r02'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
SEEDS=[392005,2026092701,2026092702,2026092703,2026092704]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    cfgroot=ROOT/'experiments/cvs_identity_ce/configs'
    old=json.loads((WORKSPACE/'automation_reports/CV-SincNet/20260927-phase1-baselines-practical-manysig-m5-r01/experiment.json').read_text(encoding='utf-8'))
    spec=copy.deepcopy(old)
    spec.update(run_id=RUN,group_id='cvs-identity-backbone-ce-vs-cvcnn-matched',
        display_name='CVS身份骨干：仅交叉熵与CVCNN同条件对照',
        description='检验CVS lite_d身份网络在仅CE、同物理划分、同星地增强及同优化预算下是否优于CVCNN-CE。域骨干不实例化，关闭DAOT/FastTrust/PL/MixStyle及所有附加损失；原生cosine分类头不传标签，不启用margin。',
        kind='cvs',stage='Phase1',aliases=[],tags=['cvs_identity_ce','identity_only','cvcnn_matched','ablation','residual_noeq','scratch','final200'],
        authorization='2026-10-01用户：CVS基础网络只使用交叉熵和同样星地信道增强、同样数据划分；跑实验说明；域骨干不启用，只使用身份骨干。',
        parent_run_ids=[],replaces_run_id='20261001-phase1-cvs-identity-ce-practical-manysig-m5-r01',status='PLANNED',rows=[])
    spec['code']=dict(commit='release_commit.txt resolves the committed runtime version',checkout=str(ROOT),
        environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=PROJECT+'/releases/'+RELEASE,
        architecture_base_commit='691a03c7c6559714b3222e3ebce0364e3955302c')
    spec['checkpoint']=dict(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',
        contract_check_ref=old['data']['contract_ref'],selection_rule='fixed last.pt epoch200; no source/target ranking or early stop')
    spec['permissions']=dict(regime='source_only_CVS',query_use='frozen Phase1 inference only; separate truth-last scorer',
        external_method_exception=None,claim_scope='Architecture plus native classifier comparison under matched CE/augmentation/optimization; no target-driven tuning, no Phase2 inference or adaptation. Previously exposed target benchmark, not a fresh blind confirmation.')
    spec['execution']=dict(host='N607',launch_owner='codex/root/cvs-identity-ce-20261001',
        gpu_policy='five separate legal GPU slots, one row/GPU preferred, never more than2 total training jobs/GPU',
        remote_run_root=PROJECT+'/runs/'+RUN,remote_log_root=PROJECT+'/logs/'+RUN,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_identity_ce.publish --output local_artifacts/cvs_identity_ce_20261001_r02',
        stop_rule='Nonfinite loss/gradient, source or target permission violation, output collision or nonzero exit fails only affected row; no low-performance stop, automatic retry, fallback or unrelated intervention.')
    spec['expected_artifacts']=['launch.json','dispatcher.json','each source/initial_smoke.pt','each source/initialization.json',
        'each source/source_contract.json','each source/resolved_config.json','each source/step_metrics.jsonl',
        'each source/epoch_metrics.jsonl','each source/epoch_metrics.csv','each source/last.pt','each source/completion.json',
        'each prediction/phase1_predictions.npz','each prediction/phase1_complete.json','phase1_scored_results.json',
        'phase1_scored_results.csv','phase1_summary.json','scoring_p1_complete.json','completion.json']
    spec['metrics_plan']=dict(metric_names=['accuracy','macro_f1','source_V_accuracy','clean_ce','satellite_ce',
        'gradient_norm','optimizer_steps','parameter_count','peak_cuda_allocated_bytes','wall_time'],
        dimensions=['model_seed','view','receiver'],
        prediction_ref=PROJECT+'/runs/'+RUN+'/<row>/prediction/phase1_predictions.npz',
        scorer_ref='comparison_suite.score:p1 (all five predictions fixed before truth)',
        primary='four matched seeds2026092701..04 mean and sample SD; seed392005 separate; paired seed differences vs CVCNN-CE',
        full_cvs_reference='DAOT+RC4 results descriptive only; its U_s use and optimization-step budget differ')
    spec['notes']=['CE trains L_s only; U_s split remains identical and unused. L/U/V=6300/56700/27000.',
        'Same CVCNN AdamW lr2e-4 wd1e-4 cosine eta_min1e-6, FP32, no gradient clipping, batch128, no drop_last, 200x50=10000 steps.',
        'Enhancement uses the existing practical residual/post_sync/noeq adapter, seeds2027; only E80 onward clean+sat concat, sat CE weight.68. Effective E80-90 mid/urban p.60, E91-200 three scenes p.80. No satellite forward before E80.',
        'Same physical role IDs and deterministic augmentation draw for the same ID/epoch; batch order and initialization differ across architectures.',
        'CVS M/lite_d identity backbone branch_ablation=no_dac, raw_fft/raw_iq native branches and native cosine head scale30; label-dependent margin disabled by y=None. Domain backbone, adversaries, pseudo labels, EMA, MixStyle, DAOT and all auxiliary objectives absent.',
        'Classifier and capacity are part of architecture comparison, not an isolated convolution-only causal test. No new second arm or extra matrix.',
        'Source contract and existing VALIDATED_ONCE Phase1 capsule reused without rebuilding. Target results cannot select weights/seeds or trigger tuning/retraining.',
        'data seed null: deterministic fixed physical allocation; loader torch stream follows model seed. support seed null: no support used. evaluation null: immutable existing paired capsule; index fixes received scenes and observations.',
        'Native unused diagnostic projection/head parameters may have no CE gradient; report total requires_grad count and retain absence of their losses.']
    runtime=dict(run_id=RUN,launch_owner=spec['execution']['launch_owner'],
        runtime_root=spec['execution']['remote_run_root'],log_root=spec['execution']['remote_log_root'],rows=[],
        p1_truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    for seed in SEEDS:
        rowid='cvs-identity-ce-s'+str(seed)
        source=runtime['runtime_root']+'/'+rowid+'/source'; prediction=runtime['runtime_root']+'/'+rowid+'/prediction'
        config=dict(method='cvs_identity_ce',model_seed=seed,source_contract=old['data']['contract_ref'],
            dataset=old['data']['dataset'],output_root=source,device='cuda:0',epochs=200,batch_size=128,
            lr=.0002,lr_min=.000001,weight_decay=.0001,augmentation_seed=2027,receiver_seed=2027,
            split_seed=392005,drop_last=False,sat_ce_start=80,lambda_sat_cls=.68,domain_backbone=False,
            mixstyle=False,extra_losses=[],selection='fixed_last_epoch',branch_ablation='no_dac')
        pred=dict(method='cvs_identity_ce',model_seed=seed,source_output=source,output_root=prediction,
            device='cuda:0',source_contract=old['data']['contract_ref'],
            p1_capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule')
        srcref='experiments/cvs_identity_ce/configs/source-s'+str(seed)+'.json'
        predref='experiments/cvs_identity_ce/configs/predict-s'+str(seed)+'.json'
        write(ROOT/srcref,config);write(ROOT/predref,pred)
        srcpath=PROJECT+'/releases/'+RELEASE+'/'+srcref; predpath=PROJECT+'/releases/'+RELEASE+'/'+predref
        command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_identity_ce.dispatch --source-config '+srcpath+' --predict-config '+predpath
        spec['rows'].append(dict(row_id=rowid,method='cvs_identity_ce',purpose='historical_reference' if seed==392005 else 'matched_model_seed',
            gpu=None,config_ref=srcref,prediction_config_ref=predref,resolved_config_ref=source+'/resolved_config.json',
            data_overrides={},seeds=dict(model=seed,split=392005,data=None,augmentation=2027,support=None,evaluation=None),
            seed_notes='Fixed contract IDs; loader follows model RNG; receiver hardware seed2027; no support; immutable evaluation capsule.',
            k=None,scenario='clean,practical_high,practical_mid,practical_low_urban',optimizer='AdamW+cosine',lr=.0002,
            epochs=200,fl_rounds=None,budget_ref='200x50 optimizer steps; 6300 L_s rows/epoch; satellite concat E80..200',
            output_root=source,prediction_root=prediction,log_path=runtime['log_root']+'/'+rowid+'.log',command=command,
            expected_artifacts=['last.pt','completion.json','epoch_metrics.jsonl','epoch_metrics.csv']))
        runtime['rows'].append(dict(row_id=rowid,method='cvs_identity_ce',model_seed=seed,stage='phase12',
            source_config=srcpath,config=predpath,output_root=prediction))
    write(ROOT/'experiments/cvs_identity_ce/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_identity_ce/configs/experiment_spec.json',spec)
    print(RUN)


if __name__=='__main__':main()
