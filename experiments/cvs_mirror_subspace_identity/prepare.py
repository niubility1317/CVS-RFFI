"""Preregister two spectral temporal relation architectures, four own-scratch seeds."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_mirror_subspace_identity.model import build,VARIANTS,relation_contract
from experiments.cvs_mirror_subspace_identity.dispatch import control_rows,CANDIDATES,CONTROL_RUN,ANCHOR_CONTROL_RUN,SPECTRAL_CONTROL_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-phase1-cvs-mirror-subspace-identity-manysig-m8-r01'
RELEASE='cvs_mirror_subspace_identity_20261003_r01'
CONFIRM_RUN='20261003-phase1-cvs-mirror-subspace-clean-manysig-m52-r01'
OLD_CLEAN_RUN='20261003-phase1-cvs-spectral-relation-clean-manysig-m48-r01'

def main():
    configs=ROOT/'experiments/cvs_mirror_subspace_identity/configs'
    if configs.exists():raise FileExistsError('Preserve existing spectral relation registration/configs')
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-mirror-subspace-identity-ce',
        display_name='CVS镜像频对关系：能量与子空间投影×四seed',
        description='保留完整Shallow主干，将接收机IQ混合及理想逐频信道写为镜像频对左作用；同参数比较能量关系与有界子空间投影。',
        authorization='2026-10-03用户授权源域架构优化实验；保持原单CE、原数据与训练策略。',
        status='PLANNED',rows=[],parent_run_ids=[CONTROL_RUN,ANCHOR_CONTROL_RUN,SPECTRAL_CONTROL_RUN],
        tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','mirror_relation','source_selection'])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        architecture_base_commit='f929f4d1116640474d324844e2f0824f97ca7dd0')
    spec['permissions']['claim_scope']='Architecture-only mirror-pair subspace branch; qualified invertible mirror mixing property is not general finite-window channel removal or arbitraryRX invariance. Historically exposed benchmark; no target-derived design/ranking, hardware recovery, or first-blind-test claim'
    spec['checkpoint'].update(initialization='scratch_only',sources=[],
        selection_rule='Own scratch fixedE200;max four-seed mean(0.5V+0.5worstsourceRX). Shallow, Anchor and source winner frequency-relation controls are source metadata only; no inherited weights or target scores.')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-mirror-subspace-identity-20261003',
        runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[],
        source_controls=control_rows(),numerical_policy=FULL_FP32_POLICY)
    spec['source_controls']=control_rows()
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],
        remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_mirror_subspace_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Available capacity only;at most two total training jobs/GPU;>=12GB free;no healthy task interference')
    spec['metrics_plan'].update(primary='CompleteE200/10000steps CE,V/worstRX,fixedsource ranking;actual relation gradients/output/rank2 norms/energy and determinant floor fractions;90sourceTXRXday cells and resources',
        later_test='Only source-selected new variant:4new predictions+48immutable controls;52rows clean. If control retained,reuse its already completed test.')
    spec['notes']=[
        'Architecture only; original single TX CE, original data/optimizer/FP32 and E200; no augmentation, new loss, teacher, EMA, weighting, sampling, staged training or adaptation.',
        'Own scratch Shallow; Shallow/Anchor/frequency-relation are source metadata controls only. Design/ranking never uses target scores.',
        'Real periodic Hann64/hop32/7frames; 31 nonselfconjugate mirror pairs. Apply shared learned complex4x7 temporal matrix to both rows after negative-frequency conjugation.',
        'mirror_energy Q=N^HN; mirror_subspace Q=(alpha/2) N^H adj(NN^H) N / max(det(NN^H),1e-3). Energy denominator max(pairenergy,meanpairenergy/64,1e-6); fixed determinant floor, no sweep.',
        'Qualified invertible left-mixing invariance requires both energy and determinant floors inactive before and after transformation. Rank-one subspace output tends to zero; TX linear IQ distinctions can also be removed.',
        '32 real relation channels x31 pairs; same two Conv1d32k3GELU, pool4, biasfree Linear128to160 zero exit, added to original frequency projection before stats/fusion. All original paths retained; initial function exactly own scratch Shallow.',
        'Both total247731/new26744/base220987. No cross-packet state or RX/TX/role identifiers at inference. No whole-model or arbitrary-channel/RX invariance claim.',
        'L6300,U56700unused,V27000,RX1/3/4/6/8,days1/2/3,split392005,seeds2026092701to04,batch128/dropfalse,E200x50,AdamW2e-4/wd1e-4/cosine1e-6,FP32/TF32false.',
        'Twenty source records=8new+4Shallow+4Anchor+4frequency-relation. Fixed performance-first four-seed score .5V+.5worstRX; exact ties V then worstRX then cost. No bestepoch/earlystop/target rerank/selective rerun.',
        'Historically exposed benchmark, not first-blind confirmation. Only new source winner gets4new clean predictions+48 immutable controls=52rows/416ALL-RX scores. Old winner reuses its frozen test; unselected candidates get no query. NoLEO/support/SFT/newclasses.',
        'Full step/epoch/compactJSONL/CSV and27000sourceVpacket scalars/90cells include energy floor, determinant floor, alpha, trace and norm. Public algebra/IQ/FIR diagnostic is forward-only, not augmentation or selection.',
        'Design docs/CVS_MIRROR_SUBSPACE_20261003.md; qualified row-space projection is established algebra, not claimed invented. No first or publishability claim before adequate evidence.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,condition='Only new source winner; otherwise reuse selected fixed control test with zero newquery',
        views=['clean'],query_count=168000,registered_classes=6,new_prediction_rows=4,reused_rows=48,total_rows=52,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        retained_baseline_test_run=OLD_CLEAN_RUN,baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',
        candidate_universe=list(CANDIDATES),target_tx_ids=['14-10','14-7','20-15','20-19','6-15','8-20'],
        target_rx_ids=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],target_feedback=False,selection='source_only')
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_compact.jsonl','epoch_metrics.csv','source_final_diagnostics.json',
        'source_physical_diagnostics.json','source_mirror_relation_scalars.npz','source_selection.json','conditional independent clean test']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/cvs_mirror_subspace_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            cfg=dict(method='cvs_mirror_subspace_identity',variant=variant,model_seed=seed,
                source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=out,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,
                augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                split_seed=392005,device='cuda:0',mirror_relation=relation_contract(variant),numerical_policy=FULL_FP32_POLICY)
            write(ROOT/ref,cfg)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_mirror_relation_candidate',gpu=None,
                config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Onlymodel/loaderRNGvaries;fixedphysicalsplit;noaugmentation/support/evaluationRNG',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50;6300LperEpoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_mirror_subspace_identity.source --config '+remote,
                expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    for name,data in [('experiment_spec.json',spec),('launch_spec.json',runtime),
                      ('local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})]:write(configs/name,data)
    print(json.dumps(dict(run_id=RUN,new_rows=8,source_control_rows=12,candidate_universe=list(CANDIDATES))))

if __name__=='__main__':main()
