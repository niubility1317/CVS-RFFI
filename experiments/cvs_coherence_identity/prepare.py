"""Preregister paired source-only hypotheses and a frozen clean completion plan."""
import copy,json
from pathlib import Path
from experiments.cvs_coherence_identity.model import VARIANTS,build
from experiments.cvs_residual_identity.prepare import PROJECT,SEEDS,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261001-phase1-cvs-coherence-identity-manysig-m8-r01'
RELEASE='cvs_coherence_identity_20261001_r01'
CONFIRM_RUN='20261001-phase1-cvs-coherence-clean-manysig-m24-r01'
PREDECESSOR='20261001-phase1-cvs-stability-identity-manysig-m8-r01'

def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-coherence-identity-ce',display_name='CVS继续研发：连续复相关增量身份线索',
        description='两个连续复相关表征假设各四seed从零训练；相同source物理划分和200x50步；仅CE/身份骨干/无增强。性能优先源选定一个候选后默认clean测试。',
        authorization='2026-10-01用户要求根据原目标继续改进，并要求每次发布最后默认测试集测试。',
        parent_run_ids=[PREDECESSOR],status='PLANNED',rows=[],tags=['cvs','ce_only','clean_only','no_augmentation','performance_priority','complex_coherence','source_selection'])
    spec['checkpoint'].update(sources=[],selection_rule='Every row scratch fixed E200. Two candidates chosen by four-seed source V/worst RX only; clean scores never consumed. Parent source is an analysis reference, no inherited state.')
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='7dcafe9fef0febd1c54e40c9a704ef0c52a178a9')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-coherence-identity-20261001',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[])
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_coherence_identity.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Only free capacity; at most two total training processes/GPU and >=12GB free; preserve all other processes')
    spec['metrics_plan'].update(primary='Full 200-epoch source CE/V/worstRX; select via performance first: maximum0.5 mean V+0.5 mean worstRX;costs only after exactly equal performance',
        later_test='Automatically register '+CONFIRM_RUN+' after source-only freeze: selected4 new clean predictions, read-only reuse old20 fixed predictions; independently score all24; no target feedback')
    spec['notes']=[
        'Only full E200 source curves from prior source research and algebraic branch interfaces inform this work. No target score, TX/RX errors, prediction or truth is consumed by training/selection. Performance is primary; parameter cost secondary.',
        'Preserve residual CVS native extractors, average pools, dense fusion and PA cosine head. Candidate1 uses regularized complex increments with the same native8-channel projection; candidate2 additionally uses the unchanged native4-channel DSQ stem. No latent width/depth or parameter increase versus paired stability variants. Original nativeepsilon1e-6 retained, no epsilon sweep.',
        'Supplement packet-local normalized envelope, envelope difference and real/imaginary parts of regularized neighbor complex coherence, optionally local spectral residual. Supplemental cue is phase invariant and attenuates weak pairs;full raw network is not invariant. No claimed CFO removal or causal TX-only identification. No receiver/role inputs, domain gate or batch/query statistics.',
        'No global IQ phase canonicalization, receiver label input, target batch statistics, domain backbone, extra loss, data/channel augmentation or increased feature-extractor depth/width; only the registered deterministic supplemental feature stems and their required input columns differ.',
        'All scratch200x50, batch128, AdamW2e-4/wd1e-4/cosine1e-6, FP32/no clipping; dedicated matched loader RNG and original four seeds. U_s unused.',
        'Test data same validated capsule clean.npy/index.npz/manifest,168000 queries,7RX,6classes. Truth only after all predictions fixed. Clean-only Phase1 closed-set, no LE0/new-class/support claims.',
        'Compare selected candidate to original native/fixed common networks and frozen previous residual_fusion on identical physical queries/seeds; keep all negative gains. Report parameter/MAC/state/time/actual memory and full RX/TX/F1 tables.',
        'Existing target is historically exposed repeated benchmark; no novel blind/general satellite claim. No default LEO tests because explicit clean-only scope.',
        'One launch owner, immutable source and confirmation roots; health/lowperformance never stop or restart jobs; no ancestor/resume/teacher/EMA.'
    ]
    spec['test_completion_plan']=dict(run_id=CONFIRM_RUN,views=['clean'],query_count=168000,registered_classes=6,
        capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json',
        baseline_run='20261001-phase1-clean-baselines-manysig-m16-r01',previous_cvs_run='20261001-phase1-cvs-selected-clean-manysig-m20-r01',
        new_prediction_rows=4,reused_rows=20,total_rows=24,selection='source_only',target_feedback=False)
    spec['expected_artifacts']=['last.pt','initialization.json','source_contract.json','resolved_config.json','completion.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv','step_metrics.jsonl','source_selection.json','later independently scored clean confirmation']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/source';ref='experiments/cvs_coherence_identity/configs/'+rid+'.json';remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            c=dict(method='cvs_coherence_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],
                output_root=out,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0')
            write(ROOT/ref,c)
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_physical_cue_candidate',gpu=None,config_ref=ref,resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),seed_notes='Only model/loader RNG varies; fixed physical source split; no augmentation/support/evaluation randomness',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='200x50 updates;6300L/epoch',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_coherence_identity.source --config '+remote,expected_artifacts=spec['expected_artifacts'][:-2]))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=out))
    write(ROOT/'experiments/cvs_coherence_identity/configs/experiment_spec.json',spec);write(ROOT/'experiments/cvs_coherence_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_coherence_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,rows=len(runtime['rows']))))

if __name__=='__main__':main()
