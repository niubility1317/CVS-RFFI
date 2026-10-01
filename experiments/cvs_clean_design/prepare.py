"""Fixed clean-only architecture research matrix; registration never opens target data."""
import copy
import json
from pathlib import Path
from experiments.cvs_clean_design.model import VARIANTS, BASELINES, build
ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=Path('E:/type10-7')
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261001-phase1-cvs-clean-architecture-manysig-m32-r01'
RELEASE='cvs_clean_architecture_20261001_r01'
SEEDS=(2026092701,2026092702,2026092703,2026092704)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    old=json.loads((WORKSPACE/'automation_reports/CV-SincNet/20261001-phase1-cvs-identity-ce-practical-manysig-m5-r02/experiment.json').read_text(encoding='utf-8'))
    spec=copy.deepcopy(old)
    spec.update(run_id=RUN,group_id='cvs-clean-physics-architecture-ce',display_name='CVS轻量身份网络：纯CE无信道增强源域架构筛选与clean基准',
        description='同物理划分、无数据/信道增强、纯CE的原CVS/CVCNN/real CNN/ResNet1D基线及四个物理结构候选；8架构x4固定seed，仅源V选择一个候选，冻结后仅clean测试。',
        authorization='2026-10-01用户目标：从数学、通信、物理、射频指纹原理优化CVS基础网络；仅CE、不使用星地增强、只测clean、轻量化轻约束；另要求无增强CVCNN等常见网络基准。',
        tags=['cvs','clean_only','architecture','physics','ce_only','no_augmentation','lightweight','cvcnn','resnet1d','real_cnn','source_selection'],
        parent_run_ids=[],replaces_run_id=None,status='PLANNED',rows=[],aliases=[],stage='Phase1-source-architecture-screen')
    spec['checkpoint']=dict(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',contract_check_ref=old['data']['contract_ref'],
        selection_rule='fixed E200 checkpoint for every row; architecture chosen from final source V only across four seeds; no target feedback')
    spec['permissions']=dict(regime='source_only_CVS',query_use='source matrix never opens target; later frozen clean inference and independent truth-last scorer only',
        external_method_exception=None,claim_scope='Fixed clean-only architecture plus native classifier comparison; previously exposed target benchmark disclosed; no clean score driven architecture tuning or selective retraining')
    spec['code']=dict(commit='release_commit.txt resolves runtime commit',checkout=str(ROOT),environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=PROJECT+'/releases/'+RELEASE,
        architecture_base_commit='a00d731ef2881aae926c16797b6b702a8a0e8379')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-clean-architecture-20261001',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,rows=[])
    spec['execution']=dict(host='N607',launch_owner=runtime['launch_owner'],gpu_policy='at most2 total training jobs per GPU;12GB free required before each source launch; respect other owners',
        remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_clean_design.publish --output local_artifacts/cvs_clean_architecture_20261001_r01',
        stop_rule='Only own affected row fails on nonfinite/permissions/output collision/nonzero exit; preserve all artifacts; no retry, low-score stop or unrelated process intervention')
    spec['metrics_plan']=dict(metric_names=['source_V_accuracy','worst_source_RX_accuracy','clean_ce','gradient_norm','parameters','gradient_used_parameters','conv_linear_MACs','FFT_calls','train_ms','inference_ms','peak_cuda_allocated_bytes'],
        dimensions=['architecture','model_seed','source_receiver'],prediction_ref='N/A until source_selection.json is frozen',scorer_ref='N/A: source-only release does not score target',
        primary='four-seed final source score=0.5 accuracy+0.5 worst_RX_accuracy;0.2pp tie favors smaller conv/linear MACs then parameters',
        later_test='four fixed baselines and one selected CVS candidate, four seeds, clean only; no other candidates receive target scores')
    spec['expected_artifacts']=['pipeline_state.json','source_selection.json','each source/initial_smoke.pt','each source/initialization.json','each source/source_contract.json',
        'each source/resolved_config.json','each source/step_metrics.jsonl','each source/epoch_metrics.jsonl','each source/epoch_metrics.csv',
        'each source/last.pt','each source/resource_profile.json','each source/completion.json']
    spec['notes']=['Source L/U/V IDs remain exactly unchanged; only6300 labeled source samples train;56700 unlabeled unused;27000 single V evaluated without state updates.',
        'Every architecture scratch200epochsx50steps, batch128/no drop, AdamW2e-4/wd1e-4/cosine1e-6,FP32/no gradient clipping. Dedicated loader seed fixes batch order across architectures.',
        'No satellite simulator, no noise/phase/mixup augmentation, no PL/EMA/domain backbone or extra loss. Native dropout remains a network layer and is disclosed; CE is the only objective.',
        'Four candidates: orthogonal/radially clipped/RMS-normalized PA lift; moment pooling; their combination; compact shared complex trunk with orthogonal PA and relative-phase readout.',
        'Light constraint: candidate parameters<=1.1x native; full measured MAC/time/peak/state cost reported; no claim that small parameters prove low compute.',
        'Architecture source selection fixed before target. Previously exposed target scores are not used to rank or tune current candidates. No target test until source matrix fully complete.',
        'ResNet1D is a compact6-basic-block project baseline with widths32/64/96, not an author-original ResNet18; all classifier differences disclosed.',
        'Native cosine classifier scale30 no margin retained; common baselines retain standard Linear heads. A gain cannot be assigned to one convolution alone.',
        'data seed null: role allocation fixed by existing source contract; augmentation/support/evaluation seeds null: no augmentation/support; clean evaluation IDs fixed after source selection.']
    counts={v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS}
    write(ROOT/'experiments/cvs_clean_design/configs/local_parameter_counts.json',counts)
    # Interleave architectures within each seed so queue time does not favor a method.
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);source=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/cvs_clean_design/configs/'+rid+'.json'
            cfg=dict(method='cvs_clean_design',variant=variant,model_seed=seed,source_contract=old['data']['contract_ref'],dataset=old['data']['dataset'],output_root=source,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],
                selection='fixed_last_epoch',split_seed=392005,device='cuda:0')
            write(ROOT/ref,cfg);remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            row=dict(row_id=rid,method=variant,purpose='fixed_clean_baseline' if variant in BASELINES else 'source_only_architecture_candidate',
                gpu=None,config_ref=ref,resolved_config_ref=source+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Dedicated batch-loader RNG follows model seed; same physical data; no augmentation/support; no target access in source stage',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='200x50 updates;6300 L_s samples/epoch;no augmentation',output_root=source,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_clean_design.source --config '+remote,
                expected_artifacts=['last.pt','completion.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv'])
            spec['rows'].append(row);runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=source))
    write(ROOT/'experiments/cvs_clean_design/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_clean_design/configs/launch_spec.json',runtime)
    print(json.dumps(dict(run_id=RUN,rows=len(runtime['rows']),parameters=counts),ensure_ascii=False))


if __name__=='__main__':main()
