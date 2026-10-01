"""Register two CVS head hypotheses from source-only evidence, four seeds each."""
import copy
import json
from pathlib import Path
from experiments.cvs_residual_identity.model import VARIANTS,build
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261001-phase1-cvs-residual-identity-manysig-m8-r01'
RELEASE='cvs_residual_identity_20261001_r01'
PREDECESSOR='20261001-phase1-cvs-clean-architecture-manysig-m32-r01'
SEEDS=(2026092701,2026092702,2026092703,2026092704)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_clean_design/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-residual-identity-ce',display_name='CVS基础网络研发：原物理三分支＋轻量残差身份融合',
        description='保留原时域/频域/PA分支，纯CE、无增强、从零训练两个残差分类头候选；2架构x4seed；source-only研发，现有基准继续运行。',
        authorization='2026-10-01原目标纯CE/无增强/clean/轻量CVS；用户追加：让基准实验先跑着，先研发CVS。',
        parent_run_ids=[PREDECESSOR],status='PLANNED',stage='Phase1-CVS-source-research',rows=[],
        tags=['cvs','clean_only','architecture','ce_only','no_augmentation','lightweight','residual_fusion','source_selection'])
    spec['checkpoint']['sources']=[]
    spec['checkpoint']['selection_rule']='Scratch E200 for both variants; six registered source-only CVS candidates chosen by four-seed E200 source V before any current clean prediction'
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='583be084d91dc1fd8540bbf1da33b61b91b38975')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-residual-identity-20261001',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,predecessor_pipeline=PROJECT+'/runs/'+PREDECESSOR+'/pipeline_state.json',
        predecessor_selection=PROJECT+'/runs/'+PREDECESSOR+'/source_selection.json',rows=[])
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_residual_identity.publish --output local_artifacts/cvs_residual_identity_20261001_r01',
        gpu_policy='At most2 total training jobs/GPU and12GB free; new dispatcher waits until predecessor has no QUEUED rows, then uses free slots while its running jobs finish')
    spec['metrics_plan']['later_test']='No target inference/scoring in this release. Before any current clean test, combine all six CVS source-only candidate summaries, freeze one final candidate, retain four fixed baselines.'
    spec['notes']=[
        'Current source evidence is an E60 paired two-seed diagnostic; partial curves cannot substitute E200/four-seed selection or clean results.',
        'Native parameters382146;CE-active317665;removing64481 inactive parameters preserves exact training logits/active gradients. This control is not a performance improvement.',
        'Residual head: LN(base)+tanh(gain)*LN(PA_local+PA_delta), gain initialized0.25;cosine scale30/no label margin. Native branches unchanged. No hard input phase invariance.',
        'Candidate2 additionally changes only three temporal pools to mean+tanh(a)*std with192 zero-initialized scalars.',
        'Candidates164225/164417 parameters;all receiveCEgradient locally. MAC reduction only1.56percent,not proportional to parameter reduction. CPU timing is synthetic,not a GPU speedup claim.',
        'Every row scratch200epochsx50steps,6300L/56700Uunused/27000Veval,same physical contract and fixed4seeds;no augmentation/domain backbone/extra loss.',
        'Existing release, queue, baselines and healthy jobs are preserved; queue-drain dependency avoids simultaneous launch-owner races. No target-based tuning or automatic retraining.',
        'All six original/new CVS candidates join final source-only selection before target. The first run retains its own source selection record; no overwrite or retroactive target choice.',
        'Parent run is a source-analysis reference only,not a checkpoint ancestor. No inherited weights/optimizer/teacher/EMA;no target inputs.'
    ]
    spec['expected_artifacts']+=['research_selection.json when predecessor source_selection exists']
    for seed in SEEDS:
        for variant in VARIANTS:
            rid=variant+'-s'+str(seed);output=runtime['runtime_root']+'/'+rid+'/source'
            ref='experiments/cvs_residual_identity/configs/'+rid+'.json'
            cfg=dict(method='cvs_residual_identity',variant=variant,model_seed=seed,source_contract=spec['data']['contract_ref'],
                dataset=spec['data']['dataset'],output_root=output,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,
                drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,device='cuda:0')
            write(ROOT/ref,cfg)
            remote=PROJECT+'/releases/'+RELEASE+'/'+ref
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='source_only_CVS_head_candidate',gpu=None,config_ref=ref,
                resolved_config_ref=output+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Source roles fixed;dedicated matched loader seed;no augmentation/support;no target access',
                k=None,scenario='clean',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='200x50updates;6300L/epoch',
                output_root=output,log_path=runtime['log_root']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_residual_identity.source --config '+remote,
                expected_artifacts=['last.pt','completion.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv']))
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=remote,source_output=output))
    write(ROOT/'experiments/cvs_residual_identity/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_residual_identity/configs/launch_spec.json',runtime)
    write(ROOT/'experiments/cvs_residual_identity/configs/local_parameter_counts.json',{v:sum(p.numel() for p in build(v).parameters()) for v in VARIANTS})
    print(json.dumps(dict(run_id=RUN,rows=8)))


if __name__=='__main__':main()
