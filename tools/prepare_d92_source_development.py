"""Register the bounded source-only D92 successor diagnostic."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=Path('E:/type10-7')
RUN='20260928-diagnostic-d92-scv-source-s2026092701-r01'
RELEASE='d92_scv_source_20260928_r01'


def main():
    base=json.loads((ROOT/'configs/cvs_d92_matched_20260927.json').read_text(encoding='utf-8'))
    spec=json.loads((ROOT/'configs/d92_upgrade_source_20260928.json').read_text(encoding='utf-8'))
    project='/home/szu2070436088/2510044040/CV-SincNet'
    remote=project+'/runs/'+RUN;release=project+'/releases/'+RELEASE
    source=base['source_run_root']+'/cvs-daot-rc4-s2026092701'
    development=dict(source_root=source,source_contract=base['source_contract'],native_code=base['native_code'],
        model_seed=2026092701,device='cuda:0',scene_seed=2026092801,augmentation_seed=2026092802,
        support_seed=2026092803,validation_per_class=50,k=[1,5,10,20],
        scenarios=['practical_high','practical_mid','practical_low_urban'],
        methods=['frozen_dg','D92','D92-SCV-v1'],feature_output=remote+'/features',evaluation_output=remote+'/evaluation',
        ground=project+'/runs/20260927-phase2-cvs-d92-practical-manytx-m5-r01/cvs-daot-rc4-s2026092701/ground')
    command=('/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python '+release+
        '/tools/run_d92_source_development.py --spec '+release+'/configs/d92_upgrade_source_20260928.json --commit RELEASE_HEAD')
    spec.update(run_id=RUN,group_id='d92-fixed-phase1-support-cv-upgrade',display_name='固定Phase1的D92-SCV源域适应诊断',
        description='固定第一个新模型种子，不按目标成绩选择。源域L拟合、单一V验证；60个RX/scene/K切片对照D92与冻结DG。目标是为后续新旧类H优化验证实现和旧类行为。',
        stage='source-development',authorization='2026-09-28本对话：优化D92，允许大幅改动；用户选择固定Phase1、优先各K综合H并约束旧类退化。',
        parent_run_ids=['20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'],
        tags=['d92','source_only','fixed_phase1','support_cv'],development=development)
    spec['code']=dict(commit=None,checkout=str(ROOT),cwd=release,environment='N607 CVS-RFFI; local ssr-gpu; exact frozen native code',
        base_commit='21e952081',commit_note='Publisher injects exact pushed HEAD into startup/launch evidence; self-referential config omitted.')
    spec['data']=copy.deepcopy(base['data'])
    spec['data'].update(dataset=source+'/resolved_config.json#data',representation='source L/V fixed one residual_noeq scene per physical record; identity160+FFT96',
        target_receivers=[],target_days=[],capsule_id=None,split_id='source-original-contract-L/V; deterministic-scene2026092801',
        validation_ref=source+'/source_contract.json#role_ids.V',support_query_ref=source+'/source_contract.json#role_ids',
        k=[1,5,10,20],new_class_counts=[0],support_seeds=[2026092803])
    spec['data'].pop('phase1_final_seeds',None)
    spec['permissions']=dict(regime='source_only',query_use='single source V evaluation only; no persistent fit from V',external_method_exception=None,
        claim_scope='Source six-seen-TX adaptation diagnostic only; not target generalization or new-class H.')
    spec['checkpoint']=dict(initialization='frozen matching source final200; no retraining',sources=[source+'/final_ssdg.pth'],
        contract_check_ref=base['source_contract'],provenance_verdict='CHECK_AT_LOAD exact source roles, scratch empty ancestors, final-only selection; original source-ground binding checked',
        selection_rule='fixed first fresh model seed2026092701, epoch200; no target score selection')
    spec['execution']=dict(host='N607',launch_owner='codex/root/d92-upgrade-20260928',gpu_policy='one inference worker GPU0, then CPU2 threads; no encoder training',
        remote_run_root=remote,remote_log_root=remote,local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_source_development.py',
        stop_rule='On technical failure preserve all artifacts; no automatic retry; do not stop other tasks.')
    spec['rows']=[dict(row_id='source-development-s2026092701',method='frozenDG vs D92 vs D92-SCV-v1',purpose='source_only_development',
        config_ref='configs/d92_upgrade_source_20260928.json',resolved_config_ref=remote+'/startup.json',data_overrides={},
        seeds=dict(model=2026092701,split=392005,data=None,augmentation=2026092802,support=2026092803,evaluation=2026092801),
        seed_notes='data=null: existing fixed physical contract; evaluation=scene-assignment seed. Encoder frozen, no model randomness. V first50 per class in stable physical order.',
        k=[1,5,10,20],scenario=','.join(development['scenarios']),optimizer='closed-form and legal L-support internal CV; legacy D92 unchanged',
        lr=None,epochs=None,fl_rounds=None,budget_ref='one export all6300 L and27000 V, then5RX x3scenes x4K x3methods=180 rows',
        output_root=remote,log_path=remote+'/run.log',command=command,gpu=0,
        expected_artifacts=['features/complete.json','evaluation/metrics.jsonl','evaluation/summary.csv','evaluation/complete.json'])]
    spec['expected_artifacts']=spec['rows'][0]['expected_artifacts']
    spec['metrics_plan']=dict(metric_names=['source_validation_accuracy','fit_seconds','persistent_state_bytes'],dimensions=['receiver','scene','K','method'],
        prediction_ref=None,scorer_ref='tools/evaluate_d92_source_development.py; source V only')
    spec['notes']=['No new TX in Phase1 source contract: H and new-class generalization remain unverified.',
        'Existing target benchmark is exposed. It must not choose hyperparameters, candidates, seeds or reruns.',
        'User acceptance objective: improve per-K joint H with no material old accuracy regression. Source diagnostic is not goal completion.',
        'No gradient training: loss/lr/gradient null with explicit reasons; full per-fold selection trace and compact JSONL/CSV retained.',
        'RELEASE_HEAD is replaced by publisher with the independently verified pushed commit; exact argv in startup.json.']
    path=ROOT/'configs/d92_upgrade_source_20260928.json'
    path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
