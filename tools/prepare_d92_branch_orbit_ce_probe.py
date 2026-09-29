"""Preregister the complete fixed orbit/CE factorial support diagnostic."""
import copy
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN='20260929-phase2-d92-branch-orbit-ce-support-m4-r01'
RELEASE='d92_branch_orbit_ce_support_20260929_r01'
SPEC='configs/d92_branch_orbit_ce_support_20260929.json'


def documents():
    original=json.loads((ROOT/'configs/d92_branch_support_probe_20260929.json').read_text(encoding='utf-8'))
    s=copy.deepcopy(original)
    remote=str(Path(original['execution']['remote_run_root']).parent).replace('\\','/')+'/'+RUN
    release=str(Path(original['code']['cwd']).parent).replace('\\','/')+'/'+RELEASE
    algorithm=json.loads((ROOT/'configs/d92_branch_orbit_ce_frozen_20260929.json').read_text(encoding='utf-8'))['algorithm']
    s.update(run_id=RUN,group_id='d92-branch-orbit-ce-support-development',
        display_name='完整分支四相位轨道与交叉熵的support四臂诊断',
        description='固定Phase1与同一received观测，四相位完整分支轨道表示和交叉熵判别头的2×2对照；完整物理OOF及support内部全部单样本anchors。',
        tags=['d92','branch-orbit-ce','support-only','received-views','frozen-phase1','no-query-access'],
        parent_run_ids=['20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01','20260929-phase2-d92-branch-metric-support-m4-r01'],
        status='PLANNED',notes=['No second LEO realization and no additional physical samples; four C4 views share one physical loss mass.',
            'All current-row support and all registered classes use the same fixed formula and loss constants.',
            'True K1 numerical diagnostics are separate from parent K5/10/20 one-shot proxy; all anchors retained.',
            'Support diagnostic informs this design; historical query results are not supplied to the method designer.',
            'Frozen original source-only models only; no adapted head or source samples/statistics are loaded.'])
    s['code'].update(cwd=release,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    s['data']['representation']='Only registered support from immutable received IQ; deterministic C4 views; frozen z/t/f/pa [N,4,160] and original FFT96'
    s['permissions'].update(regime='current_row_support_only_orbit_ce_diagnostic',
        truth_use='Only held legal support labels after fixed support predictions; no query scorer')
    s['checkpoint']['current_auxiliary_training']='Frozen Phase1; only current support centred RKHS heads: two fixed ridge controls and two regularized multiclass CE heads'
    s['execution'].update(remote_run_root=remote,remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_branch_orbit_ce_probe.py --spec {SPEC}')
    s['metrics_plan'].update(metric_names=['support_OOF_old_accuracy','support_OOF_new_accuracy','support_OOF_H',
        'support_oneshot_proxy_old_accuracy','support_oneshot_proxy_new_accuracy','support_oneshot_proxy_H',
        'train_CE','RKHS_regularization','objective','RKHS_gradient_norm','solver_iterations','actual_resource_cost'],
        dimensions=['model_seed','cohort','receiver','scenario','parent_K','train_K','new_count','support_seed','diagnostic','arm'],
        interpretation='At EACH parent K5/10/20, BOTH physical OOF and one-shot proxy require orbit_ce joint H/new improvement against single_ridge, single_ce and orbit_ridge, and joint old decline <=1pp. Strict old improvement separately reported. Full 2x2 controls retained, no per-K method stitching, no query or formal-K1 generalization claim. All-old tasks separately reported.')
    s['probe'].update(algorithm_config=release+'/configs/d92_branch_orbit_ce_frozen_20260929.json',
        view_count=4,reuse_support_cache=False,candidate='orbit_ce',controls=['single_ridge','single_ce','orbit_ridge'],
        expected_true_k1_parents=1200,expected_oof_parents=3600,expected_proxy_anchors=42000,
        max_factorizations=105600,expected_ce_fits=105600,
        diagnostics=['physical_oof','support_oneshot_proxy'],
        proxy_policy='All sorted positions in each current parent support set; no cross-row borrowing; parent-mean before task means')
    s['expected_artifacts'] += ['each row/probe/training_steps.jsonl','each row/probe/training_steps.csv',
        'each row/probe/fit_stages.jsonl','each row/probe/fit_stages.csv']
    outputs={}
    for key,co in s['probe']['cohorts'].items():
        name=f'configs/d92_branch_orbit_ce_support_{key}_20260929.json'
        co['evaluation_config']=release+'/'+name
        outputs[name]=dict(algorithm=algorithm,matrix=co['matrix'])
    for row in s['rows']:
        out=remote+'/'+row['row_id']
        row.update(method='D92-BranchOrbitCE fixed factorial support diagnostic',
            purpose='Separate full-branch phase orbit representation from multiclass CE competition',
            config_ref=SPEC,resolved_config_ref=out+'/probe/startup.json',output_root=out,log_path=out+'/probe.log',
            optimizer='Fixed regularized centred RKHS accelerated gradient for CE; exact ridge controls; train-gradient stopping only',
            lr=None,epochs=None,
            budget_ref='Full4800 parents;3600 physical three-fold OOF and all42000 one-shot anchors; four fixed arms; CE numeric budget in frozen config',
            command=f'{s["code"]["environment"]} -u {release}/tools/run_d92_branch_orbit_ce_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
        row['expected_artifacts'] += ['probe/training_steps.jsonl','probe/training_steps.csv','probe/fit_stages.jsonl','probe/fit_stages.csv']
    outputs[SPEC]=s
    return outputs


if __name__=='__main__':
    outputs=documents()
    if any((ROOT/name).exists() for name in outputs):raise FileExistsError('Already prepared; reconcile instead of overwrite')
    for name,value in outputs.items():
        with (ROOT/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',files=list(outputs))))
