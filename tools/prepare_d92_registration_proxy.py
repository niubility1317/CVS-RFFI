"""Preregister all20 source classifier-entry holdouts before reading proxy scores."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN='20260928-diagnostic-d92-registration-proxy-s2026092701-r01'
RELEASE='d92_registration_proxy_20260928_r01'


def main():
    spec=json.loads((ROOT/'configs/d92_upgrade_source_20260928.json').read_text(encoding='utf-8'))
    project='/home/szu2070436088/2510044040/CV-SincNet';remote=project+'/runs/'+RUN
    code=project+'/releases/'+RELEASE
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    spec.update(run_id=RUN,display_name='D92新旧类竞争的源域分类头遮蔽代理校准',
        description='源域六个已见TX的全部20种3旧/3注册角色组合；只隐藏分类头入口，不声称真正新TX。L支持拟合、单一V评分；K1权重source-only选择。',
        parent_run_ids=['20260928-diagnostic-d92-scv-source-s2026092701-r01'],stage='source-registration-proxy',
        development=dict(features=project+'/runs/20260928-diagnostic-d92-scv-source-s2026092701-r01/features',
            support_seed=2026092803,validation_per_class=50,k=[1,5,10,20],k1_fft_grid=[0.0,0.5,1.0,4.0],expected_rows=2100))
    spec['code'].update(commit=commit,cwd=code)
    spec['data']['new_class_counts']=[3];spec['data']['tx_sets_ref']='all6 source TX; all20 choices of3 masked classifier entries; proxy roles only'
    spec['permissions']['claim_scope']='Source classifier-entry holdout proxy only; allTX seen byPhase1, no actual novel-TX claim.'
    command=f'/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python {code}/tools/evaluate_d92_registration_proxy.py --spec {code}/configs/d92_registration_proxy_20260928.json --commit RELEASE_HEAD'
    spec['execution'].update(remote_run_root=remote,remote_log_root=remote,local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_registration_proxy.py',gpu_policy='CPU one lane twoBLASthreads; no checkpoint reload or GPU')
    row=spec['rows'][0];row.update(row_id='source-registration-proxy-s2026092701',method='D92-SCV source classifier-entry holdout',
        config_ref='configs/d92_registration_proxy_20260928.json',resolved_config_ref=remote+'/startup.json',
        output_root=remote,log_path=code+'/run.log',command=command,gpu=None,
        expected_artifacts=['metrics.jsonl','complete.json'],budget_ref='15RXscene cells x20role partitions x(4K1weights+3otherK)=2100 source-only rows')
    spec['expected_artifacts']=row['expected_artifacts'];spec['metrics_plan']['metric_names']=['old_accuracy','new_proxy_accuracy','harmonic_mean','fit_seconds']
    spec['metrics_plan']['scorer_ref']='tools/evaluate_d92_registration_proxy.py; source V only'
    spec['notes']=['Frozen Phase1 unchanged. All six TX are source seen, role holdout is not actual novelTX.',
        'Select K1 FFT weight from sourceV only: maximum proxyH, old and proxy-new each no worse than identity baseline by1pp.',
        'K>=5 algorithm and all grids fixed; no target capsule, query or scores are accessed.',
        'All20 combinations avoid selection of favorable class roles; input features reused from verified complete33300 source export.']
    path=ROOT/'configs/d92_registration_proxy_20260928.json';path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
