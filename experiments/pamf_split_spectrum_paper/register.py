"""Create the experiment-management spec from the fixed registered matrix."""
import copy
import json
import subprocess
from pathlib import Path
from .plan import RUN, rows


def register(repo, out):
    repo, out = Path(repo), Path(out)
    old = repo/'automation_reports/CV-SincNet/20261109-phase1-pamf-no-reference-completion-m190-r01/experiment.json'
    d = json.loads(old.read_text(encoding='utf8'))
    context = json.loads((out/'context.json').read_text(encoding='utf8'))
    d.update(run_id=RUN, group_id='phase1-pamf-split-spectrum-paper',
             display_name='PAMF 时域、PA 与频谱统计完整实验',
             description='226 rows: separate PA IQ and spectral statistics, eleven data settings, five training conditions, PL permutation and E200; direct and matched DURA readouts.',
             authorization='用户明确授权226逻辑模型及DURA-L/LU读出；旧任务保持运行；每卡本实验0–6最多3进程，7最多1进程。',
             aliases=[], tags=['PAMF','split-spectrum','no-reference','external-methods','receiver-splits','label-ratio','truth-last'],
             parent_run_ids=['20261109-phase1-pamf-branch-factorial-m36-r01','20261023-phase1-practical-residual-full2x2-m600-r01'], replaces_run_id=None)
    for key in ('result_summary','registered_at','launch_verification_ref'):
        d.pop(key, None)
    d['code'].update(commit='release Git HEAD recorded in runtime.json',entry='experiments.pamf_split_spectrum_paper.runner',cwd='PROJECT/releases/'+RUN)
    d['checkpoint'].update(initialization='211 scratch models; 15 same-contract completed branch models reused; 528 external results reused',
        sources=['20261109 completed branch panel','20261023 external matrix','same-contract DURA caches','14-network clean E200 results'],
        selection_rule='fixed final epoch; all registered rows retained; no target feedback',
        contract_check_ref='Existing source roles, seed, frozen checkpoint, prediction and score checked per reused row')
    d['execution'].update(launch_owner='codex-pamf-split-spectrum-20261109',
        gpu_policy='GPU0-6 <=3 own workers/card; GPU7 <=1 own; other owners excluded; free >=12000 MiB; CPU and memory guard',
        remote_run_root='PROJECT/runs/'+RUN,remote_log_root='PROJECT/runs/'+RUN+'/logs',
        local_artifact_root='local_artifacts/pamf_split_spectrum_paper_20261109',
        launch_command='python -m experiments.pamf_split_spectrum_paper.publish --repo <worktree> --output <artifact-root>',
        stop_rule='actual source checkpoint smoke; two identical technical failures or contract/query/output fault stop new dispatch; no metric-based stops')
    d['expected_artifacts']=['211 new frozen models and associated predictions and independent scores','15 reused scored structural models',
        '528 external scored rows','main direct and DURA-L/LU readouts','paired-seed structure and external comparisons',
        'fixed six-receiver curve','E200 eight-seed clean architecture comparison']
    template = d['rows'][0]; new = []
    for row in rows():
        r=copy.deepcopy(template); rid=row['row_id']; base='PROJECT/runs/'+RUN; reuse=not row['new_training']
        r.update(row_id=rid,method='PAMF '+row['method'],purpose='fixed split-spectrum Phase1 paper comparison',paper_row=row,
            config_ref='verified historical source' if reuse else base+'/configs/train__'+rid+'.json',
            resolved_config_ref='verified historical source' if reuse else base+'/F/'+rid+'/source/resolved_config.json',
            seeds={'model':row['model_seed'],'split':392005,'data':None,'augmentation':2026102301 if row['family'] in ('branch','structure_ce') else 20261016421,'support':None,'evaluation':2027},
            seed_notes='Fixed role IDs and declared model/augmentation namespaces; no support sampling.',scenario=row['views'],epochs=row['epochs'],
            optimizer='registered native AdamW and fixed method schedule',lr='native registered schedule',
            budget_ref='E200/10000 pure CE' if row['family']=='architecture' else 'E50=20+10+20, 11100 updates',
            output_root='EXISTING/'+row['reuse_key'] if reuse else base+'/F/'+rid,
            log_path='EXISTING/'+row['reuse_key'] if reuse else base+'/logs/train__'+rid+'.log',
            command='reuse frozen row' if reuse else 'python -m experiments.pamf_split_spectrum_paper.runner worker --kind train --config '+base+'/configs/train__'+rid+'.json',
            expected_artifacts=['source freeze','registered frozen predictions','independent truth-last score'])
        new.append(r)
    d['rows']=new
    d['matrix']={'rows':rows(),'logical_pamf_rows':226,'new_F':211,'reused_F':15,'stages':633,
        'reused_external_rows':528,'historical_clean_networks':14,'source_protocols':context['protocols'],
        'target_populations':['cross_rx','cross_both'],'source_receiver_curve_common_targets':[0,2,5,7,9,10],
        'resource':'GPU0–6 <=3 own; GPU7 <=1 own; other owners excluded','target_feedback':False}
    d['notes']=['Original frequency CNN and reference removed, original PA spectral statistics retained as separate named branch.',
        'CE structure vs full supervised and Tri-SIG-TIME SSL are distinct; previous target exposure disclosed.',
        'User authorized full fixed matrix; metric values do not stop or change other rows.']
    spec=out/'registration_spec.json'
    spec.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    subprocess.run(['F:/App/miniconda3/python.exe','-X','utf8','E:/type10-7/tools/experiment_registry.py',
                    '--root',str(repo),'new','--spec',str(spec)],check=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();register(a.repo,a.output)
