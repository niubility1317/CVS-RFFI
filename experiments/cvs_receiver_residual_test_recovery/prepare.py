"""Pre-register two independent test completions of the already trained matrix."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from experiments.cvs_receiver_residual_test_recovery import design as d


def main():
    output=d.ROOT/'local_artifacts'/d.RELEASE;output.mkdir(parents=True,exist_ok=True)
    for run in d.RUNS:
        parent=d.ROOT/'automation_reports/CV-SincNet'/run['parent_run_id']
        spec=copy.deepcopy(json.loads((parent/'experiment.json').read_text(encoding='utf-8')))
        spec['run_id']=run['run_id'];spec['display_name']='冻结E200独立测试：'+('R1随机混杂诊断' if run['package']=='cvs_receiver_residual' else 'R2修复后四seed正式消融')
        spec['description']='修复旧source role契约无classes字段的冻结检查错误；只使用对应原run全部16个已训练E200模型，不训练、不适应、不改变算法/预算；clean+六完整practical视图全部预测后独立truth-last评分。'
        spec['parent_run_ids']=[run['parent_run_id']];spec['replaces_run_id']=None
        spec['authorization']='原完整训练测试授权；用户2026-10-09要求完整测试集结果。仅恢复原预登记测试，不重训/选择性重跑，保留原失败及R1随机混杂。'
        spec['tags']=list(dict.fromkeys(spec['tags']+['test_completion','frozen_checkpoint','schema_repair']))
        spec['code']=dict(commit='release_commit.txt fixes actual evaluation Git commit before publish',
            checkout=str(d.ROOT),environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',
            cwd=d.PROJECT+'/releases/'+d.RELEASE,source_runtime_commit=run['source_commit'])
        sources=[]
        for row in spec['rows']:
            rid=row['row_id'];source=d.PROJECT+'/runs/'+run['parent_run_id']+'/'+rid+'/source'
            destination=d.PROJECT+'/runs/'+run['run_id']+'/'+rid+'/prediction'
            sources.append(dict(row_id=rid,checkpoint=source+'/last.pt',initialization_ref=source+'/initialization.json',
                contract_ref=source+'/source_contract.json',selection_ref=source+'/source_selection.json',
                parent_run_id=run['parent_run_id'],source_runtime_commit=run['source_commit'],
                selection='own fixed E200;10000 updates; original source roles; no target access',ancestors=[]))
            row.update(purpose='frozen_all_fixed_controls_test',optimizer=None,lr=None,epochs=0,
                budget_ref='No training/adaptation; use own original E200x50 steps checkpoint',
                output_root=destination,prediction_root=destination,source_root=source,
                config_ref=source+'/resolved_config.json',resolved_config_ref=destination+'/resolved_config.json',
                log_path=d.PROJECT+'/logs/'+run['run_id']+'/'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_receiver_residual_test_recovery.dispatch --worker --run '+run['run_id']+' --row '+rid,
                expected_artifacts=['provenance.json','resolved_config.json','predictions.npz','complete.json'])
        spec['checkpoint']=dict(initialization='own_frozen_source_checkpoint',sources=sources,
            contract_check_ref='source_matrix_frozen.json',provenance_verdict='Must exact-match all original role fields and ordered physical class map before load',
            selection_rule='All16 own fixed E200; .5V+.5worstRX descriptive preference; every fixed control tested')
        spec['execution']=dict(host='N607',launch_owner=d.OWNER,
            gpu_policy='<=4 total experiment processes/GPU including pre-CUDA reservations; shared capacity lock; >=12GB free',
            remote_run_root=d.PROJECT+'/runs/'+run['run_id'],remote_log_root=d.PROJECT+'/logs/'+run['run_id'],
            local_artifact_root='automation_reports/CV-SincNet/'+run['run_id'],
            launch_command='python -m experiments.cvs_receiver_residual_test_recovery.publish --output local_artifacts/'+d.RELEASE,
            stop_rule='Technical prediction failure retains artifacts; no automatic retry, source retraining, or unrelated intervention')
        spec['expected_artifacts']=['source_matrix_frozen.json','launch_predict.json','queue_state.json','scores.json','scores.csv',
            'summary.json','summary.csv','paired_results.json','resources.json','analysis.md','scoring_complete.json','completion.json']
        spec['notes']=['Original role contract lacks classes; training export adds actual ordered classes. Compare every original field and separately enforce the original fixed class map. No checkpoint contract relaxation.',
            'Original training roots and failed controllers remain unchanged. Original immutable model code is imported from each corresponding source release. No fresh training or target adaptation.',
            'All32 sources frozen before any query; all32x7 predictions integrity-validated before either scorer opens truth. No target feedback to model/seed/arm selection.',
            run['disposition'],'Exposed ManySig benchmark; six complete practical residual views share168000 physical IDs. Not disjoint formalLEO strata or real satellite validation.',
            'Phase2/K/old support adaptation/new classes/H: N/A. Full RX/TX/seed/view scores, Macro-F1, confusion, paired deltas and resources retained.']
        spec['status']='PLANNED';spec.pop('registered_at',None)
        path=output/(run['run_id']+'.json');path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-X','utf8','E:/type10-7/tools/experiment_registry.py','--root',str(d.ROOT),'new','--spec',str(path)],check=True)
        report=d.ROOT/'automation_reports/CV-SincNet'/run['run_id']/'report.md'
        with report.open('a',encoding='utf-8') as f:
            f.write('\n## 独立测试恢复\n\n源训练已全部完成，原pipeline在访问目标前因`expected[classes]`报错。旧角色契约没有该字段，训练导出才追加实际物理类别顺序。修复在新评估release/独占输出中执行，原模型、权重和失败产物保留。根因已本地复现，新增核验逐项比较全部原契约字段，并单独检查原固定六类顺序、receivedIQ与scratch来源。\n\n'
                '原训练run：`'+run['parent_run_id']+'`，source commit：`'+run['source_commit']+'`。本轮无训练、无适应，测试全部16个ownE200。R1只作随机混杂诊断；正式收益以R2为准。所有32×7数组完成并检查后才开truth；当前尚未评分。\n')
    print('Registered both fixed independent evaluation matrices')


if __name__=='__main__':main()
