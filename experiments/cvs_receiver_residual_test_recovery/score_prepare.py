"""Register independent scorer outputs; retain failed output and predictions."""
import copy
import json
import subprocess
import sys
from experiments.cvs_receiver_residual_test_recovery import design as d


def main():
    output=d.ROOT/'local_artifacts'/d.SCORE_RELEASE;output.mkdir(exist_ok=True)
    for run in d.SCORE_RUNS:
        previous=d.ROOT/'automation_reports/CV-SincNet'/run['prediction_run_id']
        spec=copy.deepcopy(json.loads((previous/'experiment.json').read_text(encoding='utf-8')))
        spec.update(run_id=run['run_id'],status='PLANNED',parent_run_ids=[run['parent_run_id'],run['prediction_run_id']],
            display_name='完整冻结预测独立评分：'+('R1混杂诊断' if run['package']=='cvs_receiver_residual' else 'R2四seed正式消融'),
            description='固定并复用已经完成的16×7预测，修复接收机编号与物理名称的评分口径；CPU独立truth-last评分，输出完整RX/TX/day分层。无训练、无重新预测、无目标反馈。')
        spec.pop('registered_at',None)
        spec['tags']=list(dict.fromkeys(spec['tags']+['score_only','physical_receiver_metadata_repair']))
        spec['code'].update(commit='release_commit.txt records actual scoring release commit',cwd=d.PROJECT+'/releases/'+d.SCORE_RELEASE)
        base=d.PROJECT+'/runs/'+run['run_id']
        for row in spec['rows']:
            row.update(purpose='immutable_prediction_score_only',output_root=base+'/'+row['row_id'],
                log_path=base+'/scorer.stdout.log',resolved_config_ref=row['prediction_root']+'/resolved_config.json',
                command='python -m experiments.cvs_receiver_residual_test_recovery.evaluate --mode score --run '+run['run_id'],
                expected_artifacts=['scores.json','scores.csv','day_scores.json','day_scores.csv','summary.json',
                    'paired_results.json','resources.json','receiver_map.json','scoring_complete.json'])
        spec['checkpoint']['contract_check_ref']=d.PROJECT+'/runs/'+run['prediction_run_id']+'/source_matrix_frozen.json'
        spec['execution'].update(remote_run_root=base,remote_log_root=base,
            gpu_policy='CPU-only scoring; no GPU experiment processes; reuse all existing predictions',
            launch_command='python -m experiments.cvs_receiver_residual_test_recovery.score_publish',
            local_artifact_root='automation_reports/CV-SincNet/'+run['run_id'],
            stop_rule='Retain score failure artifacts; no train, prediction, tuning or automatic retry')
        spec['expected_artifacts']=['lineage.json','scores.json','scores.csv','day_scores.json','day_scores.csv','summary.json',
            'summary.csv','paired_results.json','resources.json','receiver_map.json','analysis.md','scoring_complete.json','completion.json']
        spec['notes'] += ['Prediction matrices are complete and immutable before this scoring correction; original scoring failed before writing any result.',
            'RX map is original ManySig rx_list at registered target indices0,2,5,7,9,10,11; truth physical labels must match. Days are all four registered dates.',
            '1568 main rows plus448 supplemental day rows perrun; all fixed rows are scored with no selective reporting.']
        path=output/(run['run_id']+'.json');path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-X','utf8','E:/type10-7/tools/experiment_registry.py','--root',str(d.ROOT),'new','--spec',str(path)],check=True)
        report=d.ROOT/'automation_reports/CV-SincNet'/run['run_id']/'report.md'
        with report.open('a',encoding='utf-8') as f:
            f.write('\n## 评分恢复\n\n原32×7预测全部固定后，评分器把truth中的物理RX名称当作编号而报错；没有写出评分结果。原预测与失败目录保留，本轮只在独占目录完成CPU评分。映射来自原数据集rx_list，不从性能推断。正式结论使用R2，R1为随机混杂诊断；所有负结果保留。\n')
    print('Registered two CPU-only full scoring matrices')


if __name__=='__main__':main()
