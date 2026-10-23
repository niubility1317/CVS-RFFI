"""Compare fixed hard-PL ablations only after both complete target reports exist."""
import argparse
import json
import math
from pathlib import Path
import statistics

HARD_RUN='20261024-phase1-PAMF-HardPL-T1U-m6-r01'
PAIRS={
    '20261024-phase1-PAMF-HardPL-noUAs-m3-r01':('HPL_NO_UAS','HPLT1U','Only U channel transform removed'),
    '20261024-phase1-PAMF-HardPL-LS-m3-r01':('HPL_LS2','HPL2','Only U CE label smoothing0->.01'),
    '20261024-phase1-PAMF-HardPL-MixStyle-m3-r02':('HPL_MS2','HPL_LS2','Only accepted same-pseudo cross-RX U feature style'),
    '20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01':('HPL_NAMS2','HPL_NO_UAS','Only shared accepted U feature style in existing six forwards'),
}
REFERENCES={
    '20261024-phase1-PAMF-HardPL-MixStyle-m3-r02':'20261024-phase1-PAMF-HardPL-LS-m3-r01',
    '20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01':'20261024-phase1-PAMF-HardPL-noUAs-m3-r01',
}
SEEDS=(2026092701,2026092702,2026092703)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban','six_mean')
READOUTS=('raw','DURA-LU')
DATA_KEYS=('dataset','version','representation','physical_ids_ref','label_map_ref',
           'source_receivers','target_receivers','source_days','target_days','tx_sets_ref','roles','capsule_id','split_id')


def compare(candidate,reference,candidate_spec,reference_spec,candidate_completion,reference_completion):
    run=candidate['run_id'];method,control,factor=PAIRS[run]
    reference_run=REFERENCES.get(run,HARD_RUN);reference_models=6 if reference_run==HARD_RUN else 3
    if reference['run_id']!=reference_run:raise ValueError('Fixed matched hard-PL reference required')
    for summary,spec,completion in ((candidate,candidate_spec,candidate_completion),(reference,reference_spec,reference_completion)):
        if (summary['model_seeds']!=list(SEEDS) or spec['run_id']!=summary['run_id']
            or completion['run_id']!=summary['run_id'] or completion['status']!='VERIFIED_COMPLETE'
            or completion['independent_truth_last'] is not True or completion['readouts']!=list(READOUTS)
            or completion['views_per_readout']!=7 or completion['count_per_view']!=168000
            or completion['completed_F_models']!=(3 if summary is candidate else reference_models)
            or completion['completed_readout_view_cells']!=(42 if summary is candidate else 14*reference_models)):
            raise ValueError('Both fixed three-seed full target matrices must be complete')
    if candidate['class_order']!=reference['class_order'] or len(candidate['class_order'])!=6:
        raise ValueError('Matched class order differs')
    if any(candidate_spec['data'][key]!=reference_spec['data'][key] for key in DATA_KEYS):
        raise ValueError('Matched Phase1 physical data contract differs')

    def counts(summary,name,readout,view):
        rows=[r for r in summary['summary'] if (r['method'],r['readout'],r['view'],r['dimension'],r['stratum'])
              ==(name,readout,view,'overall','ALL')]
        if len(rows)!=1:raise ValueError('Missing or duplicated matched summary cell')
        row=rows[0];denom=1008000 if view=='six_mean' else 168000
        values=[row['correct_s'+str(seed)] for seed in SEEDS]
        if (row['count_per_seed']!=denom or any(type(v) is not int or not 0<=v<=denom for v in values)
            or row['correct_total']!=sum(values)
            or not math.isclose(row['mean_accuracy_percent'],100.*sum(values)/(3*denom),rel_tol=0.,abs_tol=1e-8)):
            raise ValueError('Matched exact counts and reported means differ')
        return values,denom

    rows=[]
    for readout in READOUTS:
        candidate_six=[0,0,0];reference_six=[0,0,0]
        for view in VIEWS:
            left,denom=counts(candidate,method,readout,view);right,other=counts(reference,control,readout,view)
            if denom!=other:raise ValueError('Matched count per seed differs')
            if view in VIEWS[1:-1]:
                candidate_six=[a+b for a,b in zip(candidate_six,left)]
                reference_six=[a+b for a,b in zip(reference_six,right)]
            if view=='six_mean' and (left!=candidate_six or right!=reference_six):
                raise ValueError('Six-view aggregate is not the sum of all six registered scenes')
            deltas=[100.*(a-b)/denom for a,b in zip(left,right)]
            rows.append(dict(readout=readout,view=view,count_per_seed=denom,
                candidate_mean_accuracy_percent=100.*sum(left)/(3*denom),
                reference_mean_accuracy_percent=100.*sum(right)/(3*denom),
                paired_delta_pp_mean=statistics.mean(deltas),paired_delta_pp_sample_sd=statistics.stdev(deltas),
                positive_seeds=sum(value>0 for value in deltas),net_correct_total=sum(left)-sum(right),
                per_seed=[dict(model_seed=seed,candidate_correct=a,reference_correct=b,net_correct=a-b,delta_pp=delta)
                          for seed,a,b,delta in zip(SEEDS,left,right,deltas)]))
    return dict(status='VERIFIED_PAIRED_MATCHED_COMPARISON',candidate_run=run,reference_run=reference_run,
        candidate_method=method,reference_method=control,changed_factor=factor,model_seeds=list(SEEDS),rows=rows,
        primary_goal_unchanged='Raw clean>=+1pp and six>+1.5pp against both F20/F21;this ablation does not replace that goal',
        interpretation='Differences and sample SD paired across three F seeds on the same fixed physical target records;no independent-sample or statistical-significance claim')


def markdown(result):
    text=f"# {result['candidate_method']} 对 {result['reference_method']} 的匹配比较\n\n"
    text+='两个固定三种子矩阵已全部完成独立 truth-last。下表差值为候选减比较项，SD 为三个配对模型 seed 的样本标准差；六景复用相同物理样本，不视为新增独立样本。该消融不替代相对 F20／F21 的原目标判定。\n\n'
    text+='| 读出 | 场景 | 候选准确率（%） | 比较项准确率（%） | 配对差 ± SD（pp） | 正收益种子 |\n|---|---|---:|---:|---:|---:|\n'
    for r in result['rows']:
        text+=f"| {r['readout']} | {r['view']} | {r['candidate_mean_accuracy_percent']:.6f} | {r['reference_mean_accuracy_percent']:.6f} | {r['paired_delta_pp_mean']:+.6f} ± {r['paired_delta_pp_sample_sd']:.6f} | {r['positive_seeds']}/3 |\n"
    return text+'\n逐 seed 正确计数与配对差见 matched_comparison.json。固定参考为 '+result['reference_run']+'；完整主目标、逐 TX／RX／day 与修复损伤仍见各自 target_results.md 及附表。\n'


def complete_report(report_root,run):
    if run not in PAIRS:raise ValueError('Declared hard-PL ablation required')
    root=Path(report_root);folders=(root/run,root/REFERENCES.get(run,HARD_RUN))
    completions=[]
    # Inspect completion metadata before opening either score summary.
    for folder in folders:
        path=folder/'evaluation_completion.json'
        if not path.is_file():return dict(status='WAIT_COMPLETE_MATCHED_REPORTS',missing_run=folder.name,scores_read=False)
        value=json.loads(path.read_text(encoding='utf-8'))
        if value.get('status')!='VERIFIED_COMPLETE':
            return dict(status='WAIT_COMPLETE_MATCHED_REPORTS',missing_run=folder.name,scores_read=False)
        completions.append(value)
    summaries=[json.loads((folder/'target_summary.json').read_text(encoding='utf-8')) for folder in folders]
    specs=[json.loads((folder/'experiment.json').read_text(encoding='utf-8')) for folder in folders]
    result=compare(*summaries,*specs,*completions)
    output=folders[0];json_path=output/'matched_comparison.json';md_path=output/'matched_comparison.md'
    if json_path.exists() or md_path.exists():raise FileExistsError('Matched comparison already exists;preserve original output')
    json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md_path.write_text(markdown(result),encoding='utf-8')
    return dict(status=result['status'],run_id=run,files=['matched_comparison.json','matched_comparison.md'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report-root',required=True)
    parser.add_argument('--run',required=True,choices=tuple(PAIRS));args=parser.parse_args()
    print(json.dumps(complete_report(args.report_root,args.run)))
