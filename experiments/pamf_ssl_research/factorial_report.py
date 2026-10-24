"""Fixed complete Joint/Path/As/combined recipe comparison,without target selection."""
import argparse
import json
import math
from pathlib import Path
import statistics
from .matched_report import SEEDS,VIEWS,READOUTS,DATA_KEYS
from .hard_pl_joint import RUN as JOINT_RUN,METHODS as JOINT_METHODS
from .hard_pl_path import RUN as PATH_RUN,METHODS as PATH_METHODS
from .hard_pl_reliable_as import RUN as AS_RUN,METHODS as AS_METHODS
from .hard_pl_path_reliable_as import RUN as COMBINED_RUN,METHODS as COMBINED_METHODS

MATRICES=dict(joint=(JOINT_RUN,JOINT_METHODS[0]),path=(PATH_RUN,PATH_METHODS[0]),
    As=(AS_RUN,AS_METHODS[0]),combined=(COMBINED_RUN,COMBINED_METHODS[0]))


def analyze(report_root):
    root=Path(report_root);read=lambda p:json.loads(p.read_text(encoding='utf-8'));completions={}
    # No target score file is opened until all four complete metadata records exist.
    for key,(run,method) in MATRICES.items():
        p=root/run/'evaluation_completion.json'
        if not p.is_file():return dict(status='WAIT_COMPLETE_FACTORIAL',missing_run=run,scores_read=False)
        c=read(p)
        if c.get('status')!='VERIFIED_COMPLETE':return dict(status='WAIT_COMPLETE_FACTORIAL',missing_run=run,scores_read=False)
        if (c['run_id']!=run or c['independent_truth_last'] is not True or c['completed_F_models']!=3
            or c['readouts']!=list(READOUTS) or c['views_per_readout']!=7 or c['count_per_view']!=168000
            or c['completed_readout_view_cells']!=42 or c['fixed_G_version']!='dura_v1'):
            raise ValueError('Four full fixed three-seed raw/DURA matrices required')
        source=read(root/run/'source_training_complete.json')
        if (source['cumulative_updates_per_model']!=11100 or source['model_seeds']!=list(SEEDS) or len(source['rows'])!=3
            or any(r['method']!=method or r['source_epoch']!=50 or r['source_updates']!=11100
                or r['actual_adapter_SSL'].get('default_head_joint') is not True or r['actual_adapter_SSL'].get('U_truth_read') is not False for r in source['rows'])):
            raise ValueError('Fixed final E50 source students and jointly trained default heads required')
        completions[key]=c
    specs={key:read(root/run/'experiment.json') for key,(run,_) in MATRICES.items()}
    for key,(run,_) in MATRICES.items():
        if specs[key]['run_id']!=run or any(specs[key]['data'][k]!=specs['joint']['data'][k] for k in DATA_KEYS):
            raise ValueError('Same Phase1 physical data contract required')
    summaries={key:read(root/run/'target_summary.json') for key,(run,_) in MATRICES.items()}
    for key,(run,_) in MATRICES.items():
        s=summaries[key]
        if s['run_id']!=run or s['model_seeds']!=list(SEEDS) or s['class_order']!=summaries['joint']['class_order'] or len(s['class_order'])!=6:
            raise ValueError('Same full model seeds and all registered classes required')

    def counts(key,readout,view):
        m=MATRICES[key][1];rows=[v for v in summaries[key]['summary'] if
            (v['method'],v['readout'],v['view'],v['dimension'],v['stratum'])==(m,readout,view,'overall','ALL')]
        if len(rows)!=1:raise ValueError('One complete cell per declared method/readout/view required')
        c=rows[0];n=1008000 if view=='six_mean' else 168000;values=[c['correct_s'+str(seed)] for seed in SEEDS]
        if (c['count_per_seed']!=n or any(type(v) is not int or not 0<=v<=n for v in values)
            or c['correct_total']!=sum(values) or not math.isclose(c['mean_accuracy_percent'],100*sum(values)/(3*n),abs_tol=1e-8,rel_tol=0)):
            raise ValueError('Exact full integer counts or reported mean differ')
        return values,n

    neighbors=[];interaction=[]
    for readout in READOUTS:
        six={key:[0,0,0] for key in MATRICES}
        for view in VIEWS:
            values={key:counts(key,readout,view)[0] for key in MATRICES};n=1008000 if view=='six_mean' else 168000
            if view in VIEWS[1:-1]:six={key:[a+b for a,b in zip(six[key],values[key])] for key in MATRICES}
            if view=='six_mean' and six!=values:raise ValueError('Six mean must sum every registered scene for every seed')
            for key in ('joint','path','As'):
                nets=[a-b for a,b in zip(values['combined'],values[key])];pp=[100*v/n for v in nets]
                neighbors.append(dict(reference=key,reference_run=MATRICES[key][0],reference_method=MATRICES[key][1],readout=readout,view=view,
                    count_per_seed=n,candidate_accuracy_percent=100*sum(values['combined'])/(3*n),reference_accuracy_percent=100*sum(values[key])/(3*n),
                    net_correct_total=sum(nets),paired_delta_pp_mean=statistics.mean(pp),paired_delta_pp_sample_sd=statistics.stdev(pp),
                    positive_seeds=sum(v>0 for v in nets),per_seed=[dict(model_seed=s,candidate_correct=a,reference_correct=b,net_correct=a-b,delta_pp=d)
                        for s,a,b,d in zip(SEEDS,values['combined'],values[key],pp)]))
            nets=[c-p-a+j for c,p,a,j in zip(values['combined'],values['path'],values['As'],values['joint'])];pp=[100*v/n for v in nets]
            interaction.append(dict(readout=readout,view=view,count_per_seed=n,interaction_correct_total=sum(nets),interaction_pp_mean=statistics.mean(pp),
                interaction_pp_sample_sd=statistics.stdev(pp),per_seed=[dict(model_seed=s,interaction_correct=a,interaction_pp=b) for s,a,b in zip(SEEDS,nets,pp)]))
    return dict(status='VERIFIED_COMPLETE_FACTORIAL',run_id=COMBINED_RUN,method=COMBINED_METHODS[0],model_seeds=list(SEEDS),
        matrices={k:dict(run_id=r,method=m,training_execution_commit=completions[k]['training_execution_commit']) for k,(r,m) in MATRICES.items()},
        neighbors=neighbors,interaction=interaction,independent_truth_last=True,
        primary_goal_unchanged='Raw clean>=+1pp and six strict>+1.5pp against both F20/F21;DURA separate;use complete primary target report to judge goal.',
        interpretation='Three paired F seeds on the same physical records;sample SD is across F seeds only.Six scenes repeat physical records;no IID/sample-significance claim.Interaction is combined-Path-As+Joint for fixed operational recipes,not proof of causal disentanglement.Joint target results were historically exposed before combination registration;not a new blind test.No target-driven parent/checkpoint/parameter choices.')


def markdown(r):
    text='# 固定四项配方比较\n\nJoint、主干互补、可靠As、组合版均完成三个seed及raw／同一DURA七景。下表为组合版减固定比较项，SD为三个配对模型seed的样本标准差。原F20/F21双目标仍由主报告判定。Joint历史目标已暴露，组合在其余三个方法目标分数读取前按源域证据固定，不能称新盲测。\n\n'
    text+='| 比较项 | 读出 | 场景 | 组合（%） | 比较项（%） | 差 ± SD（pp） | 正收益seed |\n|---|---|---|---:|---:|---:|---:|\n'
    for v in r['neighbors']:
        text+=f"| {v['reference_method']} | {v['readout']} | {v['view']} | {v['candidate_accuracy_percent']:.6f} | {v['reference_accuracy_percent']:.6f} | {v['paired_delta_pp_mean']:+.6f} ± {v['paired_delta_pp_sample_sd']:.6f} | {v['positive_seeds']}/3 |\n"
    text+='\n配方交互量定义为组合−Path−As＋Joint，只描述这四项实际配方的非加性，不作因果解耦或统计显著结论。六景复用物理样本，三个seed SD只跨F；相同G仍为用户固定DURA v1.0。\n\n| 读出 | 场景 | 交互量 ± SD（pp） |\n|---|---|---:|\n'
    for v in r['interaction']:text+=f"| {v['readout']} | {v['view']} | {v['interaction_pp_mean']:+.6f} ± {v['interaction_pp_sample_sd']:.6f} |\n"
    return text+'\n所有逐seed整数、运行标识及完整对比见factorial_comparison.json；完整F20/F21差值、逐TX/RX/day、修复损伤及成本仍见各自目标报告。\n'


def complete_report(report_root):
    r=analyze(report_root)
    if r['status']!='VERIFIED_COMPLETE_FACTORIAL':return r
    folder=Path(report_root)/COMBINED_RUN;names=('factorial_comparison.json','factorial_comparison.md')
    if any((folder/name).exists() for name in names):raise FileExistsError('Preserve existing full factorial evidence')
    (folder/names[0]).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (folder/names[1]).write_text(markdown(r),encoding='utf-8')
    return dict(status=r['status'],run_id=COMBINED_RUN,files=list(names))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',required=True);a=p.parse_args()
    print(json.dumps(complete_report(a.report_root)))
