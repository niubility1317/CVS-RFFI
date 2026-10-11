"""Fixed cross-run comparisons of existing scored JSON; never opens IQ/truth."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
from . import design as d
from experiments.cvs_scratch_boundary_style import design as old

COUNT=168000


def _index(value,design):
    expected={r['row_id']:r for r in design.rows()}
    if value is None:return {},dict(status='MISSING',run_id=design.RUN,missing_rows=list(expected))
    if value.get('run_id')!=design.RUN:raise ValueError('Comparison input has a different registered run')
    if value.get('status') not in ('COMPLETE','PARTIAL'):raise ValueError('Only scored complete/partial analysis is supported')
    if (value.get('target_feedback_to_training') is not False
            or value.get('checks',{}).get('source_config_budget') is not True
            or value.get('checks',{}).get('confusion_recount') is not True):
        raise ValueError('Existing score lacks source-config/budget or metric verification')
    completed=value['completed_rows']
    if len(set(completed))!=len(completed) or not set(completed)<=set(expected):
        raise ValueError('Completed rows are duplicated or unregistered')
    result={}
    for record in value['records']:
        if record['dimension']!='overall':continue
        rid=record['row_id']
        if rid not in expected or rid not in completed:raise ValueError('Score row is not a completed registered row')
        row=expected[rid];c=design.config(row);design.validate(c)
        if (record['arm']!=row['arm'] or type(record['model_seed']) is not int
                or record['model_seed']!=row['model_seed'] or record['stage']!=c['stage']
                or record['view']!='clean' or record['stratum']!='ALL' or record['query_count']!=COUNT):
            raise ValueError('Score identity, seed, clean view, or query count differs')
        accuracy=record['accuracy']
        if isinstance(accuracy,bool) or not isinstance(accuracy,(int,float)) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
            raise ValueError('Invalid stored accuracy')
        key=(row['arm'],row['model_seed'])
        if key in result:raise ValueError('Duplicate overall accuracy')
        result[key]=dict(accuracy=accuracy,row_id=rid,query_count=COUNT)
    if {r['row_id'] for r in result.values()}!=set(completed):raise ValueError('Completed row is missing its overall score')
    missing=set(expected)-set(completed)
    if set(value['missing_rows'])!=missing:raise ValueError('Missing-row list differs from actual scored coverage')
    if value['status']=='COMPLETE' and missing:raise ValueError('Incomplete matrix incorrectly marked complete')
    return result,dict(status=value['status'],run_id=design.RUN,completed_rows=completed,missing_rows=sorted(missing),
        actual_config_verification='Existing collector source_config_budget=true; collector checked source/freeze against design',
        local_config_verification='Reconstructed each scored row from the registered local design and validated it')


def _correspondence(treatment,control,seed):
    if d.REFERENCE_RUN!=old.RUN or tuple(d.SEEDS)!=tuple(old.SEEDS) or d.CAPSULE!=old.CAPSULE:
        raise ValueError('Fixed comparison reference or observation contract differs')
    row=next(r for r in d.rows() if r['arm']==treatment and r['model_seed']==seed)
    ref=next(r for r in old.rows() if r['arm']==control and r['model_seed']==seed)
    c=d.config(row);prior=old.config(ref);d.validate(c);old.validate(prior)
    if control not in (c['base_arm'],'CE','SSL'):raise ValueError('Unregistered method/control correspondence')
    for config in (c,prior):
        if (config['model_initialization']!='scratch' or config['parent_checkpoint'] is not None
                or config['epochs']!=20 or config['steps_per_epoch']!=222 or config['updates']!=4440
                or config['evaluation']['truth_last'] is not True or config['evaluation']['feedback_to_training']):
            raise ValueError('Reference must use the fixed scratch20 truth-last protocol')
    for key in ('source_contract','reference_config','reference_checkpoint','batch_size','optimizer'):
        if c[key]!=prior[key]:raise ValueError('Registered comparable training basis differs: '+key)
    return c,prior


def build(current,reference):
    expected=[(arm,arm.split('_',1)[1]) for arm in d.ARMS]+[(arm,base) for arm in d.ARMS for base in ('CE','SSL')]
    if list(d.REFERENCE_PAIRS)!=expected or len(expected)!=36 or len(set(expected))!=36:
        raise ValueError('Expected all thirty-six preregistered comparisons')
    live,live_meta=_index(current,d);prior,prior_meta=_index(reference,old)
    comparisons=[]
    for treatment,control in d.REFERENCE_PAIRS:
        seeds=[];values=[]
        for seed in d.SEEDS:
            _correspondence(treatment,control,seed)
            a=live.get((treatment,seed));b=prior.get((control,seed))
            delta=a['accuracy']-b['accuracy'] if a and b else None
            values.append(dict(seed=seed,treatment_accuracy=a['accuracy'] if a else None,
                reference_accuracy=b['accuracy'] if b else None,difference=delta,
                status='COMPLETE' if delta is not None else 'MISSING',
                treatment_row=treatment+'-s'+str(seed),reference_row=control+'-s'+str(seed)))
            if delta is not None:seeds.append(seed)
        differences=[v['difference'] for v in values if v['difference'] is not None]
        complete=len(seeds)==3
        comparisons.append(dict(treatment=treatment,reference=control,metric='accuracy',view='clean',
            status='COMPLETE' if complete else 'PARTIAL',complete_three_seeds=complete,
            paired_seeds=seeds,paired_seed_count=len(seeds),per_seed=values,
            mean_difference=statistics.mean(differences) if complete else None,
            sd_difference=statistics.stdev(differences) if complete else None,
            available_pair_mean=statistics.mean(differences) if differences else None,
            available_pair_sd=statistics.stdev(differences) if len(differences)>1 else None))
    complete=all(row['complete_three_seeds'] for row in comparisons)
    return dict(schema='scratch_calibrated_reference_comparisons_v1',status='COMPLETE' if complete else 'PARTIAL',
        run_id=d.RUN,reference_run_id=d.REFERENCE_RUN,query_count_per_row=COUNT,view='clean',
        seeds=list(d.SEEDS),expected_comparisons=36,complete_comparisons=sum(r['complete_three_seeds'] for r in comparisons),
        inputs=dict(current=live_meta,reference=prior_meta),comparisons=comparisons,
        score_JSON_only=True,IQ_read=False,truth_read=False,predictions_read=False,
        scores_control_training=False,configuration_selection=False,rerun_selection=False,
        configuration_evidence_scope='Local registered design plus existing collector config verification; no source artifact re-read',
        cost_scope='Both students scratch20; non-CE methods retain two pretrained E20 teachers (60 historical model epochs); CE is20')


def markdown(value):
    def percent(v):return 'N/A' if v is None else f'{v*100:.4f}%'
    def pp(v):return 'N/A' if v is None else f'{v*100:+.4f}'
    lines=['# 固定前批对照：三种子配对结果','',
        f"状态：{value['status']}；完整比较 {value['complete_comparisons']}/36。新批 `{value['run_id']}`，参考批 `{value['reference_run_id']}`。",'',
        '每个方法预先固定与对应原方法、CE、SSL 比较；每行使用相同 168,000 条 clean 测试样本。只读取已有评分 JSON，不读取 IQ、预测或真值。学生均从零训练 20 轮；非 CE 方法仍有两教师的历史训练成本，不能称为整个方法无预训练。', '',
        '逐行身份、种子、场景和预算映射依据本地登记设计核验；实际 source/freeze 配置一致性引用已有 collector 的 source_config_budget 核验。本工具不重新读取训练配置产物。', '',
        '|新方法 − 前批参考|2701：新／参考／差（百分点）|2702：新／参考／差（百分点）|2703：新／参考／差（百分点）|完整配对|三种子均值差（百分点）|SD（百分点）|状态|',
        '|---|---|---|---|---:|---:|---:|---|']
    for row in value['comparisons']:
        cells=[percent(v['treatment_accuracy'])+'／'+percent(v['reference_accuracy'])+'／'+pp(v['difference']) for v in row['per_seed']]
        sd='N/A' if row['sd_difference'] is None else f"{100*row['sd_difference']:.4f}"
        lines.append('|'+row['treatment']+' − '+row['reference']+'|'+'|'.join(cells)+f"|{row['paired_seed_count']}/3|{pp(row['mean_difference'])}|{sd}|{row['status']}|")
    lines+=['','缺少任何配对种子时，完整三种子均值和 SD 显示 N/A；JSON 另外保留现有配对的描述性均值及实际数量。缺项不补值，不触发选择性重跑。差值与三个种子的 SD 是描述性结果，不自动构成显著性证明；全部 36 项保留，不据成绩调参或改变队列。']
    return '\n'.join(lines)+'\n'


def self_check():
    def fixture(design,delta):
        rows=design.rows();records=[]
        for row in rows:
            config=design.config(row);i=design.SEEDS.index(row['model_seed'])
            records.append(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],stage=config['stage'],
                dimension='overall',stratum='ALL',view='clean',query_count=COUNT,accuracy=.70+i*.01+delta))
        return dict(run_id=design.RUN,status='COMPLETE',completed_rows=[r['row_id'] for r in rows],missing_rows=[],records=records,
            target_feedback_to_training=False,checks=dict(source_config_budget=True,confusion_recount=True))
    current=fixture(d,.025);reference=fixture(old,0.)
    full=build(current,reference)
    assert full['status']=='COMPLETE' and len(full['comparisons'])==36
    assert all(math.isclose(r['mean_difference'],.025,abs_tol=1e-12) and r['sd_difference']==0 for r in full['comparisons'])
    partial=deepcopy(reference);rid='M_BOTH-s'+str(d.SEEDS[1])
    partial['status']='PARTIAL';partial['completed_rows'].remove(rid);partial['missing_rows']=[rid]
    partial['records']=[r for r in partial['records'] if r['row_id']!=rid]
    missing=build(current,partial)
    affected=[r for r in missing['comparisons'] if r['reference']=='M_BOTH']
    assert missing['status']=='PARTIAL' and len(affected)==3
    assert all(r['paired_seed_count']==2 and r['mean_difference'] is None and r['sd_difference'] is None for r in affected)
    assert 'N/A' in markdown(missing)
    for key,bad in [('run_id','unregistered'),('target_feedback_to_training',True)]:
        value=deepcopy(current);value[key]=bad
        try:build(value,reference)
        except ValueError:pass
        else:raise AssertionError('Bad comparison input accepted')
    bad=deepcopy(current);bad['records'][0]['query_count']=167999
    try:build(bad,reference)
    except ValueError:pass
    else:raise AssertionError('Wrong query count accepted')
    assert build(None,reference)['status']=='PARTIAL'
    return dict(status='PASS',synthetic_only=True,complete_comparisons=36,missing_one_seed_affected_comparisons=3,
        complete_mean=.025,missing_complete_mean=None,run_and_query_negatives_rejected=True,target_artifacts_read=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    parser.add_argument('--reference-output',type=Path);parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),allow_nan=False));return
    if args.output is None:parser.error('--output is required unless --self-check')
    reference=args.reference_output or args.output.parent/d.REFERENCE_RUN
    def load(root):
        path=root/'evidence/test_results_analysis.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    value=build(load(args.output),load(reference))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'reference_comparisons.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (args.output/'reference_comparisons.md').write_text(markdown(value),encoding='utf-8')
    print(json.dumps(dict(status=value['status'],complete_comparisons=value['complete_comparisons'],expected_comparisons=36)))


if __name__=='__main__':main()
