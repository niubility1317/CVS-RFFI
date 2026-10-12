"""The two preregistered raw-energy vs normalized-margin JOINT contrasts."""
import argparse,json,math,statistics
from copy import deepcopy
from pathlib import Path
from . import design as d
from experiments.cvs_scratch_identity_style import design as old
from experiments.cvs_scratch_calibrated_style.reference_report import _index

def build(current,reference):
    assert d.REFERENCE_RUN==old.RUN and d.REFERENCE_PAIRS==[(arm,'DELAYCAP_JOINT') for arm in d.ARMS]
    a,am=_index(current,d);b,bm=_index(reference,old)
    comparisons=[]
    for arm,control in d.REFERENCE_PAIRS:
        per=[]
        for seed in d.SEEDS:
            row=next(r for r in d.rows() if r['arm']==arm and r['model_seed']==seed)
            c=d.config(row);prior=old.config(next(r for r in old.rows() if r['arm']==control and r['model_seed']==seed))
            mc=deepcopy(c['multilevel'])
            for key in ('sensitivity_objective','sensitivity_normalization','scale_calibration_pairs','scale_calibration_readouts','calibration_total_forwards','calibration_total_readouts','scale_after_identity_updates'):mc.pop(key)
            mc['statistic']=prior['multilevel']['statistic'];mc['fishr_estimator']=prior['multilevel']['fishr_estimator']
            assert mc==prior['multilevel']
            for key in ('source_contract','reference_config','reference_checkpoint','batch_size','optimizer','epochs','updates','checkpoint_sources','semi'):
                assert c[key]==prior[key],key
            left=a.get((arm,seed));right=b.get((control,seed))
            per.append(dict(seed=seed,treatment_accuracy=left['accuracy'] if left else None,reference_accuracy=right['accuracy'] if right else None,difference=left['accuracy']-right['accuracy'] if left and right else None))
        diffs=[r['difference'] for r in per if r['difference'] is not None];complete=len(diffs)==3
        comparisons.append(dict(treatment=arm,reference=control,per_seed=per,paired_seed_count=len(diffs),complete_three_seeds=complete,mean_difference=statistics.mean(diffs) if complete else None,sd_difference=statistics.stdev(diffs) if complete else None))
    return dict(status='COMPLETE' if all(r['complete_three_seeds'] for r in comparisons) else 'PARTIAL',run_id=d.RUN,reference_run_id=old.RUN,comparisons=comparisons,expected_comparisons=2,complete_comparisons=sum(r['complete_three_seeds'] for r in comparisons),inputs=dict(current=am,reference=bm),IQ_read=False,truth_read=False,predictions_read=False,scores_control_training=False)

def markdown(v):
    fmt=lambda x:'N/A' if x is None else f'{100*x:.4f}'
    lines=['# 幅度敏感度的固定对照','',f"状态{v['status']}，完整三种子比较{v['complete_comparisons']}/2。只读取两个固定run的既有评分JSON，不读取IQ或truth。",'', '|敏感度目标|2701新/原/差（百分点）|2702新/原/差|2703新/原/差|三种子均值差（百分点）|SD（百分点）|','|---|---|---|---|---:|---:|']
    for r in v['comparisons']:
        cells=['/'.join(fmt(x[k]) for k in ('treatment_accuracy','reference_accuracy','difference')) for x in r['per_seed']]
        lines.append('|'+r['treatment']+'|'+'|'.join(cells)+'|'+fmt(r['mean_difference'])+'|'+fmt(r['sd_difference'])+'|')
    lines+=['','所有学生均从零训练20轮，复用另外两个原E20源域教师。原对照对每个样本的 margin 梯度分别归一化；新方案保留 margin/CE 原始幅度，只除以每层固定源 L 尺度。新方案增加一遍尺度校准（24 前向、1440 次 L 读出），主训练预算和困难风格相同。种子不齐时三种子均值/SD为N/A。','同seed不保证共享完全相同的预热权重，已知GPU数值非确定性必须保留；不把跨run差值称为固定相同权重的因果实验。','纯CE原E20基线P0及逐RX/TX/day结果由本run独立评分器同时报告。同预算scratch20的CE/SSL完整对照保留在前批20261012-phase1-scratch-boundary-style-manysig-m27-r01；本表报告预登记的幅度敏感度对照；额外校准成本单列。']
    return '\n'.join(lines)+'\n'

def self_check():
    def fixture(design,delta):
        records=[]
        for r in design.rows():
            records.append(dict(**r,dimension='overall',stage=design.config(r)['stage'],view='clean',stratum='ALL',query_count=168000,accuracy=.7+.01*design.SEEDS.index(r['model_seed'])+delta))
        return dict(run_id=design.RUN,status='COMPLETE',completed_rows=[r['row_id'] for r in design.rows()],missing_rows=[],records=records,target_feedback_to_training=False,checks=dict(source_config_budget=True,confusion_recount=True))
    current=fixture(d,.025);reference=fixture(old,0)
    v=build(current,reference);assert v['status']=='COMPLETE' and len(v['comparisons'])==2
    assert all(math.isclose(r['mean_difference'],.025,abs_tol=1e-12) for r in v['comparisons'])
    partial=deepcopy(current);rid='CE_ENERGY-s'+str(d.SEEDS[1]);partial['status']='PARTIAL';partial['completed_rows'].remove(rid);partial['missing_rows']=[rid];partial['records']=[r for r in partial['records'] if r['row_id']!=rid]
    v=build(partial,reference);assert v['status']=='PARTIAL' and v['complete_comparisons']==1
    assert next(r for r in v['comparisons'] if r['treatment']=='CE_ENERGY')['mean_difference'] is None
    for key,value in [('run_id','bad'),('target_feedback_to_training',True)]:
        bad=deepcopy(current);bad[key]=value
        try:build(bad,reference)
        except ValueError:pass
        else:raise AssertionError('Invalid score input accepted')
    return dict(status='PASS',synthetic_only=True,comparisons=2,partial_mean_none=True,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:print(json.dumps(self_check()));raise SystemExit
    if a.output is None:p.error('--output required')
    def load(f):return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None
    v=build(load(a.output/'evidence/test_results_analysis.json'),load(a.output.parent/old.RUN/'evidence/test_results_analysis.json'))
    (a.output/'reference_comparisons.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.output/'reference_comparisons.md').write_text(markdown(v),encoding='utf-8')
    print(json.dumps({k:v[k] for k in ('status','complete_comparisons','expected_comparisons')}))
