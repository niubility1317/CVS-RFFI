"""Six fixed matched references; no target-driven choice or training."""
import argparse,json,math,statistics
from copy import deepcopy
from pathlib import Path
from . import design as d
from experiments.cvs_scratch_boundary_style import design as old
from experiments.cvs_scratch_calibrated_style.reference_report import _index

def build(current,reference):
    assert d.REFERENCE_RUN==old.RUN and len(d.REFERENCE_PAIRS)==6
    a,am=_index(current,d);b,bm=_index(reference,old);comparisons=[]
    for arm,control in d.REFERENCE_PAIRS:
        per=[]
        for seed in d.SEEDS:
            c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==seed))
            p=old.config(next(r for r in old.rows() if r['arm']==control and r['model_seed']==seed))
            for key in ('source_contract','reference_config','reference_checkpoint','batch_size','epochs','updates','checkpoint_sources','semi','losses','validation','checkpoint_selection','unlabeled','unlabeled_cache','preparation','model_initialization'):
                assert c[key]==p[key],key
            assert {k:v for k,v in c['teacher'].items() if k!='cache_root'}=={k:v for k,v in p['teacher'].items() if k!='cache_root'}
            assert {k:v for k,v in c['optimizer'].items() if k not in ('schedule','peak_lr')}=={k:v for k,v in p['optimizer'].items() if k not in ('schedule','peak_lr')}
            assert not c['multilevel']['enabled'] and not p['multilevel']['enabled']
            assert control==('L_REPLAY' if c['engineering']['hidden_mix'] else 'SSL')
            left=a.get((arm,seed));right=b.get((control,seed))
            per.append(dict(seed=seed,treatment_accuracy=left['accuracy'] if left else None,reference_accuracy=right['accuracy'] if right else None,difference=left['accuracy']-right['accuracy'] if left and right else None))
        diffs=[r['difference'] for r in per if r['difference'] is not None];complete=len(diffs)==3
        comparisons.append(dict(treatment=arm,reference=control,per_seed=per,paired_seed_count=len(diffs),complete_three_seeds=complete,mean_difference=statistics.mean(diffs) if complete else None,sd_difference=statistics.stdev(diffs) if complete else None))
    return dict(status='COMPLETE' if all(r['complete_three_seeds'] for r in comparisons) else 'PARTIAL',run_id=d.RUN,reference_run_id=old.RUN,comparisons=comparisons,expected_comparisons=6,complete_comparisons=sum(r['complete_three_seeds'] for r in comparisons),inputs=dict(current=am,reference=bm),IQ_read=False,truth_read=False,predictions_read=False,scores_control_training=False)

def markdown(v):
    fmt=lambda x:'N/A' if x is None else f'{100*x:.4f}'
    lines=['# 工程方案固定对照','',f"状态 {v['status']}，完整三种子比较 {v['complete_comparisons']}/6。只读取固定 run 的既有评分 JSON。",'', '|方案／固定参考|2701 新／原／差|2702 新／原／差|2703 新／原／差|均值差（百分点）|SD（百分点）|','|---|---|---|---|---:|---:|']
    for r in v['comparisons']:
        cells=['/'.join(fmt(x[k]) for k in ('treatment_accuracy','reference_accuracy','difference')) for x in r['per_seed']]
        lines.append('|'+r['treatment']+'／'+r['reference']+'|'+'|'.join(cells)+'|'+fmt(r['mean_difference'])+'|'+fmt(r['sd_difference'])+'|')
    lines+=['','所有学生随机训练 20 轮，同样合法 L/U、固定源教师和 AdamW。五个方案对照原 SSL；hidden mix 对照原 L_REPLAY，均有 3996 次额外 L 前向及 clean Dropout 重放。新方案只改变预登记的学习率、SAM、L 标签平滑或混合特征及软标签。','SAM 每步多一遍目标与反向，更新次数仍 4440，实际样本模型读出与成本单列。相同 epoch 不代表相同算力。跨 run 同 seed 不保证 GPU 权重逐位一致，不选择性重跑。']
    return '\n'.join(lines)+'\n'

def self_check():
    def fixture(design,delta):
        records=[dict(**r,dimension='overall',stage=design.config(r)['stage'],view='clean',stratum='ALL',query_count=168000,accuracy=.7+.01*design.SEEDS.index(r['model_seed'])+delta) for r in design.rows()]
        return dict(run_id=design.RUN,status='COMPLETE',completed_rows=[r['row_id'] for r in design.rows()],missing_rows=[],records=records,target_feedback_to_training=False,checks=dict(source_config_budget=True,confusion_recount=True))
    current=fixture(d,.025);reference=fixture(old,0);v=build(current,reference)
    assert v['status']=='COMPLETE' and len(v['comparisons'])==6 and all(math.isclose(r['mean_difference'],.025,abs_tol=1e-12) for r in v['comparisons'])
    partial=deepcopy(current);rid='SSL_SAM-s'+str(d.SEEDS[1]);partial['status']='PARTIAL';partial['completed_rows'].remove(rid);partial['missing_rows']=[rid];partial['records']=[r for r in partial['records'] if r['row_id']!=rid]
    v=build(partial,reference);assert v['status']=='PARTIAL' and v['complete_comparisons']==5
    assert next(r for r in v['comparisons'] if r['treatment']=='SSL_SAM')['mean_difference'] is None
    for key,value in [('run_id','bad'),('target_feedback_to_training',True)]:
        bad=deepcopy(current);bad[key]=value
        try:build(bad,reference)
        except ValueError:pass
        else:raise AssertionError('Invalid score input accepted')
    return dict(status='PASS',synthetic_only=True,comparisons=6,partial_mean_none=True,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:print(json.dumps(self_check()));raise SystemExit
    if a.output is None:p.error('--output required')
    def load(f):return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None
    v=build(load(a.output/'evidence/test_results_analysis.json'),load(a.output.parent/old.RUN/'evidence/test_results_analysis.json'))
    (a.output/'reference_comparisons.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(a.output/'reference_comparisons.md').write_text(markdown(v),encoding='utf-8')
    print(json.dumps({k:v[k] for k in ('status','complete_comparisons','expected_comparisons')}))
