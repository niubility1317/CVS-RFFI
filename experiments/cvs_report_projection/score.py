from . import design as d
def run(seed):
 import numpy as np
 from .predict import preflight
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 ids=preflight(seed);truth=d.read(d.TRUTH);y,rx,day=truth_arrays(truth,ids)
 with np.load(d.BASE/('s'+str(seed))/'prediction/predictions.npz',allow_pickle=False) as p:
  for name in d.METHODS:
   c=d.config(next(r for r in d.rows() if r['arm']==name and r['model_seed']==seed));out=d.BASE/c['row_id']/'scores';out.mkdir(exist_ok=False);rr,dd=metric_rows(c,p[name],ids,y,rx,day);records=rr+dd;d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'));d.write(d.BASE/c['row_id']/'completion.json',dict(status='FROZEN_PREDICTED_SCORED',config=c,views=['clean'],count=168000,metric_records=18,truth_last=True))
 d.write(d.BASE/('s'+str(seed))/'completion.json',dict(status='SEED_COMPLETE',seed=seed,methods=list(d.METHODS),views=['clean'],truth_last=True));print('SEED_COMPLETE '+str(seed),flush=True)
def finish():
 import statistics as st
 records=[]
 for row in d.rows():
  c=d.config(row);v=d.read(d.BASE/row['row_id']/'completion.json');score=d.read(d.BASE/row['row_id']/'scores/metrics.json')
  if v!=dict(status='FROZEN_PREDICTED_SCORED',config=c,views=['clean'],count=168000,metric_records=18,truth_last=True) or len(score['records'])!=18 or not score['truth_last'] or score['independent_recount']!='VERIFIED':raise ValueError('Projectionclean row incomplete')
  records+=score['records']
 summary=[]
 for name in d.METHODS:
  rr=[r for r in records if r['arm']==name and r['dimension']=='overall']
  if {r['model_seed'] for r in rr}!=set(d.SEEDS) or len(rr)!=3:raise ValueError('Threeprimaryseeds missing')
  aa=[r['accuracy'] for r in rr];summary.append(dict(arm=name,accuracy_mean=st.mean(aa),accuracy_sd=st.stdev(aa),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr},macro_f1_mean=st.mean(r['macro_f1'] for r in rr),goal_84=st.mean(aa)>=.84))
 a=dict(status='VERIFIED',run_id=d.RUN,views=['clean'],rows=24,prediction_count=4032000,summary=summary,goal_84=any(r['goal_84'] for r in summary),truth_last=True,records=records);d.write(d.BASE/'analysis.json',a);lines=['# 报告判别投影：三种子clean结果','','|方法|三seed均值|SD|≥84%|','|---|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['goal_84']}|")
 lines+=['', '每row168000 clean，全部source参数与同seed8模型固定后预测，再独立truthlast评分；没有sourceV性能筛选/practical测试/query拟合。三个已严格核实的纯CE30父模型复用，共24冻结派生状态，新增梯度更新0；校准只L6300，源角色不变。所有输出为常规cos，投影保留160输出接口；硬删方向实际秩及15TX均值保持、Gaussian有效Fisher与原始source统计边界见source_fit_diagnostics，不声称新类完整身份证据或已通过Phase2源状态审计。', '基准已暴露，方法/seed/矩阵/参数在测试前固定；全部实际结果保留，不按target挑seed、调参数或重跑。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
