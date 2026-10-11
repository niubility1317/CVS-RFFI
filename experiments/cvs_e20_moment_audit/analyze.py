import argparse,statistics as st,csv
from pathlib import Path
from . import design as d
def analyze(v):
 if v['status']!='VERIFIED' or len(v['rows'])!=3:raise ValueError('Complete validated source matrix required')
 fishr=[];style=[];references=[]
 for row in v['rows']:
  seed=row['row']['model_seed']
  for model,data in row['models'].items():
   for level,values in data['levels'].items():
    for ref in values['fishr']['references']:references.append(dict(seed=seed,model=model,level=level,**ref))
    for r in values['fishr']['rounds']:
     for method,value in r['losses'].items():
      fishr.append(dict(seed=seed,model=model,level=level,control=r['control'],day=r['day'],m=r['m'],round=r['round'],method=method,
       exact_loss=r['exact_loss'],loss=value['loss'],error=value['error_vs_exact'],squared_error=value['squared_error_vs_exact'],
       gradient_cosine=value['gradient_vs_exact']['cosine'],gradient_squared_error=value['gradient_squared_error_vs_exact'],
       cosine_to_probe_CE=value['gradient_vs_full_L_CE']['cosine'],overlap=r['cross_batch_PID_overlap']))
    style.append(dict(seed=seed,model=model,level=level,**values['style']['summary']))
 assert len(fishr)==7776 and len(style)==12 and len(references)==72
 groups=[]
 for model in ('P0','CE20'):
  for level in ('time_down','t2.pool'):
   for ctrl in ('original_RX','within_TX_day_shuffled_RX'):
    for m in (2,8,16):
     for method in ('old_pooled_n_square','stratified_unbiased_square','independent_AB_cross'):
      rs=[r for r in fishr if (r['model'],r['level'],r['control'],r['m'],r['method'])==(model,level,ctrl,m,method)]
      assert len(rs)==108
      exact=st.mean(r['exact_loss'] for r in rs)
      groups.append(dict(model=model,level=level,control=ctrl,m=m,method=method,records=len(rs),exact=exact,
       mean=st.mean(r['loss'] for r in rs),ratio_of_means=st.mean(r['loss'] for r in rs)/exact if exact else None,
       bias=st.mean(r['error'] for r in rs),RMSE=st.mean(r['squared_error'] for r in rs)**.5,
       gradient_cosine_mean=st.mean(r['gradient_cosine'] for r in rs if r['gradient_cosine'] is not None),
       negative_loss_fraction=sum(r['loss']<0 for r in rs)/len(rs),negative_gradient_fraction=sum(r['gradient_cosine'] is not None and r['gradient_cosine']<0 for r in rs)/len(rs)))
 return dict(status='ANALYZED',source_only=True,target_access=False,identity_updates=0,head_updates=1200,L_model_readouts=75600,groups=groups,style=style,references=references,fishr_records=fishr)
def report(a):
 lines=['# 浅中层有限样本与风格统计：完整源诊断','','三种子、原 E20 与固定 CE＋20 均已完成；身份骨干更新为 0。全部固定 L_s 参与总体参考与统计计算，没有 U/V 前向或目标访问。本实验不产生新 clean 测试成绩，不能用于宣称 84% 达成。','','## Fishr 估计误差与方向','','每个单元含 3 seed×3 day×12 round=108 次估计。ratio 是估计值均值／全 L 参考均值，不是逐样本比值平均。方向只涉及共享虚拟通道仿射变量，并非完整身份骨干。跨批估计允许负值。','','|模型|层|RX分组|每TX/RX样本|方法|估计／参考|RMSE|对参考梯度cos|负loss比例|负梯度比例|','|---|---|---|---:|---|---:|---:|---:|---:|---:|']
 short={'old_pooled_n_square':'旧平方','stratified_unbiased_square':'分层修正平方','independent_AB_cross':'独立AB内积'}
 for r in a['groups']:
  ratio='N/A' if r['ratio_of_means'] is None else f"{r['ratio_of_means']:.3f}"
  lines.append(f"|{r['model']}|{r['level']}|{'真实' if r['control']=='original_RX' else '类内打乱'}|{r['m']}|{short[r['method']]}|{ratio}|{r['RMSE']:.3e}|{r['gradient_cosine_mean']:.3f}|{r['negative_loss_fraction']:.1%}|{r['negative_gradient_fraction']:.1%}|")
 lines+=['','修正方差后直接平方仍含估计噪声；独立内积的期望正确不等于有限轮次误差必然较小。不能按一次负值挑选估计器，也不能把虚拟仿射的方向直接称为真实骨干更新方向。固定参数下的无偏结论不自动适用于由同批梯度范数决定的随机自适应权重。','','## MixStyle 与条件 RX 统计','','每行是一个 seed、模型和层，完整 6300 L 上的 18 个 TX×day 条件格等权聚合；扰动是固定 36 次 B60 的局部缓存特征运算，不是沿主模型连续执行两个 hook 的端到端误差。','','|模型|seed|层|均值RX间/内|logσ RX间/内|实际改变比例|相对特征变化|','|---|---:|---|---:|---:|---:|---:|']
 for r in a['style']:
  m=r['moments'];mu=m['mu']['ratio_of_mean_between_to_mean_within'];sd=m['log_sigma']['ratio_of_mean_between_to_mean_within']
  fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
  lines.append(f"|{r['model']}|{r['seed']}|{r['level']}|{fmt(mu)}|{fmt(sd)}|{r['changed_ratio']:.2%}|{r['feature_relative_l2_mean']:.2%}|")
 lines+=['','RX 间关联仍可能混合内容、接收条件和样本差异，不是孤立硬件因果效应。相关坐标可能承载 TX 身份，因此不能仅凭 RX 可分性删除。样本噪声估计依赖 cell 内观测独立假设；真实无线包可能相关，完整字段与计数见原始源诊断 JSON。','','## 证据与成本','','三个 worker，共六个固定模型、12 个模型层、2592 个独立批对、7776 个估计器读数、432 次局部 MixStyle 试验。共 1200 次辅助头更新、75600 次 L 样本模型读出；真实模型基础状态在前后逐位不变。全部读出与异常检查见 evidence/source_results.json；Fishr 逐轮平表见 evidence/fishr_rounds.csv。没有部署 checkpoint、目标预测、目标评分或目标驱动选择。']
 return '\n'.join(lines)+'\n'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();a=analyze(d.read(args.output/'evidence/source_results.json'))
 d.write(args.output/'summary.json',a);(args.output/'analysis.md').write_text(report(a),encoding='utf-8')
 with (args.output/'evidence/fishr_rounds.csv').open('w',encoding='utf-8',newline='') as stream:
  writer=csv.DictWriter(stream,list(a['fishr_records'][0]));writer.writeheader();writer.writerows(a['fishr_records'])
 print('ANALYZED 3 seeds / 6 models / 7776 Fishr estimates / 432 style rounds')
