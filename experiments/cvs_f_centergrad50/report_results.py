"""Parse every native and auxiliary step; curate all fixed negative/positive rows."""
import gzip,json,math
from pathlib import Path
from . import design as d
from experiments.cvs_f_crossrx50_local.report_results import training_logs
def contrastive_logs(base,row):
 records=[];details=[]
 with gzip.open(Path(base)/row['row_id']/'source/supcon_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   v=json.loads(line);expected=len(records)+1
   if v['step']!=expected or v['epoch']!=(expected-1)//222+1 or v['weight']!=.1*min(max((v['epoch']-5)/10.,0.),1.) or v['statistic_policy']!=row['treatment']:raise ValueError('Ordered native SC policy/schedule differs')
   if not math.isfinite(v['loss']) or v['extra_model_forwards'] or v['head_fit_updates'] or v['U_truth_read']:raise ValueError('Invalid SC execution')
   records.append(v);details.extend(v['details'])
 if len(records)!=11100 or len(details)!=39960 or sum(bool(v['details']) for v in records)!=9990:raise ValueError('Full auxiliary budget differs')
 for v in details:
  b=v['backward']
  if b['backward_calls']!=1 or b['mode']!=row['treatment'] or any(not math.isfinite(b[k]) or b[k]<0 for k in ('native_descriptor_grad_norm','policy_descriptor_grad_norm','delivered_descriptor_grad_norm','norm_ratio','translation_removed_norm','scale_removed_norm')):raise ValueError('Missing/nonfinite actual statistic derivative')
  if abs(b['delivered_descriptor_grad_norm']-b['policy_descriptor_grad_norm'])>1e-5*max(1.,b['policy_descriptor_grad_norm']):raise ValueError('Applied mean derivative norm differs')
 return dict(row_id=row['row_id'],parsed_auxiliary_steps=11100,active_steps=9990,layer_forward_backward_objectives=39960,policy=row['treatment'],full_auxiliary_log_read=True,gradient_norm_ratio_mean=sum(v['backward']['norm_ratio'] for v in details)/len(details),gradient_norm_ratio_min=min(v['backward']['norm_ratio'] for v in details),gradient_norm_ratio_max=max(v['backward']['norm_ratio'] for v in details),extra_model_forwards=0,head_fit_updates=0)
def publish(base,output):
 base=Path(base);output=Path(output);result=d.read(base/'result.json')
 if result['run_id']!=d.RUN or len(result['records'])!=3:raise ValueError('Complete fixed three-row results required')
 summary=dict(result=result,training_logs=[training_logs(base,r['row_id']) for r in d.rows()],contrastive_logs=[contrastive_logs(base,r) for r in d.rows()])
 lines=['# F跨RX：均值梯度消融的三种子结果','', '各行从零50轮（33L＋17U），默认分类头正常联合训练、RDrop1、旧门控/ownEMA/cos、LS.01与普通增强不变。无星地增强，只测试相同168000条clean样本。预测固定后独立CPU truth-last评分；三个模型种子不是三批独立物理测试样本。','', '|方法|seed|clean%|BASE%|整体差/百分点|14-7差|20-19差|弱TX均差|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r,p in zip(result['records'],result['paired']):lines.append('|'+r['treatment']+'|'+str(r['model_seed'])+'|'+format(100*r['current']['accuracy'],'.4f')+'|'+format(100*r['reused_BASE']['accuracy'],'.4f')+'|'+format(p['accuracy_pp'],'+.4f')+'|'+format(p['per_TX_pp'][1],'+.4f')+'|'+format(p['per_TX_pp'][3],'+.4f')+'|'+format(p['weak_TX_mean_pp'],'+.4f')+'|')
 lines+=['','全部逐seed配对差、均值、样本标准差、正差数、Macro-F1、最差RX/day、逐TX及成本见results_summary.json。与事先固定STAT_GRAD/NORM_ONLY的配对在对应三种子齐全时另存secondary_comparisons，不按目标分数择优重排。','', '固定科学判断：'+json.dumps(result['scientific_verdict'],ensure_ascii=False),'', 'CENTER_GRAD在原描述子上仅保留当前均值的导数，当前标准差仍停止梯度；其辅助描述子梯度为原梯度减批均值。它保留原来的尺度方向响应，同输入前向/掩码/温度/损失值/默认头完全相同。不声称完整重算标准差前向的总导数，也没有梯度范数匹配；与完整统计梯度的比较同时改变梯度方向和大小。无部署BN、运行统计或额外网络前向。','', '目标基准已暴露，本次是事先固定全部三种子及final50的探索确认，不声称新的独立盲测、Phase2迁移或卫星性能。负结果全部保留；源V不做梯度、另训头或目标反馈。']
 output.mkdir(parents=True,exist_ok=False);d.write(output/'results_summary.json',summary);(output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf8');return summary
