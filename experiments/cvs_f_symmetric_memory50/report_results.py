"""Full fixed logs and independent clean results, all negative rows retained."""
import argparse,gzip,json,math
from pathlib import Path
from . import design as d
from .analysis import stats
from experiments.cvs_f_crossrx50_local.report_results import training_logs
from experiments.cvs_f_supcon50.mechanism import schedule

def contrastive_logs(base,row):
 root=Path(base)/row/'source';o=d.read(root/'crossbatch_training_observation.json');calls=applied=valid=objectives=history=0;maxage=0;pern=[0]*6;perp=[0.]*6;perneg=[0.]*6;curves=[]
 with gzip.open(root/'supcon_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);calls+=1;epoch=(calls-1)//222+1
   if r['step']!=calls or r['epoch']!=epoch or r['weight']!=schedule(epoch) or not math.isfinite(r['loss']) or r['mode']!='CROSS_RX' or r['contrastive_mode']!=o['treatment'] or len(r['details'])!=(4 if epoch>5 else 0):raise ValueError('Full fixed memory records required')
   if len(curves)<epoch:curves.append(dict(epoch=epoch,calls=0,loss_sum=0.,valid_anchors=0,historical_keys=0))
   curve=curves[-1];curve['calls']+=1;curve['loss_sum']+=r['loss'];applied+=bool(r['details'])
   for v in r['details']:
    if v['mode']!=o['treatment'] or v['level'] not in ('time_down','t2.pool') or v['stage'] not in ('first','second') or v['total_anchors']!=128 or not 0<=v['valid_anchors']<=128 or not math.isfinite(v['loss']) or not 0<=v['historical_keys']<=384 or v['key_count']!=128+v['historical_keys'] or not 0<=v['max_history_age']<=3:raise ValueError('Source memory keys/count/age differ')
    if o['treatment']=='ANCHOR' and v['historical_keys']:raise ValueError('Current control read historical keys')
    nv=v['valid_anchors'];valid+=nv;objectives+=1;history+=v['historical_keys'];maxage=max(maxage,v['max_history_age']);curve['valid_anchors']+=nv;curve['historical_keys']+=v['historical_keys']
    if len(v['per_TX_keys'])!=6 or sum(t['count'] for t in v['per_TX_keys'])!=nv:raise ValueError('Complete source class key counts required')
    for c,t in enumerate(v['per_TX_keys']):
     if t['TX']!=c or t['count']<0 or (t['count'] and (t['positive_mean']<=0 or t['negative_mean']<=0)):raise ValueError('Invalid perTX key count')
     pern[c]+=t['count'];perp[c]+=t['count']*(t['positive_mean'] or 0.);perneg[c]+=t['count']*(t['negative_mean'] or 0.)
 if (calls,applied,objectives,valid)!=(11100,9990,39960,o['valid_anchors']) or abs(history/objectives-o['mean_historical_keys'])>1e-10 or maxage!=o['max_history_age']:raise ValueError('Allstep memory counters differ')
 for c in range(6):
  if pern[c]!=o['per_TX_keys'][c]['valid_anchor_occurrences'] or abs(perp[c]/pern[c]-o['per_TX_keys'][c]['positive_mean'])>1e-10 or abs(perneg[c]/pern[c]-o['per_TX_keys'][c]['negative_mean'])>1e-10:raise ValueError('Allstep class key counters differ')
 return dict(row_id=row,all_11100_records_parsed=True,**o,epoch_curve=curves)

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Complete fixed comparison required')
 native=[training_logs(base,row['row_id']) for row in rows];contrast=[contrastive_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=native,contrastive_logs=contrast,all_log_records_parsed=True,absolute_accuracy_percent={a:stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==a]) for a in d.ARMS})
 text='# F跨RX：保留当前批次梯度的跨批次记忆 clean 结果\n\n原F＋跨RX同TX为基底；从零50＝33L＋17U，RDrop1、旧门控、自身EMA、cos、LS.01及普通增强保持。默认头正常联合训练；无星地增强，只测clean，预测固定后独立CPU评分。\n\n|方法|seed|clean%|BASE%|整体差/百分点|弱TX均差|Macro-F1差|最差RX差|最差day差|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  vals=[v for v in r['paired'] if v['treatment']==arm];text+='\n## '+arm+' 逐TX配对差\n\n|TX|seed数|均差/百分点|样本SD|正差数|\n|---|---:|---:|---:|---:|\n'
  for i,name in enumerate(d.CLASSES):
   ns=[v['per_TX_pp'][i] for v in vals];s=stats(ns);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(n>0 for n in ns))+' |\n'
 text+='\n## 执行与成本\n\n|row|U曝光|接受数|接受率%|平均历史键数|最大步龄|训练秒|预测秒|CUDA峰值MiB|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for n,o,c in zip(native,contrast,r['cost']):text+='| '+n['row_id']+' | '+str(n['U_seen'])+' | '+str(n['U_selected'])+' | '+format(100*n['U_acceptance'],'.4f')+' | '+format(o['mean_historical_keys'],'.2f')+' | '+str(o['max_history_age'])+' | '+format(c['source_seconds'],'.1f')+' | '+format(c['prediction_seconds'],'.1f')+' | '+format(c['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\nSYM4 保留当前批次参照特征的梯度，仅历史参照停止梯度。空历史时，损失值和完整特征梯度退化为原始当前批次 SupCon（数值检查容差内），不引入当前键梯度截断。历史仍为最近三个真实 L 更新批次，每层和每次 RDrop 独立维护，按物理 ID 保留最新项并排除当前重复项。当前 detached 均值/标准差同时标准化所有 raw 描述子，再单位化。温度.2、跨RX同TX正样本、不同TX负样本、−log正样本数常数、5warm10rampmax.1均保持。\n\n不增加网络前向、IQ曝光、可训练参数、优化器步或推理流程。记忆只由可见源L训练步更新，U真值隐藏、V无更新；默认头原联合训练，辅助梯度不进入头。ownEMA保持原生更新，历史不导出。旧两行 ANCHOR/MEM4 仍按其独立冻结计划运行，没有修改或重启。两次RDrop复用同一次实际增强IQ。\n\n完整50轮、11100原生及11100对比记录逐条解析；9990启用step、39960层位／前向目标。固定final50，扩种子由源域规则在本候选目标访问前冻结，首种子无论源判断如何均自动clean测试。全部负结果保留，n1无SD。历史85.1139%只作方法锚点，实际配对复用本机同seed BASE。无独立盲测或星地收益声称。\n\n[XBM原论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)提供跨批次特征记忆思路，图像任务慢漂移假设不预设在当前训练中成立。本比较包括独立历史配对数、陈旧特征和参照梯度范围变化，不能仅归因于难负样本。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf8') as f:f.write(text)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(v['result']['records']),all_logs=True)))
