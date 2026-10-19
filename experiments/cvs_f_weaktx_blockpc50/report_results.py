"""All original logs and all weak-pair feature records, no IQ or truth access."""
import argparse,gzip,json,math
from pathlib import Path
from collections import Counter,defaultdict
from . import design as d
from .analysis import stats
from .core import active,strength
from experiments.cvs_f_crossrx50_local.report_results import training_logs

def pair_logs(base,row):
 root=Path(base)/row/'source';o=d.read(root/'weak_pair_training_observation.json');epochs=defaultdict(lambda:dict(steps=0,scheduled=0,eligible=0,changed=0,VJP=0,max_norm_error=0.));calls=scheduled=eligible=changed=vjps=0;conflicts=Counter();nonzero=Counter();max_error=0.
 with gzip.open(root/'weak_pair_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   v=json.loads(line);calls+=1;e=(calls-1)//222+1
   if v['step']!=calls or v['epoch']!=e or v['scheduled']!=active(calls) or not math.isfinite(v['original_L_identity_loss']) or v['surrogate_value']!=0. or v['strength']!=(strength(calls) if active(calls) else 0.):raise ValueError('Complete ordered finite weak-pair records required')
   epoch=epochs[e];epoch['steps']+=1;epoch['scheduled']+=int(v['scheduled']);epoch['eligible']+=int(v['eligible']);epoch['changed']+=int(v['nonzero_blocks']>0);epoch['VJP']+=v['feature_VJP_calls'];epoch['max_norm_error']=max(epoch['max_norm_error'],v['maximum_norm_match_error']);scheduled+=int(v['scheduled']);eligible+=int(v['eligible']);changed+=int(v['nonzero_blocks']>0);vjps+=v['feature_VJP_calls']
   if v['eligible'] and (not v['scheduled'] or not all(n>0 for n in v['weak_counts']) or v['feature_VJP_calls']!=3):raise ValueError('Actual source weak pair VJP coverage differs')
   if not v['eligible'] and (v['feature_VJP_calls'] or v['blocks']):raise ValueError('Inactive event executed feature VJP')
   for b in v['blocks']:
    if not 0<=b['fp32_norm_match_error']<=1e-5 or not all(math.isfinite(b[k]) for k in ('L_before_norm','L_PC_target_norm','L_control_norm','norm_gain','delta_norm')):raise ValueError('Finite matched block gradient records required')
    conflicts[b['block']]+=int(b['conflict']);nonzero[b['block']]+=int(b['delta_norm']>0);max_error=max(max_error,b['fp32_norm_match_error'])
 if (calls,scheduled,eligible,changed,vjps)!=(11100,2498,o['eligible'],o['changed_steps'],o['extra_feature_VJP_calls']) or dict(conflicts)!=o['block_conflict_calls'] or dict(nonzero)!=o['block_nonzero_calls'] or max_error!=o['maximum_norm_match_error']:raise ValueError('Complete weak-pair record counters differ')
 return dict(row_id=row,all_11100_records_parsed=True,**o,epoch_curve=[dict(epoch=e,**v) for e,v in sorted(epochs.items())])

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Complete frozen comparison only')
 logs=[training_logs(base,row['row_id']) for row in rows];pairs=[pair_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=logs,weak_pair_logs=pairs,all_log_records_parsed=True)
 text='# 弱 TX 模块梯度协调：完整 clean 结果\n\n原 F＋跨 RX 同 TX 对比学习基底，默认头联合训练，从零固定 final50。无星地增强，只测 clean168000；预测固定后独立 CPU truth-last 评分。同 seed BASE 复用。\n\n| 方法 | seed | clean准确率% | BASE准确率% | 整体差/百分点 | 弱2类差/百分点 | Macro-F1差/百分点 | 最差RX差/百分点 | 最差day差/百分点 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  vals=[p for p in r['paired'] if p['treatment']==arm];summary.setdefault('absolute_accuracy_percent',{})[arm]=stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==arm]);text+='\n## '+arm+' 的逐类配对差\n\n| TX | 配对数 | 均差/百分点 | 样本SD | 正差数 |\n|---|---:|---:|---:|---:|\n'
  for i,name in enumerate(d.CLASSES):
   values=[p['per_TX_pp'][i] for p in vals];s=stats(values);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(v>0 for v in values))+' |\n'
 text+='\n## 执行和成本\n\n| row | U曝光 | 接受数 | 接受率% | 额外特征VJP | 触发/计划 | 非零修正率% | 训练秒 | 预测秒 | CUDA峰值MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for log,p,cost in zip(logs,pairs,r['cost']):text+='| '+log['row_id']+' | '+str(log['U_seen'])+' | '+str(log['U_selected'])+' | '+format(100*log['U_acceptance'],'.4f')+' | '+str(p['extra_feature_VJP_calls'])+' | '+str(p['eligible'])+'/'+str(p['scheduled'])+' | '+format(100*p['nonzero_fraction'],'.4f')+' | '+format(cost['source_seconds'],'.1f')+' | '+format(cost['prediction_seconds'],'.1f')+' | '+format(cost['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n弱 TX 固定为源 V 校准的14-7／20-19。每模型前5轮不变，其后每4步对当前同一 L128 的两次 RDrop CE 平均贡献做3次特征 VJP。类贡献为该类样本损失之和/128，不增加类权重。两个弱类梯度在模块内内积为负时，采用对称 PCGrad 思路去除相反分量，强度第6轮.1至第15轮1。NORM_L_ID 保持当前 L 身份目标原方向，逐模块匹配 BLOCK_PAIR 的 L 身份分量范数。两者匹配当前状态下的该分量；U／其他损失随后相加并执行原物理 clip5／AdamW，不保证完整梯度或实际 AdamW 下降。\n\n模块范围是实际部署的全部身份特征，排除默认分类头；原浅／中层 CROSS_RX SupCon 保持。无冲突、零梯度或弱类缺失时跳过。未使用参数保持 None。Sinc 两个 q 叶参数共用25000Hz坐标，投影系数／范数比不受共同单位缩放影响；原物理裁剪、单位修正和包含 origin 的物理衰减不再叠乘。新增参数、模型前向、真实 optimizer 步和另训头均为0；新增 VJP 成本单列。默认头直接辅助梯度为 None，但全局裁剪可改变主头梯度的最终比例。\n\n原基底50轮=33L＋17U、旧门控＋ownEMA＋cos、RDrop1／普通增强／LS.01保持。全部原生 epoch、step、stdout 及11100个弱对记录均完整解析。实际 U 接受率不代表伪标签真值正确率。源域扩种子决定在目标访问前固定，失败的初始两行仍 clean 评分；目标结果不用于调参、选模或补种子。n=1没有样本SD。所有负结果保留。\n\n历史85.1139%为方法锚点，增量与本机同 seed BASE 配对。已暴露 Phase1 clean 探索，不宣称独立盲测、Phase2或星地部署。完整配对、均值／样本SD／正差数及 BLOCK_PAIR−NORM_L_ID 见 results_summary.json。方法灵感来源：[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)，本实现是固定弱类、分模块、有范数对照的适配，不宣称原论文的完整复现或保证有效。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf-8') as f:f.write(text)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(result['result']['records']),full_native_and_feature_logs=True)))
