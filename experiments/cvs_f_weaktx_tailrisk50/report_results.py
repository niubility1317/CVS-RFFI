"""Parse all native and feature-tail logs; no IQ or truth connection."""
import argparse,gzip,json,math
from pathlib import Path
from collections import Counter,defaultdict
from . import design as d
from .analysis import stats
from experiments.cvs_f_crossrx50_local.report_results import training_logs

def tail_logs(base,row):
 root=Path(base)/row/'source';o=d.read(root/'feature_tail_training_observation.json');epochs=defaultdict(lambda:dict(steps=0,scheduled=0,eligible=0,loss_sum=0.,weight_sum=0.,valid_cells=0,available_anchor_objectives=0,selected_anchor_objectives=0));calls=scheduled=eligible=cells=available=selected=0;groups=Counter()
 with gzip.open(root/'feature_tail_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   v=json.loads(line);calls+=1;e=(calls-1)//222+1
   if v['step']!=calls or v['epoch']!=e or v['scheduled']!=(calls>1110) or not math.isfinite(v['loss']):raise ValueError('Complete ordered finite feature-tail logs required')
   epoch=epochs[e];epoch['steps']+=1
   if not v['scheduled']:continue
   scheduled+=1;epoch['scheduled']+=1;epoch['weight_sum']+=v['weight']
   if not v['eligible']:continue
   eligible+=1;epoch['eligible']+=1;epoch['loss_sum']+=v['loss'];cells+=v['valid_cells'];available+=v['available_anchor_objectives'];selected+=v['selected_anchor_objectives']
   for key in ('valid_cells','available_anchor_objectives','selected_anchor_objectives'):epoch[key]+=v[key]
   for level in v['levels']:
    for c in level['cells']:groups[str(c['TX'])+'@'+str(c['RX'])]+=1
 if calls!=11100 or scheduled!=9990 or (eligible,cells,available,selected)!=(o['applied'],round(o['weak_RX_cell_coverage']*9990*20),o['available_weak_anchor_objectives'],o['selected_weak_anchor_objectives']) or dict(groups)!=o['cell_level_calls']:raise ValueError('Feature-tail complete counters differ')
 curve=[]
 for e,v in sorted(epochs.items()):curve.append(dict(epoch=e,steps=v['steps'],scheduled=v['scheduled'],eligible=v['eligible'],mean_loss_applied=v['loss_sum']/v['eligible'] if v['eligible'] else None,mean_weight_scheduled=v['weight_sum']/v['scheduled'] if v['scheduled'] else None,weak_RX_cell_coverage=v['valid_cells']/(v['scheduled']*20) if v['scheduled'] else None,available_anchor_objectives=v['available_anchor_objectives'],selected_anchor_objectives=v['selected_anchor_objectives']))
 return dict(row_id=row,full_feature_log_parsed=True,**o,epoch_curve=curve)

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Complete frozen comparison only')
 logs=[training_logs(base,row['row_id']) for row in rows];tails=[tail_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=logs,feature_tail_logs=tails,all_log_records_parsed=True)
 text='# 弱 TX 分组特征尾部风险：完整 clean 结果\n\n默认头正常联合训练；逐模型从零固定 final50；每行 clean168000 完成预测后独立 CPU truth-last 评分。同种子 BASE 复用，不重复训练／测试。\n\n| 方法 | seed | clean准确率% | BASE准确率% | 整体差/百分点 | 弱2类差/百分点 | Macro-F1差/百分点 | 最差RX差/百分点 | 最差day差/百分点 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  pairs=[p for p in r['paired'] if p['treatment']==arm];summary.setdefault('absolute_accuracy_percent',{})[arm]=stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==arm]);text+='\n## '+arm+' 的逐类配对差\n\n| TX | 配对数 | 均差/百分点 | 样本SD | 正差数 |\n|---|---:|---:|---:|---:|\n'
  for i,name in enumerate(d.CLASSES):
   values=[p['per_TX_pp'][i] for p in pairs];s=stats(values);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(v>0 for v in values))+' |\n'
 text+='\n## 实际执行和成本\n\n| row | U曝光 | 接受数 | 接受率% | 特征锚点目标选择数/有效数 | RX格覆盖率% | 训练秒 | 预测秒 | CUDA峰值MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for log,t,cost in zip(logs,tails,r['cost']):text+='| '+log['row_id']+' | '+str(log['U_seen'])+' | '+str(log['U_selected'])+' | '+format(100*log['U_acceptance'],'.4f')+' | '+str(t['selected_weak_anchor_objectives'])+'/'+str(t['available_weak_anchor_objectives'])+' | '+format(100*t['weak_RX_cell_coverage'],'.4f')+' | '+format(cost['source_seconds'],'.1f')+' | '+format(cost['prediction_seconds'],'.1f')+' | '+format(cost['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n特征锚点目标数累计两个层位，表示损失中的样本位置，不是额外模型前向或新 IQ 读取。两行新增模型前向、参数和真实 optimizer 步数均为 0。TAIL 在每个弱 TX×RX 组内取最高一半（向上取整），每个物理 IQ 的两次 RDrop 损失先平均再选择；MEAN 取全部有效样本，其余算法相同。正对为同 TX 不同 RX，负对为不同 TX 同 RX；浅层 time_down／中层 t2.pool，原 CROSS_RX SupCon 保持，未加入 joint160 分支、掩码、元更新或另训头。\n\n原 F+CROSS_RX50=33L+17U、RDrop1、旧门控／ownEMA／cos、普通增强／LS.01、统一物理 Sinc 保持。默认头正常联合，新损失只进入实际特征；无星地增强，只测 clean。全部原生 epoch／step／stdout 和 feature_tail_steps 已解析。实际 U 接受率不代表最终置信度或伪标签真值正确率。\n\n扩种子依据目标访问前固定的逐弱 TX 源域几何、默认头健康度和实际覆盖率；out_RX 只排除描述性中心中的 RX，所有源 RX 都曾参与训练。每臂实际 seed 数、样本标准差及缺项保留。所有负结果保留；不按目标结果选模、补种子或调参。\n\n历史 85.1139% 仅作方法锚点，实际增量与本机同 seed BASE 配对。已暴露 Phase1 clean 探索，不宣称独立盲测、Phase2 或卫星部署。完整配对统计和 TAIL−MEAN 见 results_summary.json。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf-8') as f:f.write(text)
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(result['result']['records']),full_native_and_feature_logs=True)))
