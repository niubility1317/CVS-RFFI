"""Complete native and disjoint source-gap logs; all paired clean outcomes."""
import argparse,gzip,json,math
from pathlib import Path
from . import design as d
from experiments.cvs_f_supcon50.mechanism import schedule,RX_BY_DOMAIN
from experiments.cvs_f_crossrx50_local.report_results import training_logs
from .analysis import stats

def gap_logs(base,row):
 root=Path(base)/row/'source';o=d.read(root/'gap_training_observation.json');calls=scheduled=eligible=active=packets=0;cellcalls=[0]*15;curve=[];current=None
 with gzip.open(root/'gap_training_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   v=json.loads(line);calls+=1;epoch=(calls-1)//222+1
   if v['step']!=calls or v['epoch']!=epoch or v['scheduled']!=(epoch>5) or v['weight']!=schedule(epoch) or not math.isfinite(v['loss']):raise ValueError('Complete ordered source gap records required')
   if current is None or current['epoch']!=epoch:current=dict(epoch=epoch,steps=0,scheduled=0,eligible=0,active=0,cells=0,packets=0,loss_sum=0.);curve.append(current)
   current['steps']+=1;current['scheduled']+=int(v['scheduled']);scheduled+=int(v['scheduled']);current['loss_sum']+=v['loss']
   if not v['eligible']:
    if v['valid_cells'] or v['active_cells'] or v['used_packets'] or v['loss'] or v['cells']:raise ValueError('Ineligible group produced a gap loss')
    continue
   if not v['scheduled'] or not v['cells'] or len({c['domain'] for c in v['cells']})!=len(v['cells']):raise ValueError('Only actual visible source cells')
   recomputed=[];used=0;act=0
   for c in v['cells']:
    i=c['domain'];counts=c['counts'];A=c['half_A_counts'];B=c['half_B_counts'];x=c['cross_gap_squared'];s=c['self_gap_squared']
    if not 0<=i<15 or c['RX']!=RX_BY_DOMAIN[i] or c['day']!=i%3+1 or any(n<2 for n in counts) or any(n<1 for n in A+B) or any(a+b!=n for a,b,n in zip(A,B,counts)) or not all(math.isfinite(t) for t in (x,s,c['violation'])) or s+2e-6<x:raise ValueError('Disjoint sameRX/day sufficient source packets required')
    estimate=x if o['treatment']=='CROSS_GAP' else s;loss=max(0.,.16-estimate)/.16
    if abs(loss-c['violation'])>2e-5:raise ValueError('Gap formula differs from actual recorded cells')
    recomputed.append(loss);cellcalls[i]+=1;used+=sum(counts);act+=c['violation']>0
   if len(recomputed)!=v['valid_cells'] or used!=v['used_packets'] or act!=v['active_cells'] or abs(sum(recomputed)/len(recomputed)-v['loss'])>2e-5:raise ValueError('Recorded gap aggregation differs')
   eligible+=1;active+=act>0;packets+=used;current['eligible']+=1;current['active']+=act>0;current['cells']+=len(recomputed);current['packets']+=used
 if (calls,scheduled,eligible,active,packets,cellcalls)!=(11100,9990,o['eligible_calls'],o['active_calls'],o['current_L_packets_in_valid_cells'],o['cell_calls']):raise ValueError('Complete source gap observation differs')
 return dict(row_id=row,all_11100_records_parsed=True,**o,epoch_curve=curve)

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Complete fixed comparison only')
 native=[training_logs(base,row['row_id']) for row in rows];gap=[gap_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=native,gap_logs=gap,all_log_records_parsed=True,absolute_accuracy_percent={arm:stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==arm]) for arm in d.ARMS})
 text='# 同RX/day两组样本的弱TX类差：完整clean结果\n\n原F＋跨RX同TX基底，默认头正常联合，从零50=33L+17U；不使用星地增强，只测clean168000。预测完成后独立CPU truth-last评分，同seed本机BASE复用。\n\n|方法|seed|clean准确率%|BASE准确率%|整体差/百分点|弱2类均差|Macro-F1差|最差RX差|最差day差|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  vals=[v for v in r['paired'] if v['treatment']==arm];text+='\n## '+arm+'逐TX配对差\n\n|TX|seed数|均差/百分点|样本SD|正差数|\n|---|---:|---:|---:|---:|\n'
  for i,name in enumerate(d.CLASSES):
   numbers=[v['per_TX_pp'][i] for v in vals];s=stats(numbers);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(n>0 for n in numbers))+' |\n'
 text+='\n## 执行与成本\n\n|row|U曝光|接受数|接受率%|有效gap事件|有违反事件|15cell覆盖|训练秒|预测秒|CUDA峰值MiB|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for n,g,cost in zip(native,gap,r['cost']):text+='| '+n['row_id']+' | '+str(n['U_seen'])+' | '+str(n['U_selected'])+' | '+format(100*n['U_acceptance'],'.4f')+' | '+str(g['eligible_calls'])+'/9990 | '+str(g['active_calls'])+' | '+format(g['RX_day_cell_coverage'],'.3f')+' | '+format(cost['source_seconds'],'.1f')+' | '+format(cost['prediction_seconds'],'.1f')+' | '+format(cost['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n实际两次RDrop joint160逐包单位化后平均，再对同一源RX/day内的弱14-7、20-19分别按批次顺序交替划为A/B；每类至少2个独立物理样本，每组至少1个。SELF_GAP使用两组类差平方范数均值，CROSS_GAP使用两组类差内积。二者差为半个类差组间距离平方；cross项保留符号，用相同margin0.4的归一化hinge，合法cell等权，原5轮预热/10轮ramp到.1，9990计划事件。两RDrop视图不是独立样本；分组确实使用不同物理包。有限池无放回分组并非独立IID，不声称严格无偏、MMD检验或统计显著性。\n\n默认头结构和主联合训练不改，gap无可训练头、prototype分类决策或推理步骤。原shallow/middle CROSS_RX SupCon、RDrop/LS.01、普通增强、旧门控ownEMAcos及统一Sinc物理更新均保留；gap反向进入实际部署特征，不增加模型前向、IQ读取、优化器步或参数。不同gap统计会产生不同loss和梯度幅度，不能把两条件之差全部归因于去噪而忽略目标强度。\n\n全部epoch/step/stdout及11100个gap记录完整解析。U接受率不是伪标签正确率。源域逐弱类几何、执行活跃度及默认头健康规则在新目标访问前决定扩种子；失败初始行仍自动clean评分，全部负结果保留。n1无样本SD，不能宣称可重复收益。历史85.1139%仅方法锚点，不作为本机配对数值；不宣称独立盲测、Phase2或星地性能。完整数值与CROSS_GAP−SELF_GAP见results_summary.json。均值嵌入相关背景参考[Gretton等2012](https://jmlr.org/papers/v13/gretton12a.html)，本实验是有监督特征分离penalty而非该论文检验的完整复现。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf8') as f:f.write(text)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(v['result']['records']),all_logs=True)))
