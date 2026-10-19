"""Complete native logs and actual post-optimizer step evidence, no IQ/truth reads."""
import argparse,gzip,json,math
from pathlib import Path
from collections import defaultdict
from . import design as d
from .analysis import stats
from .core import active,strength
from experiments.cvs_f_crossrx50_local.report_results import training_logs

def step_logs(base,row):
 root=Path(base)/row/'source';o=d.read(root/'actual_feature_step_observation.json');curve=defaultdict(lambda:dict(steps=0,scheduled=0,eligible=0,changed=0,fallbacks=0,VJP=0,constraint_calls=0,max_norm_error=0.,max_actual_enforced_residual=0.));calls=scheduled=eligible=changed=fallbacks=vjps=enforced=0;maxerr=0.;maxrisk=0.
 with gzip.open(root/'actual_feature_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   v=json.loads(line);calls+=1;e=(calls-1)//222+1
   if v['step']!=calls or v['epoch']!=e or v['scheduled']!=active(calls) or not v['finalized'] or v['strength']!=(strength(calls) if active(calls) else 0.) or any(v[k]!=0 for k in ('extra_model_forwards','extra_optimizer_steps','head_fit_updates')):raise ValueError('Ordered complete native actual-step records required')
   q=curve[e];q['steps']+=1;q['scheduled']+=int(v['scheduled']);scheduled+=int(v['scheduled'])
   if v['eligible']:
    if not v['scheduled'] or not all(n>0 for n in v['class_counts']) or v['feature_VJP_calls']!=3 or not v['default_W_post_correction_exact'] or v['selected_feature_parameters']!=231233:raise ValueError('Actual source class groups/full feature/defaulthead boundary differs')
    if not all(math.isfinite(v[k]) for k in ('raw_step_normalized_norm','projected_stored_normalized_norm','control_stored_normalized_norm','stored_norm_match_error','candidate_full_strength_max_residual_relative','applied_full_strength_max_residual_relative','applied_normalized_correction_fraction')) or v['stored_norm_match_error']>.001:raise ValueError('Stored normalized native-step evidence differs')
    expected=v['candidate_constraint_enforced'] and o['treatment']=='UPDATE_CONE'
    if v['constraint_enforced']!=expected or v['applied_normalized_dots']!=(v['candidate_stored_normalized_dots'] if o['treatment']=='UPDATE_CONE' else v['control_stored_normalized_dots']):raise ValueError('Applied/counterfactual candidate risk conflated')
    if expected and v['applied_full_strength_max_residual_relative']>.001:raise ValueError('Stored fullstrength first-order risk constraint violated')
    bad=int(v['solver_fallback'] or v['projection_storage_fallback'] or v['storage_match_fallback']);eligible+=1;changed+=int(v['actual_correction_nonzero']);fallbacks+=bad;vjps+=3;enforced+=int(expected);maxerr=max(maxerr,v['stored_norm_match_error']);maxrisk=max(maxrisk,v['applied_full_strength_max_residual_relative'] if expected else 0.)
    q['eligible']+=1;q['changed']+=int(v['actual_correction_nonzero']);q['fallbacks']+=bad;q['VJP']+=3;q['constraint_calls']+=int(expected);q['max_norm_error']=max(q['max_norm_error'],v['stored_norm_match_error']);q['max_actual_enforced_residual']=max(q['max_actual_enforced_residual'],v['applied_full_strength_max_residual_relative'] if expected else 0.)
   elif v['feature_VJP_calls']:raise ValueError('Ineligible class group executed VJP')
 if (calls,scheduled,eligible,changed,fallbacks,vjps,enforced)!=(11100,2498,o['eligible'],o['changed_steps'],o['numerical_fallbacks'],o['extra_feature_VJP_calls'],o['actual_constraint_calls']) or maxerr!=o['maximum_norm_match_error'] or abs(maxrisk-max(0.,o['maximum_enforced_residual_relative']))>1e-15:raise ValueError('Complete native-step observation counters differ')
 return dict(row_id=row,all_11100_records_parsed=True,**o,epoch_curve=[dict(epoch=e,**v) for e,v in sorted(curve.items())])

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Complete frozen comparison only')
 logs=[training_logs(base,row['row_id']) for row in rows];steps=[step_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=logs,actual_step_logs=steps,all_log_records_parsed=True)
 text='# 实际特征位移风险约束：完整 clean 结果\n\n原 F＋跨 RX 同 TX 基底，默认头联合训练，从零 fixed final50。无星地增强，仅 clean168000；预测固定后独立 CPU truth-last 评分，同 seed BASE 复用。\n\n| 方法 | seed | clean准确率% | BASE准确率% | 整体差/百分点 | 弱2类差/百分点 | Macro-F1差/百分点 | 最差RX差/百分点 | 最差day差/百分点 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  vals=[p for p in r['paired'] if p['treatment']==arm];summary.setdefault('absolute_accuracy_percent',{})[arm]=stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==arm]);text+='\n## '+arm+' 的逐类配对差\n\n| TX | 配对数 | 均差/百分点 | 样本SD | 正差数 |\n|---|---:|---:|---:|---:|---:|\n'
  for i,name in enumerate(d.CLASSES):
   values=[p['per_TX_pp'][i] for p in vals];s=stats(values);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(v>0 for v in values))+' |\n'
 text+='\n## 实际执行与成本\n\n| row | U曝光 | 接受数 | 接受率% | 额外特征VJP | 实际纠正/计划 | 数值回退 | 范数匹配最大误差 | 训练秒 | 预测秒 | CUDA峰值MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for log,st,cost in zip(logs,steps,r['cost']):text+='| '+log['row_id']+' | '+str(log['U_seen'])+' | '+str(log['U_selected'])+' | '+format(100*log['U_acceptance'],'.4f')+' | '+str(st['extra_feature_VJP_calls'])+' | '+str(st['changed_steps'])+'/'+str(st['scheduled'])+' | '+str(st['numerical_fallbacks'])+' | '+format(st['maximum_norm_match_error'],'.3g')+' | '+format(cost['source_seconds'],'.1f')+' | '+format(cost['prediction_seconds'],'.1f')+' | '+format(cost['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n新机制只使用当前实际 L128 的原两次 RDrop CE：弱TX1、弱TX3、其他4类的实际损失贡献分别求和/128，不增类权重。3个特征VJP后，原损失、梯度、物理clip5、完整原生AdamW与moments均照常。优化器结束时，以原完整特征位移每模块RMS为尺度，约束3个当前分类风险的线性变化，最小改动投影到非正半空间交集。零位移模块保持零；存储float32后的范数对照保留原完整位移方向，匹配相同当前状态下的投影位移范数。归一化位移范数不是跨训练轨迹相同，也不是梯度范数。\n\n前5轮不变，其后每4步，1112至11100共2498计划事件；强度第6轮.1到第15轮1。前述一阶风险约束仅对完整强度、无存储匹配回退的UPDATE_CONE实际特征位移成立，容差为原归一化步范数的.001。STEP_NORM的实际风险另记，不能冒充投影后的反事实风险。矩阵病态时保守零投影、存储误差过大时零特征位移、无法匹配对照时两者保留原位移，均显式记录并阻止源域扩种子。\n\n原默认分类头（包括gain和归一化参数）保持完整原生联合更新，不投影、不冻结、不另训。新纠正在同一主步内、Sinc更新观察与自身EMA之前完成；moments仅按原梯度推进一次。Sinc物理origin衰减、坐标单位映射、物理裁剪均在原生更新中执行一次；后续额外约束力作用于实际f参数位移，不重复衰减或叠乘历史倍率。实际leaf copy递增版本，保留原参数派生滤波缓存失效。部署仍同一joint160，无新增参数、推理步骤或模型前向。\n\n约束来自更新前默认头下的当前L特征导数，不能保证更新后分类头的真实非线性CE、其他RX或目标性能。原基底50=33L+17U、RDrop1／LS.01／普通增强／旧门控ownEMAcos保持。报告完整U曝光及接受率，接受率不是伪标签真实正确率。全部epoch／step／stdout及11100个实际更新记录完整解析。\n\n扩种子只按目标访问前固定的源域逐弱类几何、默认头健康及执行条件；失败初始两行仍自动clean评分。n1没有样本SD，全部负结果保留。历史85.1139%仅方法锚点，增量与本机同seedBASE配对。已暴露Phase1探索，不宣称独立盲测、Phase2或星地性能。完整统计和UPDATE_CONE−STEP_NORM见results_summary.json。约束思路参考[Gradient Episodic Memory](https://arxiv.org/abs/1706.08840)，本实验是当前类风险、完整实际特征步与存储范数对照的适配，不宣称完整复现、首创或预设有效。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf-8') as f:f.write(text)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(result['result']['records']),full_native_and_feature_logs=True)))
