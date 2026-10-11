"""Audit every complete source record; no target predictions or scoring input."""
import argparse,gzip,json,statistics
from pathlib import Path
from . import design as d
from experiments.cvs_stage2_teacher_factorial import analyze_source as capture

def stats(values):
 values=[x for x in values if isinstance(x,(int,float)) and not isinstance(x,bool)]
 return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)) if values else None

def summarize(value):
 capture.d=d;generic=capture.summaries(value);byid={r['row_id']:r for r in value['rows']}
 for r in generic['rows']:
  raw=byid[r['row_id']];c=raw['resolved'];d.validate({k:c[k] for k in d.config(next(x for x in d.rows() if x['row_id']==r['row_id']))})
  steps=raw['steps'];n=len(steps);physical=c['rx_action']['enabled'];online=c['risk_learning']['online'];learned=c['risk_learning']['enabled']
  expected=max(n-888,0)*128 if physical else 0;expected_aux=max(n//4-222,0) if online else 0
  aux=[s for s in steps if s['details'].get('critic_packets',0)]
  observed={k:sum(s['details'].get(k,0) for s in steps) for k in ('RX_L_packets','RX_U_packets','critic_packets')}
  r['execution_check']=dict(step_prefix=n,observed=observed,expected_RX_packets=expected,expected_auxiliary_steps=expected_aux,actual_auxiliary_steps=len(aux),valid=observed['RX_L_packets']==observed['RX_U_packets']==expected and observed['critic_packets']==expected_aux*32 and len(aux)==expected_aux)
  r['action_statistics']={}
  for role in ('L_action','U_action'):
   records=[s['details'][role] for s in steps if role in s['details']]
   r['action_statistics'][role]=dict(count=len(records),learned_selection_steps=sum(x['critic_selection_active'] for x in records),metrics={k:stats([x.get(k) for x in records]) for k in ('selection_entropy','selection_total_variation','risk_score_std','full_strength_count','clipped_bin_fraction')})
   if len(records)!=expected//128 or sum(x['critic_selection_active'] for x in records)!=(expected//128 if learned else 0):raise ValueError('Selection schedule mismatch')
  r['auxiliary_statistics']={label:{k:stats([x['details']['critic'].get(k) for x in subset]) for k in ('pair_concordance','eligible_margin_pairs','target_margin_drop_mean','prediction_mean','candidate_identity_forwards')} for label,subset in [('all',aux),('first_100',aux[:100]),('last_100',aux[-100:])]}
  r['auxiliary_gradient_norm']=stats([s.get('auxiliary_gradient_norm') for s in aux])
  r['all_auxiliary_targets_detached']=all(s['details']['critic']['identity_target_detached'] for s in aux)
  r['auxiliary_identity_gradients_zero']=all(all(v==0 for v in s['gradient_audit']['weighted_dense_gradient_norms']['critic'].values()) for s in steps if s.get('gradient_audit'))
  r['augmented_identity_loss']={k:stats([s['details'].get(k) for s in steps if s['step']>888]) for k in ('clean_CE','augmented_L_CE','augmented_L_fraction')}
  enabled=c['sam']['enabled'];sam=[s['details']['SAM'] for s in steps if 'SAM' in s['details']]
  r['SAM_execution']=dict(enabled=enabled,recorded_steps=len(sam),expected_steps=n if enabled else 0,metrics={k:stats([x.get(k) for x in sam]) for k in ('first_dense_gradient_norm','actual_perturbation_l2','perturbed_identity_loss')},targets_recomputed=sum(x['targets_recomputed'] for x in sam),auxiliary_recomputed=sum(x['auxiliary_recomputed'] for x in sam))
  if len(sam)!=(n if enabled else 0) or any(x['targets_recomputed'] or x['auxiliary_recomputed'] for x in sam):raise ValueError('SAM replay execution differs')
  for epoch in raw['epochs']:
   if epoch['counts']['identity_gradient_passes']!=epoch['optimizer_steps']*(2 if enabled else 1):raise ValueError('Gradient pass count differs')
  if any(abs(x['actual_perturbation_l2']-.05)>1e-5 or x['rho']!=.05 for x in sam):raise ValueError('SAM perturbation radius differs')
  r['auxiliary_target_identity_packets']=sum(s['details']['critic']['candidate_identity_forwards'] for s in aux)
  if r['auxiliary_target_identity_packets']!=expected_aux*32*9:raise ValueError('Auxiliary all-eight target budget mismatch')
 generic['complete_source_matrix']=len(generic['rows'])==18 and all(r['source_complete'] and r['complete_steps']==8880 and r['complete_epochs']==40 for r in generic['rows'])
 generic['execution_counts_valid']=all(r['execution_check']['valid'] and r['all_auxiliary_targets_detached'] and r['auxiliary_identity_gradients_zero'] for r in generic['rows'])
 generic['scope']='All available complete source records, no target access; final only if all18rows completed'
 generic['paired']=[]
 for seed in d.SEEDS:
  for a,b in d.PAIRS:
   if b=='P0':continue
   left=byid.get(f'{a}-s{seed}');right=byid.get(f'{b}-s{seed}')
   if left is None or right is None:continue
   common=[]
   for x,y in zip(left['epochs'],right['epochs']):
    if x['epoch']!=y['epoch']:raise ValueError('Mismatched epoch order')
    common.append(dict(epoch=x['epoch'],left_movement=x['weight_movement'],right_movement=y['weight_movement'],left_losses={k:x[k] for k in ('ce','soft_identity','critic')},right_losses={k:y[k] for k in ('ce','soft_identity','critic')},V_difference=x['V_accuracy']-y['V_accuracy']))
   generic['paired'].append(dict(seed=seed,left=a,right=b,epochs=common))
 return generic

def report(s,name):
 t=['# SAM与物理风险网络：完整可用源日志审计','',f"读取时间 Unix {s['read_at']}；已启动 {len(s['rows'])}/18 行，共解析 {s['all_available_step_records']} 条完整逐步记录。完整训练矩阵：{s['complete_source_matrix']}；预算与梯度隔离核验：{s['execution_counts_valid']}。未读取目标预测或分数。",'', '|行|完整epoch|完整step|辅助更新|L动作分布距均匀的总变差|末100辅助步排序一致率|','|---|---:|---:|---:|---:|---:|']
 def fmt(v):return f"{v['mean']:.5f}" if v else 'N/A'
 for r in s['rows']:
  ex=r['execution_check'];t.append(f"|{r['row_id']}|{r['complete_epochs']}|{r['complete_steps']}|{ex['actual_auxiliary_steps']}|{fmt(r['action_statistics']['L_action']['metrics']['selection_total_variation'])}|{fmt(r['auxiliary_statistics']['last_100']['pair_concordance'])}|")
 t+=['','每行所有已完整写入的 step、epoch、CSV 和 stdout 都已读取；表中末100步只是全部记录分析后的末段摘要。不同日志文件逐一读取，在线末端可能不同，不把实时前缀当最终训练。','', 'PHYS/SAM_PHYS 的动作概率应为均匀；RANK/SAM_RANK 自第889步使用预热风险，每4步至多32L、每L全部8候选。其他配方不执行辅助更新。总变差大于0只说明采样发生变化，不证明选择正确；在线候选一致率在当前L上测量，不是独立未知RX指标。','', '已核对辅助目标detach，以及日志中所有既有分量审计点的辅助损失对身份参数梯度为0。身份学习来自所选IQ视图的CE/SSL。全阶段损失、动作熵、梯度范数、实际计数及同seed同epoch参数位移比较保存在JSON。','', '源V准确率仅用于遗忘/异常检查，不用于选模；参数位移和辅助排序都不能替代最终clean测试。U没有TX标签，其准确率保持未知。所有健康训练按原配置继续。','',f'完整可用源快照：`evidence/{name}`；汇总：`evidence/source_trajectory_summary.json`。逐U预测/接受数组仅在本地紧凑快照中省略，原始完整日志保留远端。']
 t+=['', '## SAM实际执行', '', '|行|SAM双遍步数|平均扰动L2|扰动后身份loss均值|', '|---|---:|---:|---:|']
 for row in s['rows']:
  a=row['SAM_execution']
  if a['enabled']:t.append(f"|{row['row_id']}|{a['recorded_steps']}|{fmt(a['metrics']['actual_perturbation_l2'])}|{fmt(a['metrics']['perturbed_identity_loss'])}|")
 t+=['', '已检查全部可用SAM记录的rho与实际L2、每epoch梯度次数、目标与辅助不重算标志；实现层面的同IQ/dropout与精确恢复已由发布前检查覆盖。损失均值随训练进度变化，不用于不同进度方案排序。', '', '两遍身份计算增加训练成本，测试仍为原身份网络。参数位移、低源loss或辅助排序不能直接证明未知接收机收益；全部固定40轮和测试队列保持。']
 return '\n'.join(t)+'\n'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();capture.d=d;blob,value=capture.capture();s=summarize(value)
 if not s['execution_counts_valid']:raise ValueError('Observed execution differs from fixed recipe')
 for row in value['rows']:
  for step in row['steps']:
   for k in ('U_predictions','U_accepted_mask'):step['details'].pop(k,None)
 value['snapshot_omissions']=['details.U_predictions','details.U_accepted_mask'];value['full_original_logs']=d.BASE.as_posix()+'/<row>/source/step_metrics.jsonl.gz'
 folder=a.output;name=f"source_trajectory_{int(value['read_at'])}.json.gz";(folder/'evidence').mkdir(parents=True,exist_ok=True)
 with (folder/'evidence'/name).open('xb') as f:f.write(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode()))
 d.write(folder/'evidence/source_trajectory_summary.json',s);(folder/'source_trajectory.md').write_text(report(s,name),encoding='utf-8')
 print(json.dumps(dict(rows=len(s['rows']),steps=s['all_available_step_records'],complete=s['complete_source_matrix'],counts_valid=s['execution_counts_valid'],progress=[dict(row_id=r['row_id'],epochs=r['complete_epochs'],auxiliary_steps=r['execution_check']['actual_auxiliary_steps'],L_TV=r['action_statistics']['L_action']['metrics']['selection_total_variation'],last_rank=r['auxiliary_statistics']['last_100']['pair_concordance']) for r in s['rows']])))
