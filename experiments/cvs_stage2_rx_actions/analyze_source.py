"""Full available source trajectory and actual action/critic execution audit."""
import argparse,gzip,json,statistics
from pathlib import Path
from . import design as d
from experiments.cvs_stage2_teacher_factorial import analyze_source as previous


def numeric_stats(values):
    values=[x for x in values if isinstance(x,(int,float)) and not isinstance(x,bool)]
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)) if values else None


def summarize(value):
    previous.d=d
    generic=previous.summaries(value);byid={x['row_id']:x for x in value['rows']}
    for row in generic['rows']:
        raw=byid[row['row_id']];steps=raw['steps'];arm=raw['resolved']['arm'];n=len(steps)
        physical=arm in ('PHYS','ROT','CRIT');expected_actions=max(0,n-888)*128 if physical else 0
        expected_aux=max(0,n//4-222) if arm=='CRIT' else 0
        observed={key:sum(s['details'].get(key,0) for s in steps) for key in ('RX_L_packets','RX_U_packets','critic_packets')}
        aux=[s for s in steps if s['details'].get('critic_packets',0)>0]
        row['execution_check']=dict(complete_step_prefix=n,observed=observed,expected_RX_packets=expected_actions,expected_aux_steps=expected_aux,actual_aux_steps=len(aux),valid=observed['RX_L_packets']==observed['RX_U_packets']==expected_actions and observed['critic_packets']==expected_aux*32 and len(aux)==expected_aux)
        row['action_statistics']={}
        for role in ('L_action','U_action'):
            records=[s['details'][role] for s in steps if role in s['details']]
            row['action_statistics'][role]=dict(count=len(records),learned_selection_steps=sum(x['critic_selection_active'] for x in records),metrics={k:numeric_stats([x[k] for x in records]) for k in ('full_strength_count','selection_entropy','clipped_bin_fraction')})
        row['critic_statistics']={}
        for label,records in [('all',aux),('first_100',aux[:100]),('last_100',aux[-100:])]:
            row['critic_statistics'][label]={k:numeric_stats([x['details']['critic'][k] for x in records]) for k in ('target_CE_increment_mean','prediction_mean','absolute_error_mean')}
        row['all_critic_targets_detached']=all(x['details']['critic']['identity_target_detached'] for x in aux)
        row['augmented_identity_loss']={k:numeric_stats([s['details'].get(k) for s in steps if s['step']>888]) for k in ('clean_CE','augmented_L_CE','augmented_L_fraction')}
    generic['complete_source_matrix']=len(generic['rows'])==15 and all(r['source_complete'] and r['complete_steps']==8880 and r['complete_epochs']==40 for r in generic['rows'])
    generic['execution_counts_valid']=all(r['execution_check']['valid'] and r['all_critic_targets_detached'] for r in generic['rows'])
    generic['scope']='Complete source trajectories, no target access' if generic['complete_source_matrix'] else 'All complete records in live source snapshots, not final'
    generic['paired']=[]
    for seed in d.SEEDS:
        for a,b in [('SSL','CE'),('PHYS','SSL'),('ROT','SSL'),('PHYS','ROT'),('CRIT','PHYS')]:
            left=byid.get(f'{a}-s{seed}');right=byid.get(f'{b}-s{seed}')
            if left is None or right is None:continue
            common=[]
            for x,y in zip(left['epochs'],right['epochs']):
                assert x['epoch']==y['epoch']
                common.append(dict(epoch=x['epoch'],left_movement=x['weight_movement'],right_movement=y['weight_movement'],left_loss=x['total_loss'],right_loss=y['total_loss'],V_difference=x['V_accuracy']-y['V_accuracy']))
            generic['paired'].append(dict(seed=seed,left=a,right=b,epochs=common))
    return generic


def report(s,name):
    t=['# RX 作用与辅助网络：完整可用源日志审计','',f"读取时间 {s['read_at']}；{len(s['rows'])}/15 行，共 {s['all_available_step_records']} 条完整逐步记录。完整训练矩阵：{s['complete_source_matrix']}。执行计数核验：{s['execution_counts_valid']}。未读取目标预测或评分。",'',
       '|行|epoch|step|L/U 操作各包次|辅助更新|已启用学习选择步数|末段辅助绝对误差|','|---|---:|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        ex=r['execution_check'];sel=r['action_statistics']['L_action'];mae=r['critic_statistics']['last_100']['absolute_error_mean'];m=f"{mae['mean']:.6f}" if mae else 'N/A'
        t.append(f"|{r['row_id']}|{r['complete_epochs']}|{r['complete_steps']}|{ex['observed']['RX_L_packets']}|{ex['actual_aux_steps']}|{sel['learned_selection_steps']}|{m}|")
    t+=['','每行完整读取逐步、逐 epoch 和 CSV 日志，并扫描完整 stdout 的异常标记。在线文件的末端时刻可能不同；只解析完整记录，不能把进行中的前缀当作训练结束。','',
        '核验物理分支从第889步起，每步各128个L/U操作；CRIT每4步32个L拟合辅助目标，其余方案辅助步数为0。全部辅助目标应从身份模型detach。','',
        '选择熵接近ln(8)=2.07944意味着选择仍接近均匀；辅助损失下降本身不能证明它正确排序困难操作。当前日志缺少预测风险与真实难度的候选排序相关性，因此不能从这些日志直接声称难度选择已有效。','',
        '参数位移、梯度范数和源V仅描述学习与健康状态，均不是未见RX泛化证据；源V小幅涨跌不参与选模。所有同seed、同epoch对照和完整阶段统计保存在JSON中。', '',f'源日志快照：`evidence/{name}`；统计：`evidence/source_trajectory_summary.json`。原始未删减日志继续保存在远端逐行source路径。']
    return '\n'.join(t)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();previous.d=d
    blob,value=previous.capture();s=summarize(value)
    # Summarize every complete original record before compacting per-packet arrays.
    for row in value['rows']:
        for step in row['steps']:
            for key in ('U_predictions','U_accepted_mask'):step.get('details',{}).pop(key,None)
    value['snapshot_omissions']=['details.U_predictions','details.U_accepted_mask'];value['full_original_logs']=d.BASE.as_posix()+'/<row>/source/step_metrics.jsonl.gz'
    folder=a.output;name=f"source_trajectory_{int(value['read_at'])}.json.gz";(folder/'evidence').mkdir(parents=True,exist_ok=True)
    with (folder/'evidence'/name).open('xb') as f:f.write(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode()))
    d.write(folder/'evidence/source_trajectory_summary.json',s);(folder/'source_trajectory.md').write_text(report(s,name),encoding='utf-8')
    print(json.dumps(dict(rows=len(s['rows']),steps=s['all_available_step_records'],complete=s['complete_source_matrix'],counts_valid=s['execution_counts_valid'],epochs={r['row_id']:r['complete_epochs'] for r in s['rows']})))
