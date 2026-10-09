"""Summarize all fixed source audit rows; never access target outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from experiments.cvs_multi_action_audit import design as d

def summarize(folder):
    folder=Path(folder);completed=d.read(folder/'completion.json')
    if completed['status']!='SOURCE_AUDIT_COMPLETE' or completed['rows']!=4 or completed['target_access']:
        raise ValueError('Incomplete source audit')
    flat=[];receiver=[];all_ids=None;steps=0
    for row in d.rows():
        base=folder/row['row_id'];done=d.read(base/'completion.json')
        if not done['identity_unchanged'] or done['target_access'] or done['exposures']!=list(d.EXPOSURES):raise ValueError('Identity or scope changed')
        roles=d.read(base/'physical_roles.json')
        ids={k:{r['id'] for r in roles[k]} for k in ('fit','audit')}
        if len(ids['fit'])!=1920 or len(ids['audit'])!=960 or ids['fit']&ids['audit']:raise ValueError('Role counts/overlap')
        if all_ids is None:all_ids=ids
        elif all_ids!=ids:raise ValueError('Physical roles differ across model seeds')
        for view in d.EXPOSURES:
            actions=d.read(base/(view+'_actions.json'))
            logs=[json.loads(s) for s in (base/(view+'_actions_steps.jsonl')).read_text().splitlines()]
            if len(logs)!=1600:raise ValueError('Action fit budget differs')
            steps+=len(logs)
            for kind in ('linear','temporal'):
                for mode in (*d.RECIPE['fit_modes'],'oracle_parameters'):
                    mode_logs=[r for r in logs if r['kind']==kind and r['mode']==mode]
                    if [r['step'] for r in mode_logs]!=list(range(1,201)):raise ValueError('Missing fit steps')
                    for code,metrics in actions['kinds'][kind]['modes'][mode].items():
                        if metrics['count']!=960:raise ValueError('Incomplete heldout predictions')
                        for space in ('h','z'):
                            m=metrics[space]
                            if m['target_energy']>1e-12 and abs(m['skill_vs_zero']-(1-m['mse']/m['zero_mse']))>1e-6:
                                raise ValueError('Skill recount differs')
                        flat.append(dict(row_id=row['row_id'],seed=row['model_seed'],view=view,kind=kind,mode=mode,code=code,
                            h_skill=metrics['h']['skill_vs_zero'],z_skill=metrics['z']['skill_vs_zero'],
                            h_mean_skill=metrics['h']['skill_vs_training_mean'],z_mean_skill=metrics['z']['skill_vs_training_mean'],
                            margin_mae=metrics['margin']['delta_mae'],reliability=metrics['reliability'],
                            actual_accuracy=metrics['actual_accuracy'],classification_flip_rate=metrics['classification_flip_rate']))
            r=d.read(base/(view+'_receiver.json'))
            if r['status']!='VERIFIED' or r['steps']!=200 or r['target_access'] or r['identity_updated']:raise ValueError('R diagnostic incomplete')
            rl=[json.loads(s) for s in (base/(view+'_receiver_steps.jsonl')).read_text().splitlines()]
            if len(rl)!=200:raise ValueError('R budget differs')
            steps+=len(rl)
            for code,m in r['predictive_metrics'].items():
                if abs(m['skill_over_zero']-(1-m['sse']/m['target_energy']))>1e-6:raise ValueError('R skill recount differs')
                receiver.append(dict(row_id=row['row_id'],seed=row['model_seed'],view=view,code=code,
                    h_skill=m['skill_over_zero'],centroid_G_mse=m['centroid_G_mse'],
                    group_G_mse=m['predicted_centroid_delta_vs_group_mean_G_delta_mse'],
                    group_margin_mae=m['predicted_centroid_margin_delta_vs_mean_packet_margin_delta_mae'],
                    audit_relations=r['audit_group_relations']))
            gradients=d.read(base/(view+'_gradients.json'))
            if gradients['packets']!=30 or len({tuple(v) for v in gradients['tx_rx_coverage']})!=30:
                raise ValueError('Gradient stratum coverage differs')
    for name,data in (('action_metrics',flat),('receiver_metrics',receiver)):
        with (folder/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    groups=[]
    for view in d.EXPOSURES:
        for kind in ('linear','temporal'):
            for mode in d.RECIPE['fit_modes']:
                values=[r for r in flat if (r['view'],r['kind'],r['mode'],r['code'])==(view,kind,mode,'cross_tx_same_p_q')]
                groups.append(dict(view=view,kind=kind,mode=mode,
                    **{key+'_mean':statistics.mean(r[key] for r in values) for key in ('h_skill','z_skill','margin_mae','reliability')},
                    **{key+'_sd':statistics.stdev(r[key] for r in values) for key in ('h_skill','z_skill','margin_mae')}))
    d.write(folder/'summary.json',dict(status='VERIFIED',groups=groups,total_auxiliary_optimizer_steps=steps,
        source_only=True,identity_unchanged=True,target_accuracy=None))
    lines=['# 多解耦下一步：完整源端作用诊断','','4个固定multi E200身份骨干×clean/source practical_mid。每行1920拟合包、960独立审查包；身份骨干原训练见过这些源包，因此不是全模型未见数据测试。辅助拟合总计14400步，所有行与步骤完整。',
        '三档L/T从相同初始化和样本日程开始，固定200步；这是短预算固定参考诊断，不能等同原E200在线辅助模型的复现或新身份训练。没有使用目标评分、没有新测试集准确率。',
        '', '下表使用另一TX在**相同干预参数**下生成的q。skill=1−预测MSE/零预测MSE；以百分数展示，越大越好，负数表示不如零预测。它不是识别准确率。±为四seed样本SD。',
        '', '|源视图|分支|拟合目标|h作用skill（%）|G后作用skill（%）|margin变化MAE|',
        '|---|---|---|---:|---:|---:|']
    for g in groups:
        lines.append(f"|{g['view']}|{g['kind']}|{g['mode']}|{100*g['h_skill_mean']:.2f} ± {100*g['h_skill_sd']:.2f}|{100*g['z_skill_mean']:.2f} ± {100*g['z_skill_sd']:.2f}|{g['margin_mae_mean']:.4f}|")
    lines+=['','## R组级作用诊断','','R拟合整体条件匹配源RX变化，未扣除未经验证的L/T外推。组间没有逐包反事实配对。真实参考使用mean G(h_i)及平均逐包margin；完整产物另列质心近似误差。',
        '', '|源视图|作用|h作用skill（%）|真实组margin变化预测MAE|','|---|---|---:|---:|']
    for view in d.EXPOSURES:
        for code in ('learned','zero','fit_mean'):
            values=[r for r in receiver if r['view']==view and r['code']==code]
            lines.append(f"|{view}|{code}|{100*statistics.mean(r['h_skill'] for r in values):.2f}|{statistics.mean(r['group_margin_mae'] for r in values):.4f}|")
    lines+=['','## 解释与后续边界','',
        '源practical_mid的L分支在加入跨TX拟合后，G后skill比原目标更高；clean上没有同向改善。T没有显示同样收益。R在h空间能超过零预测，但clean组margin误差反而更大，支持继续区分作用拟合与身份方向约束。',
        '不据这轮短拟合宣称新网络提升识别性能，也不直接提高辅助权重。报告下一阶段的身份训练、真实CE与可靠性拆分、原关系/类别锚定消融和完整truth-last测试仍需独立预登记；LT强化留待组合诊断证据。',
        '完整逐seed/self/cross/shuffled/oracle/zero/mean指标见action_metrics.csv，R见receiver_metrics.csv；每行原JSON保留谱、路由、边界、组合、梯度与覆盖记录。所有负结果保留。']
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(status='VERIFIED',rows=4,exposures=2,action_metric_records=len(flat),receiver_metric_records=len(receiver),
        optimizer_steps=steps,physical_split_verified=True,metric_recount=True,identity_unchanged=True,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();print(json.dumps(summarize(a.folder)))
