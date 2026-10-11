"""Exact candidate-selection counterfactual at saved source model states only."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from .archive_evidence import read_expanded


def build(value):
    if not value['source_only'] or value['target_prediction_or_scores_read'] or value['errors']:
        raise ValueError('Verified source-only evidence required')
    rows=[]
    for row in value['rows']:
        if row['arm'] not in ('HARD_STYLE','JOINT'):
            continue
        if row['status']!='COMPLETE' or len(row['style_records'])!=3996:
            raise ValueError('All active source candidate records required')
        samples=[]
        for record in row['style_records']:
            d=record['details']; clean=d['eval_clean']; selected=d['eval_selected']
            if (d['style_selection']!='hard' or d['source_role']!='L_s' or d['target_access'] or d['V_access']
                    or len(d['candidate_losses'])!=4 or d['eval_readouts']!=5 or not d['candidate_state_unchanged']):
                raise ValueError('Candidate source contract differs')
            if not math.isclose(max(d['candidate_losses']),selected['CE'],rel_tol=1e-7,abs_tol=1e-8):
                raise ValueError('Recorded selection is not candidate maximum')
            fallback=clean['CE']>selected['CE']
            picked=clean if fallback else selected  # Exact ties retain the original candidate.
            samples.append(dict(fallback=fallback,clean_CE=clean['CE'],old_CE=selected['CE'],new_CE=picked['CE'],
                old_accuracy_delta=selected['accuracy']-clean['accuracy'],new_accuracy_delta=picked['accuracy']-clean['accuracy'],
                old_margin_delta=selected['margin']-clean['margin'],new_margin_delta=picked['margin']-clean['margin']))
        result=dict(row_id=row['row_id'],arm=row['arm'],seed=row['initialization']['seed'],active_batches=len(samples),
            fallback_count=sum(s['fallback'] for s in samples),fallback_fraction=mean(s['fallback'] for s in samples),
            clean_CE=mean(s['clean_CE'] for s in samples),old_selected_CE=mean(s['old_CE'] for s in samples),
            identity_inclusive_CE=mean(s['new_CE'] for s in samples),
            original_CE_increment=mean(s['old_CE']-s['clean_CE'] for s in samples),
            identity_inclusive_CE_increment=mean(s['new_CE']-s['clean_CE'] for s in samples),
            added_CE=mean(s['new_CE']-s['old_CE'] for s in samples),
            old_accuracy_delta_pp=100*mean(s['old_accuracy_delta'] for s in samples),
            identity_inclusive_accuracy_delta_pp=100*mean(s['new_accuracy_delta'] for s in samples),
            old_margin_delta=mean(s['old_margin_delta'] for s in samples),
            identity_inclusive_margin_delta=mean(s['new_margin_delta'] for s in samples),
            extra_forward_calls_for_selection=0,prospective_training_or_generalization_measured=False)
        if result['identity_inclusive_CE_increment'] < -1e-12 or result['added_CE'] < -1e-12:
            raise ValueError('Identity-inclusive maximum unexpectedly decreased CE')
        rows.append(result)
    output=dict(status='VERIFIED',source_only=True,target_access=False,source_run=value['run_id'],source_read_at=value['read_at'],
        source_matrix_status=value['status'],scope='frozen recorded source states; exact selection readout, not a new training trajectory',rows=rows)
    lines=['# clean 恒等候选的源域反事实核算','',
        '本分析逐条读取已经保存的四候选与 clean 的源 L eval 读出，仅改变同一记录内的选择规则：原规则取四候选最大 CE；反事实取 clean 与该最大值中较大者，精确平局保留原候选。未重新推理、更新权重、读取目标数据或改动在训任务。','',
        '|行|回到 clean 的批次|原 CE 增量|含 clean 的 CE 增量|额外增加 CE|原准确率差（百分点）|含 clean 的准确率差（百分点）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['row_id']}|{100*r['fallback_fraction']:.2f}%（{r['fallback_count']}/{r['active_batches']}）|{r['original_CE_increment']:+.6f}|{r['identity_inclusive_CE_increment']:+.6f}|{r['added_CE']:+.6f}|{r['old_accuracy_delta_pp']:+.4f}|{r['identity_inclusive_accuracy_delta_pp']:+.4f}|")
    lines += ['',
        '所有增量均相对于同一批次的 clean eval 读出。加入 clean 能在这些固定状态下保证所选整批 CE 不低于 clean，但会放弃对应批次的风格扰动。它不生成新的困难样本，也不扩大候选覆盖；现有搜索已计算 clean，因此选择阶段无需增加前向。', '',
        '该表不是训练消融结果。若真的使用新规则，梯度与后续参数轨迹都会改变，之后的候选不再等于此处保存的候选。eval 最大 CE 不保证 train/dropout 下最大 CE，也不保证错误率或域泛化改善；不能用这些数字预测最终测试准确率。', '',
        '因此，恒等候选可以修正“选到比 clean 更容易的视图”，但它本身没有证明能解决困难覆盖不足。现有固定实验继续完成；这项反事实仅用于源域机制解释，没有发布新增训练或按目标成绩选模。','']
    return output,'\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,text=build(read_expanded(a.source));a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'source_identity_counterfactual.md').write_text(text,encoding='utf-8')
    (a.output/'evidence/source_identity_counterfactual.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
