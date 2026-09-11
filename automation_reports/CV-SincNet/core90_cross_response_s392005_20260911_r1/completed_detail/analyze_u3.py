"""Reparse full matched logs and write the U3 interpretation."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root/'all_completed_summary.json').read_text(encoding='utf-8'))
result = {}
for variant in ('U0', 'U1', 'U3'):
    records = [json.loads(s) for s in (root/variant/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    assert [r['epoch'] for r in records] == list(range(1,201))
    log = (root/variant/f'{variant}.log').read_text(encoding='utf-8')
    assert not any(marker in log for marker in ('Traceback (most recent call last)', 'CUDA out of memory'))
    keys = ('decision_gap','decision_valid_fraction','loss_decision','decision_identity_grad_norm','auxiliary_step_seconds')
    windows = {}
    for lo,hi in ((1,20),(21,79),(80,130),(131,200)):
        window = records[lo-1:hi]
        windows[f'E{lo}-{hi}'] = {k:sum(r.get('train_cross_response_'+k,0) for r in window)/len(window) for k in keys}
    result[variant] = dict(epochs=len(records),stdout_lines=len(log.splitlines()),windows=windows)
(root/'u3_log_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows = ['# U3相对ADV3B02的实现与结果解析','',
'U3=本次契约下从零训练的CORE90方法＋完整交叉块组织及前向隔离＋0.01×跨RX单侧分类裕量损失。不加载历史checkpoint，不新增响应头或部署参数。U0是方法语义恢复后的匹配基线，不是历史模型逐位复现；历史joint_safe分数不可直接作为此次性能基准。','',
'## 实现','',
'保留lite_d双主干、160维身份/域表征、CosFace、GRL/域损失、FISHR、原型与尾部几何、EMA伪标签及clean+LEO训练。U1/U3按P4×Q4×K2构造32条记录的完整块，每批4块、128条记录；K为独立物理记录，按同day/condition比较，MixStyle使用相同角色隔离。','',
'U3从同次前向的aux_id.feat_joint重新调用原分类头labels=None，取得不含训练真类margin修改、保留固定scale的logits。对每个样本及全部竞争类构造m_i,b=logit_y−logit_b。参考限定同TX、同day/condition、不同RX/物理ID且正确预测的记录；每RX至少2条独立正裕量记录，先RX内均值，再RX间中位数，至少1个合格RX。参考和有效性mask均detach。','',
'L_dec=mean_valid([m_ref−delta−m_i,b]_+²)，本次delta=0、lambda_dec=0.01。仅补低于参考的裕量，不把高裕量样本拉低；无可靠参考返回零。该损失沿真实分类路径反传至分类头和身份特征提取器，可影响共享前端；不受响应专用的门控或0.1梯度上限约束。它是训练特征上的推理语义裕量，不是eval特征保证。','',
'辅助只用有标签clean记录，包括pseudo阶段仍保留的L_s分支；不对U_s读取TX真值。response_enabled=false、identity_interaction_enabled=false、scheduler_mode=uniform，auxiliary_parameters=0；响应gate关闭是定义。决策参照使用整块其他RX，不限于响应任务的Y_S/R_S供体象限。','',
'## 匹配结果','', '|场景|U0|U1|U3|U3−U0|U3−U1|','|---|---:|---:|---:|---:|---:|']
scenes = ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
for s in (*scenes,'LEO均值'):
    vals = [summary[v]['groups'][s]['aggregate']['tx_acc'] if s in scenes else sum(summary[v]['groups'][t]['aggregate']['tx_acc'] for t in scenes[1:])/3 for v in ('U0','U1','U3')]
    rows.append('|'+s+'|'+'|'.join(f'{x:.4f}' for x in (*vals,vals[2]-vals[0],vals[2]-vals[1]))+'|')
rows += ['', 'U3−U0包含组织/隔离与决策损失；U3−U1才是匹配条件下新增决策任务的描述性差值。U3相对U1改善clean、降低三LEO总体均值；不能说全部2.6263个百分点clean增益来自决策损失。','',
'## 未知日期下的最弱RX','', '|场景|U0下限|U1下限|U3下限|U3−U1|','|---|---:|---:|---:|---:|']
for s in scenes:
    vals=[summary[v]['derived'][s]['worst_strict_receiver'] for v in ('U0','U1','U3')]
    rows.append('|'+s+'|'+'|'.join(f'{a:.4f}(RX{r})' for a,r in vals)+f'|{vals[2][0]-vals[1][0]:+.4f}|')
rows += ['', 'U3在三LEO总体均值略降的同时，提高了未知日期最弱RX的下限。这符合跨RX弱侧裕量保护的动机，但仅是同seed结果的一致性证据，不证明因果中介或多seed稳定性。下限为各模型独立取最小值；clean的U0和U3最弱RX不同。','',
'## 全日志与成本','', '|变体|小时|峰值allocated GiB|成功更新|决策梯度步|','|---|---:|---:|---:|---:|']
for v in ('U0','U1','U3'):
    d=summary[v]; c=d['activation']['counts']; r=d['resource']
    rows.append(f"|{v}|{r['wall_time_seconds']/3600:.4f}|{r['peak_cuda_memory_allocated_bytes']/2**30:.4f}|{c['successful_steps']}|{c['decision_gradient_steps']}|")
rows += ['', '完整重读三项共600轮及全部stdout；U3有9787个有效决策batch、9764个非零身份决策梯度步、9791次成功更新/9800计划步。没有OOM/Traceback。墙钟受同机并发影响，不是隔离速度基准；决策实现含样本/竞争类循环、元数据CPU转换与小分类头/额外梯度诊断，存在训练开销，尚无隔离profile支持瓶颈百分比。','',
'|U3区间|有效参照比例|平均正缺口|未加权决策loss|身份梯度诊断范数|辅助秒/步|','|---|---:|---:|---:|---:|---:|']
for w,d in result['U3']['windows'].items():
    rows.append('|'+w+'|'+'|'.join(f'{d[k]:.5f}' for k in ('decision_valid_fraction','decision_gap','loss_decision','decision_identity_grad_norm','auxiliary_step_seconds'))+'|')
rows += ['', '这些是区间内epoch均值的平均；缺口参照随模型与批次改变，不是冻结验证集曲线，不能仅凭下降证明目标泛化。','',
'## 证据边界与来源','',
'本次确认代码、冻结开关、真实梯度、评分和成本；没有新增实现或重训。U3针对CR10/CR11/CR13及共同采样CR02/CR03/CR17的设计要求已实现；完整20项本地追踪见原实施计划，不能将其全部归入U3。最高剩余风险是单seed泛化/稳健性证据不足；Phase2少样本、新类注册、unknown拒识及真实在轨收益未测。','',
'代码：code/cvsrffi/cross_response/decision.py、integration.py、training.py；基线审计：analysis/CORE90_CROSS_RESPONSE_BASELINE_AUDIT.md；设计：analysis/ADV3B02_CORE90_CROSS_RESPONSE_IMPLEMENTATION_PLAN_20260911.md；评分：all_results.md与all_scores_long.csv；重读脚本：analyze_u3.py，日志摘要：u3_log_analysis.json。','']
(root/'u3_analysis.md').write_text('\n'.join(rows),encoding='utf-8')
assert '\ufffd' not in (root/'u3_analysis.md').read_text(encoding='utf-8')
print(json.dumps(result['U3'],ensure_ascii=False,indent=2))
