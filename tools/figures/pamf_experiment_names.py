"""Publication names for frozen PAMF experiments; original IDs stay immutable."""
from __future__ import annotations
import csv
from pathlib import Path
import textwrap

PROTOCOL_EN = {
    'F0': 'Main receiver split', 'F1': 'Alternative receiver split 1',
    'F2': 'Alternative receiver split 2', 'ManyRx': '16-source / 16-target receivers',
    'chrono': 'Later-day receiver transfer', 'label01': '1% labeled training samples',
    'label05': '5% labeled training samples',
}
PROTOCOL_CN = {
    'F0': '主接收机划分', 'F1': '替代接收机划分一', 'F2': '替代接收机划分二',
    'ManyRx': '16 源／16 目标接收机', 'chrono': '未见接收机与后续日期',
    'label01': '训练池标注比例 1%', 'label05': '训练池标注比例 5%',
}
for _m in (1, 2, 3, 4, 6):
    for _budget, _en, _cn in [('natural', 'scaled label budget', '随源数量增加标注预算'),
                              ('fixed', 'fixed label budget', '固定标注预算')]:
        _key = f'm{_m}-{_budget}'
        PROTOCOL_EN[_key] = f'{_m} source receiver'+('' if _m == 1 else 's')+f'; {_en}'
        PROTOCOL_CN[_key] = f'{_m} 个源接收机；{_cn}'

REGIME_EN = {'native_cric': 'PAMF recognizer', 'matched_sat_supervised': 'Supervised control',
             'matched_sat_plain_pl': 'Plain pseudo-label control'}
READOUT_EN = {'raw': 'No restoration', 'DURA-L': 'DURA (labeled pool)', 'DURA-LU': 'DURA (all source IQ)'}
SYSTEM_EN = {'raw': 'Recognizer only', 'DURA-L': 'PAMF (labeled-pool restorer)',
             'DURA-LU': 'PAMF (full system)'}
CORE_EN = {'111': 'Complete recognizer', '011': 'Remove cross-receiver contrast',
    '101': 'Remove paired consistency', '110': 'Remove selective self-training',
    '000': 'Remove all three mechanisms', '001': 'Selective self-training only',
    '010': 'Paired consistency only', '100': 'Cross-receiver contrast only'}
CORE_CN = {'111': '完整识别器', '011': '去除跨接收机对比学习', '101': '去除成对预测一致性',
    '110': '去除选择性自训练', '000': '去除全部三类机制', '001': '仅保留选择性自训练',
    '010': '仅保留成对预测一致性', '100': '仅保留跨接收机对比学习'}
FINE_EN = (
    'Shallow-layer contrast only', 'Intermediate-layer contrast only', 'Final-layer contrast only',
    'Remove paired consistency', 'One stochastic pass per view pair', 'One signature view',
    'Remove time coordinates', 'Remove signature branch', 'Remove amplifier branch',
    'Remove reference branch', 'Student replaces EMA teacher', 'Remove adjacent-sample filter',
    'Remove strong-view filter', 'Remove unlabeled entropy loss', 'CE + channel augmentation recipe',
    'Remove cross-receiver contrast', 'Raw views replace channel views', 'Add channel-view consistency',
    'Remove raw-view CE pair', 'Remove unlabeled self-training', 'Plain pseudo-label replacement')
FINE_CN = (
    '对比学习仅作用于浅层', '对比学习仅作用于中层', '对比学习仅作用于末层',
    '去除成对预测一致性', '每对视图仅一次随机前向', '签名分支仅保留一个视图',
    '去除时间坐标', '去除签名分支', '去除功放特征分支', '去除参考响应分支',
    '当前学生替代 EMA 伪标签教师', '去除相邻样本一致性筛选', '去除强增强视图筛选',
    '去除无标签熵损失', '交叉熵＋信道增强配方', '去除跨接收机对比学习',
    '原始视图替代模拟信道视图', '增加信道视图一致性损失', '移除原始视图交叉熵对',
    '去除无标签自训练', '普通伪标签替代选择性自训练')
ARCHS = ('X3_CE','SIG_TIME','NO_TIME','SHUFFLE_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE','COV_X3')
ARCH_EN = dict(zip(ARCHS, ('Complete supervised architecture', 'One signature view',
    'Remove time coordinates', 'Shuffle time coordinates', 'Remove signature branch',
    'Remove amplifier branch', 'Remove reference branch', 'Perturb reference response',
    'Covariance replaces signature')))
ARCH_CN = dict(zip(ARCHS, ('完整监督结构', '签名分支仅保留一个视图', '去除时间坐标',
    '打乱时间坐标', '去除签名分支', '去除功放特征分支', '去除参考响应分支',
    '扰动参考响应', '协方差替代签名表征')))
G_EN = {'raw': 'Recognizer only (no restorer)', 'default': 'Complete DURA',
    'coarse_only': 'Coarse correction only', 'prior_no_CG': 'Remove conjugate-gradient solver',
    'no_image_action': 'Remove mirror correction', 'no_innovation_input': 'Remove innovation input',
    'copy_no_KD': 'Copied weights, no distillation', 'random_with_KD': 'Random weights + distillation',
    'random_no_KD': 'Random weights, no distillation'}
G_CN = {'raw': '仅识别器，不使用恢复器', 'default': '完整 DURA 恢复器',
    'coarse_only': '仅保留粗校正', 'prior_no_CG': '去除共轭梯度求解器',
    'no_image_action': '去除镜像校正作用', 'no_innovation_input': '去除创新残差输入',
    'copy_no_KD': '复制初始化，不蒸馏', 'random_with_KD': '随机初始化并蒸馏',
    'random_no_KD': '随机初始化，不蒸馏'}
FAIR_EN = {'Sup+G-L': 'Supervised recognizer\nDURA: labeled pool', 'PL+G-L': 'Pseudo-label recognizer\nDURA: labeled pool',
    'Sup+G-LU': 'Supervised recognizer\nDURA: all source IQ', 'PL+G-LU': 'Pseudo-label recognizer\nDURA: all source IQ'}
FAIR_CN = {'Sup+G-L': '监督识别器＋标注池恢复器', 'PL+G-L': '普通伪标签识别器＋标注池恢复器',
    'Sup+G-LU': '监督识别器＋全源池恢复器', 'PL+G-LU': '普通伪标签识别器＋全源池恢复器'}


def wrap(label, width=28):
    return '\n'.join('\n'.join(textwrap.wrap(line, width=width, break_long_words=False)) for line in label.split('\n'))


def export_names(root):
    records = []
    def add(group, key, en, cn, reference):
        records.append(dict(group=group, original_id=key, figure_name=en, chinese_name=cn, comparison_reference=reference))
    for k in PROTOCOL_EN: add('data_condition', k, PROTOCOL_EN[k], PROTOCOL_CN[k], 'Same frozen receiver/date/label contract; no cross-contract pooled mean.')
    for k in CORE_EN: add('core_ablation', k, CORE_EN[k], CORE_CN[k], 'Complete recognizer; same CE dose, E50 and frozen restorer.')
    for i, (en, cn) in enumerate(zip(FINE_EN, FINE_CN), 1):
        note = 'Final T1-E50 recognizer, same frozen DURA; one registered change.'
        if i in (5, 15, 19, 20, 21): note += ' Training recipe, forward count or loss dose also changes; not a single-operator deletion.'
        add('recognizer_ablation', f'F{i:02d}', en, cn, note)
    for k in ARCHS: add('supervised_structure', k, ARCH_EN[k], ARCH_CN[k], 'Complete supervised architecture; CE + satellite augmentation, no U/EMA/CRIC/RDrop, E50.')
    for k in G_EN: add('restorer_ablation', k, G_EN[k], G_CN[k], 'Same recognizer. Default: one fixed G; new student variants: three paired F/G seeds.')
    for k in FAIR_EN: add('unlabeled_data_control', k, FAIR_EN[k].replace('\n', '; '), FAIR_CN[k], 'Ordinary pseudo-label factorial; separate from native T1 and supervised-regularized F20.')
    for k in READOUT_EN:
        add('restoration_condition', k, READOUT_EN[k], {'raw':'不使用恢复器','DURA-L':'恢复器仅使用标注源样本池 IQ','DURA-LU':'恢复器使用标注与无标签源样本池 IQ'}[k], 'L/U indicate IQ pools. The restorer does not use TX labels, F outputs or target data.')
    with (root/'experiment_names.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# 实验命名与比较对象', '',
        '图中的实验名称描述实际操作。原始 ID 只用于追溯，不再作为图例或坐标轴上的唯一名称。原 run ID、目录、权重及评分不改名。', '',
        'PAMF 完整系统＝最终识别器 T1-E50＋DURA v1.0。识别器单独使用时标为 Recognizer only。', '',
        'DURA (labeled pool) 表示恢复器谱系只取 L_s 样本池中的 IQ；DURA (all source IQ) 表示取 L_s＋U_s 的 IQ。两者均不读取 TX 标签。仅限制恢复器使用 L_s，不代表整个 PAMF 不使用 U_s。', '',
        'scaled label budget：L/V 标注样本数随源接收机数量增加。fixed label budget：L＝1080、V＝5400，U 随源数量增加；不是固定全部数据量。1% 和 5% 的分母是 L＋U 训练池，验证标签成本另计。', '',
        '恢复与训练对比图均同时展示两端准确率；增幅只作辅助。Change (pp) 是同一 seed 内差值的均值±样本标准差，不能由两端 SD 相减得到。', '',
        '| 实验组 | 原始 ID | 新名称 | 图中英文名称 |', '|---|---|---|---|']
    lines += [f'| {r["group"]} | {r["original_id"]} | {r["chinese_name"]} | {r["figure_name"]} |' for r in records]
    (root/'实验命名说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return records
