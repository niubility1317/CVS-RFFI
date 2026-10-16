# 固定 additive As：同源折两种子确认

FAIL_FIXED_ADDITIVE_AS_CLOSED_AFTER_CONFIRMATION。两个种子均按原条件分别判断，均值不覆盖负确认。模型为 F＋跨 RX 同 TX 对比学习／默认 cosine 头联合训练；三组每组 scratch E20／8 L 前向，原 U 规则保持。数据为同一 source RX1-held 折，L5040／U45360／V21600，held L1260；不是 full L6300 或 L5580／Ldev720，也不是正式 target。

| seed | 固定条件通过 | 失败条件 |
|---|---|---|
| 2026092701 | True | 无 |
| 2026092702 | False | D00_clean, D00_clean_macroF1, D_raw_clean, D_raw_clean_macroF1 |

以下均为两模型种子均值 ± 样本标准差（n=2），单位 % 或百分点。相同 held 数据不构成两份独立接收机验证。

| 场景 | D00 accuracy % | D_raw accuracy % | D_As accuracy % | As−D00 pp | As−D_raw pp |
|---|---:|---:|---:|---:|---:|
| clean | 81.428571 ± 1.683588 | 80.833333 ± 0.505076 | 78.968254 ± 6.509872 | -2.460317 ± 8.193460 | -1.865079 ± 7.014948 |
| practical_high | 39.285714 ± 2.693740 | 38.214286 ± 0.729555 | 57.936508 ± 0.336718 | 18.650794 ± 2.357023 | 19.722222 ± 0.392837 |
| practical_mid | 38.968254 ± 2.244783 | 38.253968 ± 1.571348 | 54.365079 ± 0.561196 | 15.396825 ± 1.683588 | 16.111111 ± 1.010153 |
| practical_low_suburban | 36.428571 ± 1.234631 | 35.555556 ± 0.897913 | 52.301587 ± 1.122392 | 15.873016 ± 2.357023 | 16.746032 ± 2.020305 |
| practical_high_urban | 39.047619 ± 1.683588 | 38.492063 ± 0.000000 | 54.285714 ± 0.897913 | 15.238095 ± 2.581501 | 15.793651 ± 0.897913 |
| practical_mid_urban | 31.904762 ± 1.234631 | 31.309524 ± 1.627468 | 42.539683 ± 0.336718 | 10.634921 ± 0.897913 | 11.230159 ± 1.290750 |
| practical_low_urban | 27.738095 ± 0.505076 | 27.500000 ± 0.168359 | 34.960317 ± 1.627468 | 7.222222 ± 2.132544 | 7.460317 ± 1.795827 |

| suburban 场景 | D_As 与自身 clean 差距 pp | seed01 pp | seed02 pp |
|---|---:|---:|---:|
| practical_high | 21.031746 ± 6.173154 | 25.396825 | 16.666667 |
| practical_mid | 24.603175 ± 5.948676 | 28.809524 | 20.396825 |
| practical_low_suburban | 26.666667 ± 7.632264 | 32.063492 | 21.269841 |

| 相对对照 | clean Macro-F1 差 pp | urban 均值差 pp | seen V 差 pp |
|---|---:|---:|---:|
| D00 | -2.894737 ± 8.482354 | 11.031746 ± 1.870653 | 0.062500 ± 0.009821 |
| D_raw | -2.214330 ± 7.257461 | 11.494709 ± 1.328164 | 0.030093 ± 0.062199 |

## 类别与日期代价

首轮已有 TX 2、3 的明显 practical 负收益。下表保留两种子的 TX 准确率差；所有六 TX 和三 day 的分层、全部 42 个场景的 Macro-F1／混淆矩阵、修复／新错及逐种子值均在各 source_analysis.json 和本汇总 JSON 中，不以均值掩盖类别损失。

| 对照 | 场景 | TX 索引 | seed01 pp | seed02 pp | 两种子均为负 |
|---|---|---:|---:|---:|---|
| D00 | practical_high | 2 | -49.523810 | +34.761905 | False |
| D00 | practical_high | 3 | -20.952381 | +2.380952 | False |
| D00 | practical_mid | 2 | -52.380952 | +27.142857 | False |
| D00 | practical_mid | 3 | -17.142857 | +0.476190 | False |
| D00 | practical_low_suburban | 2 | -51.428571 | +23.333333 | False |
| D00 | practical_low_suburban | 3 | -10.000000 | +1.904762 | False |
| D00 | practical_high_urban | 2 | -50.476190 | +32.857143 | False |
| D00 | practical_high_urban | 3 | -25.714286 | -1.904762 | True |
| D00 | practical_mid_urban | 2 | -35.714286 | +15.238095 | False |
| D00 | practical_mid_urban | 3 | -16.666667 | -2.380952 | True |
| D00 | practical_low_urban | 2 | -23.809524 | +8.095238 | False |
| D00 | practical_low_urban | 3 | -10.476190 | +3.333333 | False |
| D_raw | practical_high | 2 | -28.095238 | +40.952381 | False |
| D_raw | practical_high | 3 | -27.619048 | +3.809524 | False |
| D_raw | practical_mid | 2 | -35.238095 | +30.476190 | False |
| D_raw | practical_mid | 3 | -26.190476 | +3.333333 | False |
| D_raw | practical_low_suburban | 2 | -35.714286 | +26.666667 | False |
| D_raw | practical_low_suburban | 3 | -19.047619 | +5.238095 | False |
| D_raw | practical_high_urban | 2 | -30.952381 | +37.619048 | False |
| D_raw | practical_high_urban | 3 | -31.904762 | +0.000000 | False |
| D_raw | practical_mid_urban | 2 | -26.190476 | +18.571429 | False |
| D_raw | practical_mid_urban | 3 | -22.857143 | +0.952381 | False |
| D_raw | practical_low_urban | 2 | -15.714286 | +13.333333 | False |
| D_raw | practical_low_urban | 3 | -12.380952 | +2.857143 | False |

## 判断边界与成本

固定候选在独立确认中失败，停止扩展，不做 seed03、换 RX、系数、场景概率、阈值或追加预算挽救。首轮收益及第二轮负结果同时保留。其他健康 owner 任务不改。

累计 6 模型／120 model epochs／26640 实际更新，实测 NN 34939912，含两个 dispatcher 各 2 个真实 CP 合成 smoke。首轮／确认分别为 17469956／17469956 NN；本地工程成本另计，不作为训练。默认头无额外拟合，训练无父权重／续训／target。

E20 来自用户指定的已暴露 clean 基准压缩探索，不声称新的独立盲测。尚未证明新增 Ag、B00／B10／B01／B11 交互项、完整源训练契约下的正式 target clean＋六 practical 或 low urban >60%／suburban 接近自身 clean。不能相加不同路线的增益预测最终成绩。


## 确认失败的具体含义

第二种子相对 D00 的 clean −8.253968pp、clean Macro-F1 −8.892667pp；相对 D_raw 的 clean −6.825397pp、Macro-F1 −7.346130pp，均违反原保护条件。urban 均值及 low urban 对两个控制均提高，六 practical、seen V 与共同 base 健康条件通过，但这不能补偿 clean 失败。

绝对 raw 监督系数 .75 保留这一实现事实，仍不足以保护跨 RX clean；不再将它当作充分条件，也不能据此断言旧失败只由监督稀释造成。此固定候选关闭扩展，首轮正结果完整保留。没有 seed03、换 RX、改系数或延长 E20 救跑。其他 owner 的健康实验不变。

第二种子 D_As 的 suburban 自身 clean 差距为 16.666667／20.396825／21.269841pp，仍远超探索 2pp；这些差距变小也包含 clean 下降 8.253968pp 的贡献，不能声称通过保护 clean 达到接近。其 clean 三个 day 均退化，TX 2 的 clean 相对 D00 下降 36.190476pp；首轮与第二轮受损 TX／场景并不完全相同，不能把均值当作各身份稳定获益。

[固定预测错误转移](error_transitions.md) 及 [全部转移聚合](error_transitions.json) 只解释同一源物理样本的描述性变化。完整 20 轮训练曲线、七视图绝对结果及全部 TX／day 图已生成并逐张查看，未新增前向。

累计六模型 120 model epochs／26640 更新，实测 NN 34939912。学生 train／eval 为 29657568／2851212；teacher train 为 0，teacher eval 仪表值 2378220，扣除继承计数 12 后为 2378208。另 held 52920 和 dispatcher smoke 4。旧本地工程 3858 合成 NN／3 合成更新另计，确认 profile 新检查 0 IQ／0 NN／0 更新，未重复旧工程测试。


两种子配对图（两点分别为两个 seed，菱形均值，误差线为 n=2 样本标准差）：[查看](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_additive_As_confirm20_20261016/final/two_seed_paired_gains.png)。
