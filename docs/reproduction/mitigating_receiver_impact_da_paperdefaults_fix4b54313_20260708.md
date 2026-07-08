# Mitigating Receiver Impact DA纸面默认修复记录

## 范围

- 方法：`mitigating_receiver_impact_da`
- 论文：`Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation`
- 修复提交：`4b54313 fix: align mitigating da paper defaults`
- 完整实验报告：`E:\type10-7\automation_reports\CV-SincNet\mitigating_receiver_impact_da_paperdefaults_14-7_to_3-19_20260708_fix4b54313\report.md`

## 对齐结论

| 项目 | 论文设置 | 代码修复后默认 |
|---|---|---|
| 初始伪标签阈值 | `tau=0.7` | `--base-tau 0.7` |
| Table II类别先验 | 默认均匀先验 | `--class-prior-mode uniform` |
| Eq.9类别权重 | 原始`p_prior/estimated_target_frequency`比例 | 默认无平滑、无裁剪、无均值归一化 |
| 稳定化类别权重 | 论文未提及 | 仅保留为显式可选参数 |
| 估计网络更新频率 | `m=7` | `--estimate-steps 7` |
| KL权重 | `lambda=0.005` | 默认`0.005` |
| 伪标签损失权重 | `mu=0.5` | 默认`0.5` |

## 验证

```powershell
conda activate ssr-gpu
python -m pytest tests/test_mitigating_receiver_impact_da.py -q
python -m py_compile paper_reproduction/mitigating_receiver_impact_da/losses.py paper_reproduction/mitigating_receiver_impact_da/algorithm.py paper_reproduction/mitigating_receiver_impact_da/train.py tests/test_mitigating_receiver_impact_da.py
git diff --check -- paper_reproduction/mitigating_receiver_impact_da/losses.py paper_reproduction/mitigating_receiver_impact_da/algorithm.py paper_reproduction/mitigating_receiver_impact_da/train.py tests/test_mitigating_receiver_impact_da.py
```

- 单测：22项通过。
- 编译：通过。
- `diff --check`：通过。

## N607验证边界

计划运行`mitigating_receiver_impact_da_paperdefaults_14-7_to_3-19_20260708_fix4b54313`，只验证修复后默认设置和短跑方向。该运行使用5个epoch和50样本/组合，不声明完整Table II复现。

## N607短诊断结果

| 方法 | 任务 | 设置 | 目标准确率 | 相对source_only | 相对论文Table II proposed 92.42% |
|---|---|---|---:|---:|---:|
| source_only | `14-7->3-19` | 5epoch，50样本/组合 | 23.58% | 0.00pp | - |
| proposed | `14-7->3-19` | `base_tau=0.7`，`class_prior_mode=uniform`，无类别权重稳定化默认项 | 42.33% | +18.75pp | -50.09pp |

同一行诊断显示，source预训练后的目标域预测准确率为18.25%，`tau=0.7`高置信伪标签覆盖率为94.00%，但选择正确率只有18.00%。因此修复后的短诊断已恢复`proposed > source_only`，但距离论文Table II仍大，主要剩余风险是本次运行不是完整收敛训练，且初始模型`h0`在目标receiver上的伪标签质量仍偏低。
