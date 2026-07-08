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
