# Mitigating Receiver Impact DA纸面默认修复记录

## 范围

- 方法：`mitigating_receiver_impact_da`
- 论文：`Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation`
- 修复提交：
  - `4b54313 fix: align mitigating da paper defaults`
  - `461003f fix: use standard resnet18 widths for mitigating da`
  - `760cd50 fix: keep mitigating da kl estimator path consistent`
- 完整实验报告：`E:\type10-7\automation_reports\CV-SincNet\mitigating_receiver_impact_da_paperdefaults_14-7_to_3-19_20260708_fix4b54313\report.md`

## 对齐结论

| 项目 | 论文设置 | 代码修复后默认 |
|---|---|---|
| 初始伪标签阈值 | `tau=0.7` | `--base-tau 0.7` |
| Table II类别先验 | Eq.9使用`p_prior(k)`，论文未给平滑/裁剪/归一化 | `--class-prior-mode uniform`，source先验保留为显式选项 |
| Eq.9类别权重 | 原始`p_prior/estimated_target_frequency`比例 | 默认无平滑、无裁剪、无均值归一化 |
| 稳定化类别权重 | 论文未提及 | 仅保留为显式可选参数 |
| 估计网络更新频率 | `m=7` | `--estimate-steps 7` |
| KL权重 | `lambda=0.005` | 默认`0.005` |
| 伪标签损失权重 | `mu=0.5` | 默认`0.5` |
| 特征提取器宽度 | ResNet18 1D版本 | 默认`base_channels=64`，最终特征维度512 |
| T步/E-C步KL路径 | Algorithm 1中同一`zeta(theta_T,theta_E)` | T步和E/C步使用一致train计算路径；T步恢复BN buffer避免持久改变E状态 |

## 验证

```powershell
conda activate ssr-gpu
python -m pytest tests/test_mitigating_receiver_impact_da.py -q
python -m py_compile paper_reproduction/mitigating_receiver_impact_da/losses.py paper_reproduction/mitigating_receiver_impact_da/algorithm.py paper_reproduction/mitigating_receiver_impact_da/train.py tests/test_mitigating_receiver_impact_da.py
git diff --check -- paper_reproduction/mitigating_receiver_impact_da/losses.py paper_reproduction/mitigating_receiver_impact_da/algorithm.py paper_reproduction/mitigating_receiver_impact_da/train.py tests/test_mitigating_receiver_impact_da.py
```

- 单测：23项通过。
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

## 2026-07-08追加定位结果

| run | 关键差异 | source_only | proposed | proposed相对论文92.42% | 诊断 |
|---|---|---:|---:|---:|---|
| `resnet18std_full_e5` | 标准ResNet18宽度，full数据，旧KL eval一致路径 | 18.08% | 63.40% | -29.02pp | full数据显著优于m200子集，但未收敛到论文 |
| `resnet18std_full_e10` | 同上，10epoch | 28.35% | 21.76% | -70.66pp | source_only接近论文30.25%，proposed随epoch退化 |
| `ec_kl_trainpath_full_e5` | 仅E/C步KL切train，T步仍eval | 17.25% | 18.13% | -74.29pp | `loss_kl_mean`大幅负值，证明T/E-C路径不一致会崩塌 |
| `kl_consistent_full_e5` | T步和E/C步KL路径一致，T步恢复BN buffer | 17.65% | 37.93% | -54.49pp | KL恢复正值且稳定，但仍低于论文 |

当前结论：

- 已对齐论文明确给出的`tau=0.7`、`m=7`、`lambda=0.005`、`mu=0.5`。
- 论文未提及类别权重平滑、裁剪或均值归一化；这些默认关闭，只作为显式消融开关存在。
- source-only full e10达到28.35%，距离论文source-only 30.25%为-1.90pp，说明数据任务、基础模型宽度和source训练已基本接近。
- proposed差距主要不再来自上述默认超参，而来自GAD/CPL训练动态、BN路径、预处理/裁剪细节、随机种子稳定性和论文未公开的收敛/模型选择策略。
