# PrototypeTransport r02 完整训练日志解释

日期：2026-09-30。run：`20260930-phase2-d92-prototype-transport-support-m2-r02`。release commit：`9ebdcd7dd7ecd804c440111d3a2f40d3f6685d27`。

**训练执行未见技术异常；936 个有信息阶段均降低数据损失和总目标，但内部准确率的平均提升不足 1 个百分点，且 122 个阶段下降。** 这说明冻结优化器确实执行了监督目标更新，同时说明目标下降不能替代准确率改善。本文只解释训练过程，不判断外层效果，也不据此选参、选路线或改变预算。

## 证据与统计口径

唯一证据目录为 [r02 training_diagnostics](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics)。root 已完成原始训练日志扫描，派生产物状态为 `COMPLETE_TRANSPORT_TRAINING_LOG_SCAN_VERIFIED`。本次完整解析 [training_diagnostics.json](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/training_diagnostics.json)、`summary.json` 和目录内全部 10 个 CSV，读取 `report.md`。没有打开 parent held-result、support_summary、query、ABC、registry、handoff 或历史评分。

[stages.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/stages.csv) 包含全部 4576 个状态；[stage_curves.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/stage_curves.csv) 包含全部 13543 个训练位置：936 个 initial、3728 个 gradient、4303 个 trial、936 个 final_cached 和 3640 个 no_information。accepted STEP 事件数为 3656，更新指标在对应 accepted trial 上统计，未把重复展示的梯度算作额外反向。

“有信息阶段”表示物理训练 K≥2 且存在多个注册类；“有更新阶段”表示至少接受一次更新。本次两者恰好均为 936，零更新有信息阶段为 0，两种口径仍不能互换。3640 个无信息阶段均以 `PHYSICAL_K1` 停止，优化目标和内部准确率为 N/A；没有进行目标前向、反向或试探，但准备、状态绑定和评分仍有实际成本。160 个 parent 中 40 个 true K1 parent 仅做数值检查，不把它们当独立训练验证。

以下损失和准确率均先按阶段等权平均；不把不同物理样本数的阶段合并成样本加权准确率。B 保留重复实际拟合，另列物理绑定去重结论。内部准确率来自监督优化使用的 inner held 位置，属于训练诊断，不能称为独立验证。

## 初始至最终目标

| 模式 | 全部状态 / 有信息阶段 | 数据损失 initial→final | proximal initial→final | 总目标 initial→final | 总目标平均变化 | 内部准确率 initial→final | 平均变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 1760 / 360 | 0.367307→0.360853 | 0→0.002486 | 0.367307→0.363339 | −0.003968 | 68.664%→69.435% | +0.771 个百分点 |
| C_seq | 1408 / 288 | 0.438836→0.434499 | 0→0.001225 | 0.438836→0.435723 | −0.003113 | 56.864%→57.362% | +0.498 个百分点 |
| C_reset | 1408 / 288 | 0.442205→0.436365 | 0→0.001470 | 0.442205→0.437835 | −0.004370 | 56.742%→57.261% | +0.519 个百分点 |

936 个阶段的数据损失和总目标全部下降。总目标的阶段均值相对下降分别为 1.080%、0.709% 和 0.988%；这些是内部目标变化，不是准确率的相对提升。proximal 上升表示偏离各自 anchor 的代价：B 和 C_reset 从零参数出发，C_seq 从继承 B 的参数出发且以继承参数为 anchor，所以 C_seq 的 initial proximal 也为 0。两种 C 路线的 proximal 参照点不同，不能将其总目标直接视为同一目标函数的路线排名。

| 模式 | 内部准确率提高 / 不变 / 下降 | 单阶段准确率变化范围 |
|---|---:|---:|
| B | 95 / 250 / 15 | −5.556 至 +11.111 个百分点 |
| C_seq | 127 / 107 / 54 | −6.250 至 +9.091 个百分点 |
| C_reset | 125 / 110 / 53 | −4.167 至 +9.091 个百分点 |

总计 347 个阶段提高、467 个不变、122 个下降。surrogate 数据损失下降并不保证分类正确数上升；正确数不变也不能证明逐样本预测完全不变，因为日志没有记录 prediction transitions。

## 试探、停止与梯度

| 模式 | 接受 / 拒绝 trial | 接受率 | MAX_ITERATIONS / ARMIJO_BUDGET_EXHAUSTED | 每阶段接受更新数分布 |
|---|---:|---:|---:|---|
| B | 1380 / 395 | 77.746% | 315 / 45 | 4 次：315；3 次：35；2 次：5；1 次：5 |
| C_seq | 1134 / 158 | 87.771% | 271 / 17 | 4 次：271；3 次：16；2 次：1 |
| C_reset | 1142 / 94 | 92.395% | 278 / 10 | 4 次：278；3 次：10 |

每阶段最多 4 次迭代、每次最多 3 次 Armijo trial。864 个有信息阶段到达迭代上限，72 个耗尽当次试探预算；没有 `ZERO_GRADIENT`、`ZERO_PROJECTED_STEP` 或单类停止。预算停止不是技术故障，也不是收敛证据。全部阶段最后一次已记录梯度范数均非零；全部阶段首至末次梯度范数下降。

| 模式 | 首次→末次梯度范数均值 | 全部实际反向的梯度范数范围 | β / η 梯度范数均值 | accepted update 范数范围 |
|---|---:|---:|---:|---:|
| B | 0.016633→0.004182 | 0.000109 至 0.040164 | 0.010203 / 0.000147 | 0.029790 至 0.125000 |
| C_seq | 0.011945→0.003961 | 0.000192 至 0.031780 | 0.007681 / 0.000321 | 0.009035 至 0.125000 |
| C_reset | 0.014887→0.005895 | 0.000160 至 0.038596 | 0.010298 / 0.000120 | 0.019724 至 0.125000 |

accepted step_size 为 0.125 / 0.0625 / 0.03125，次数分别为 B 的 1200 / 100 / 80、C_seq 的 1058 / 45 / 31、C_reset 的 1094 / 32 / 16。accepted trial 的 Armijo slack 均为正，最小值分别为 3.501×10⁻⁷、6.563×10⁻⁷、1.875×10⁻⁶；rejected trial 的 slack 均为负。没有把 rejected 参数作为最终参数，也没有选择历史最好步。

B 和 C_reset 的全部首次 η 梯度为零，但 β 梯度非零；后续 η 梯度参与更新。C_seq 首次 η 梯度全部非零。日志支持旋转参数 β 主导本次更新，不能据此断言 η 永远无效或删除 η。

## 参数状态

θ 含 10 个存储参数、9 个有效自由度。下表坐标范围覆盖 initial、全部 accepted/rejected trial 和 final_cached；norm 与 anchor 距离只统计最终有信息阶段。

| 模式 | β 全轨迹坐标范围 | η 全轨迹坐标范围 | 最终 θ norm 均值 / 最大值 | 最终 η norm 均值 / 最大值 | 最终距 anchor 均值 / 最大值 |
|---|---:|---:|---:|---:|---:|
| B | −0.495083 至 0.464872 | −0.024969 至 0.029682 | 0.416682 / 0.499866 | 0.006760 / 0.036986 | 0.416682 / 0.499866 |
| C_seq | −0.785398 至 0.785398 | −0.051875 至 0.069440 | 0.727792 / 0.988059 | 0.019642 / 0.088722 | 0.443989 / 0.499908 |
| C_reset | −0.492967 至 0.485566 | −0.028651 至 0.040594 | 0.472945 / 0.499947 | 0.006090 / 0.036697 | 0.472945 / 0.499947 |

C_seq 的 29 个最终状态有 β 坐标达到冻结的 ±π/4 边界；B 和 C_reset 均没有最终 β 饱和，所有 η 坐标均远离 ±0.346574 边界。达到投影边界是约束内行为，不能凭此解释为数值异常。全轨迹最大 |sum(η)| 为 4.921140×10⁻¹⁴，小于冻结浮点容差 4.925107×10⁻¹⁴。

## B 去重及实际成本

[B_physical_training_groups.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/B_physical_training_groups.csv) 的 352 个绑定均有 5 次实际拟合、1 个最终参数版本。绑定按 row、scope、parent K、train K、class IDs 和有序物理 support IDs 区分。其中 72 个绑定有信息，280 个为无信息 proxy。去重后 B 的内部准确率提高 / 不变 / 下降为 19 / 50 / 3，停止为 63 次迭代上限和 9 次试探预算耗尽；损失与准确率均值与上述 B 相同，因为每个绑定等量重复 5 次。360 次 B 更新过程不能解释为 360 个独立证据。

1408 个 B 重复实际上下文仍消耗资源，未从总成本删除。另有 [B_reuse_contexts.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/B_reuse_contexts.csv) 的 2816 个 C 继承/引用上下文，它们没有再计成 B 拟合。按代表上下文取一次得到的 B fit 时间为 52.390 s，仅是去重统计，不是实际运行耗时。

| 模式 | 有信息 fit 时间均值 / 范围（s） | 全状态 fit 累计（s） | 无信息 fit 累计（s） | 全状态已记录 score 累计（s） |
|---|---:|---:|---:|---:|
| B | 0.719 / 0.209 至 1.871 | 261.596 | 2.677 | 51.382 |
| C_seq | 3.826 / 0.290 至 21.945 | 1106.884 | 4.868 | 171.983 |
| C_reset | 3.446 / 0.267 至 19.728 | 997.311 | 4.899 | 171.949 |

有信息阶段实际 train K 为 3、4、6、7、13、14。B 的 train K=3 / 14 平均 fit 时间为 0.252 / 1.442 s，C_seq 为 0.749 / 8.984 s，C_reset 为 0.667 / 8.088 s；C 还同时改变注册类数，不能把差异全部归因于 K。3168 次准备累计 258.754 s，其中原型构造记录 45.969 s；baseline head fit 累计 50.753 s。原型时间属于准备内部测量，baseline binding 等内部计时也可能嵌套，不把这些量全部再次相加。initial objective 的完整前向时间和 final_cached 的缓存归约时间不同，二者之差不能称为优化加速。

实际资源计数如下，均保留拒绝 trial、初始目标和缓存反向；预算上界没有被当成实测次数。

| 项目 | B | C_seq | C_reset | 总计 |
|---|---:|---:|---:|---:|
| 初始 + trial 目标新前向 | 2135 | 1580 | 1524 | 5239 |
| 实际反向 / 尝试迭代 | 1425 | 1151 | 1152 | 3728 |
| 内层 head fit / 分解 | 6405 | 4740 | 4572 | 15717 |
| 最终 head fit / 分解 | 360 | 288 | 288 | 936 |
| 伴随三角 solve | 8550 | 6906 | 6912 | 22368 |
| transport forward 调用 | 6765 | 5028 | 4860 | 16653 |

4303 个 trial attempt 均有完成 trial，等于 3656 accepted + 647 rejected。加上 3168 个 baseline head，总 head fit 和分解各为 19821。原型构造 5112 次、原型距离调用 15336 次、prepared 距离调用 10224 次；最终 score 调用 7744 次、处理物理记录 1020600 条。额外 diagnostic fit 为 0。

所有阶段 adapter 常驻状态为 80 B，优化器常驻状态记录为 0 B；后者不表示没有临时优化数组。B 的完整 persistent state 为 177192 至 1560600 B，C_seq/C_reset 为 236344 至 6820440 B，包含 head、prototype、adapter 和 lineage；不能用 80 B 代替完整状态。B 原型状态为 70664 B，C 为 94216 至 306184 B。各状态字节的累计和不是同时常驻内存。

四 row 均为 Linux x86_64、float64、CPU 执行，逻辑 CPU 数记录为 96，OMP/OpenBLAS/MKL 线程设置各为 2，`gpu_use=false`。row wall 时间为 874.928 至 928.261 s；并行 row 的时间不能相加当作整体 elapsed。峰值进程 RSS 为 638234624 至 693604352 B，包含该进程的特征缓存和其他状态，不是独立方法增量峰值。具体 CPU 型号、独立方法峰值、星载硬件实测和新增部署传输字节为 N/A。

## 可解释的不足与边界

结构化技术失败和显式错误字段均为 0，四 row 没有 failure artifact；完整文本扫描的 error、warning、traceback、OOM/Killed、nonfinite 和 recovery/resume 标记均为 0。参数边界和 Armijo accept/reject 记录没有显示技术异常。

本次可确认的训练不足是：内部总目标平均仅下降约 0.7% 至 1.1%；内部准确率平均仅提高约 0.5 至 0.8 个百分点，且存在下降阶段；多数阶段用完冻结的 4 次迭代，末次已记录梯度仍非零；72 个阶段耗尽试探预算，C_seq 的 29 个最终状态达到 β 边界。这些是完整轨迹的观测，不能据此确认增加预算会改善外层结果，也不授权改动冻结方法。

source validation、逐样本预测变化、外层旧类/新类效果、泛化提升以及是否达到用户理想目标，本文均为 N/A。仅有 10 个可训练参数不证明训练省算力；本次实测包含准备、原型、数千次内层拟合/分解及伴随求解，缺少对应硬件和任务口径下的独立比较，不能宣布星载收益。本文未读取外层评分、未修改方法或配置、未拟合、未启动/停止/热修改实验。
