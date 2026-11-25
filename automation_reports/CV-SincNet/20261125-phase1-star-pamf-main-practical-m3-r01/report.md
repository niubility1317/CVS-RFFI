# STAR-RFFI 接入 PAMF Main split 主对比（practical 增强）

- run_id：`20261125-phase1-star-pamf-main-practical-m3-r01`
- group_id：`phase1-pamf-main-star-practical`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

匹配 Main split 的三个种子，训练 STAR-RFFI 并与冻结 PAMF 主行比较。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
## 方法和对照

本轮将 Smooth STAR E A 版（`bn_budget`）补入 PAMF 主表的 Main split。历史 A 版在原 RX1 纯 Loo 条件下，最后 E100 EMA 的 8 种子平均准确率为 67.840625%，样本标准差为 1.279225 个百分点。历史目标成绩参与了该配方的开发。其来源为 `smooth_e_bn_budget_dual_rx1_s16_t90_e100_20261008_r01/analysis_final_20261008/report.md`。

本轮从随机初始化训练六类原生随机分类头和骨干，保留 clean/SG 双 BN、源域 E5 冻结教师、置信度阈值 0.90、SG 伪标签权重 0.45、伪标签损失预算和最终 E100 EMA。BN 仅用源 L_s 分别重估。源 U_s 只读取 IQ、opaque ID、RX 和日期，源 V 只拟合温度。历史十类权重仅在隔离的 source-only smoke 中前向读取，本轮权重不继承。保留 RD 与正交目标在 clean U_s，practical 视图只计算 TX 交叉熵。标注 practical 辅助 CE 权重 0.68。

ManySig 每个物理记录提供一个中心 256 点等化 IQ。保留 STAR 的 2048 点时频入口，将实际 256 点和其 512 点 FFT 装入第一时频槽，第二时频槽置零。每条源 L_s/U_s 预生成一个配对 practical residual 视图，采用 PAMF 外部主对照的三场景概率 40% low suburban、35% mid urban、25% low urban。E1–20、E21–45、E46–100 的应用率分别为 0.30、0.60、0.80，未应用时辅助槽为同条 clean。逐轮记录实际应用计数与伪标签接受量。

## 测试与判定

三个模型种子固定为 2026092701、2026092702、2026092703。沿用已评分的 PAMF 主行和完全一致的 Main split 测试 IQ。各模型独立冻结后，依次完成 cross-date 30000 条、cross-RX 126000 条、cross-both 42000 条的 clean 和六种 practical 测试，同时读取匹配的 DURA-LU 恢复 IQ。全部逐行预测固定后，由独立 CPU 评分进程按 ID 连接真值。报告准确率、Macro-F1、分组结果、跨种子均值及标准差、与 PAMF 的配对准确率差。六种 practical 等权平均，clean 单列。

PAMF 参考为 `20261027-phase1-pamf-paper-completion-m119-r03` 中三个 `main_split-factorial-ssl-full-permuted` 评分行。发布前已独立核实 PAMF 的 9 组阶段预测 ID 与当前句柄逐项一致，以及 Main split 匹配 DURA-LU 缓存完成。

## 本地核验

已通过六类前向、E1/E6 目标的反向与优化器更新、置信度掩码和预测端 truth 字段拒绝。独立 P0/P1 审查发现训练输出父目录缺失，已改为同时创建父目录，并将历史模型 smoke 的 IQ 收窄到本轮合法 L_s。发布后读回远端控制器、正在训练的工作进程及实际日志。
