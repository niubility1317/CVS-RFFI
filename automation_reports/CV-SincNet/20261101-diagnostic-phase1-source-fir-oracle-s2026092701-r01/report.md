# Phase1 源域复数短滤波配对拟合诊断

- run_id：`20261101-diagnostic-phase1-source-fir-oracle-s2026092701-r01`
- group_id：`phase1-source-fir-oracle-diagnostic`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源域配对 clean 和接收 IQ 上拟合逐窗复数及共轭滤波器，检查静态信道可解释程度。

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

## 源域结果

诊断已完成。对同一组 512 条源训练记录，分别在源视图 1 和 20 上拟合逐窗广义复数 FIR；两次观测共享物理记录。滤波器直接利用每条记录的配对 clean IQ，且包含中心窗口的未来采样，只能作为特权源域诊断，不能直接部署。均以原 DURA 为逐条对照。

| 源视图 | 方法 | 与 clean 教师一致 | 修正 / 损伤 | 128 条 L_s 真标签正确 | 平均 IQ NMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 原 DURA | 491/512 | 0 / 0 | 121/128 | 0.020213 |
| 1 | DURA 后接 9 tap FIR | 498/512 | 10 / 3 | 125/128 | 0.014903 |
| 1 | DURA 后接 33 tap FIR | 501/512 | 13 / 3 | 125/128 | 0.010266 |
| 20 | 原 DURA | 492/512 | 0 / 0 | 125/128 | 0.019440 |
| 20 | DURA 后接 9 tap FIR | 501/512 | 11 / 2 | 126/128 | 0.014399 |
| 20 | DURA 后接 33 tap FIR | 507/512 | 17 / 2 | 127/128 | 0.009888 |

在 BLOCKED 子集上，9 tap 对 clean 教师的一致数分别由 65/83 增至 73/83、由 48/62 增至 56/62。直接对接收信号拟合 FIR 的表现明显弱于 DURA：33 tap 的两个视图分别只有 283/512、295/512 与 clean 教师一致。

逐条计数、方法矩阵、输入角色、预测范围、冻结基座和 DURA 身份已独立核验；远端 `complete.json` 的 SHA-256 为 `35a0d334fce5b1b1d6dbd7744ae09a9e0050c235111bf2bc7712baa25f6b4737`。本地副本为 [complete.json](complete.json)。加载 6300 条合法 L_s 标签，实际统计其中 128 条；未读取 U_s 标签、V 或测试 IQ。

结果支持进一步研究从接收信号预测滤波系数，但不能证明这种预测可行。这里的 clean 教师一致性不是测试真实准确率；源样本属于训练集，IQ 拟合最优也不是分类准确率上界。
