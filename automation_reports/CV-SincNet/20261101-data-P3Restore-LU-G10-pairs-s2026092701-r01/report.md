# Phase3恢复器原L_s/U_s全量G10-clean配对缓存

- run_id：`20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01`
- group_id：`phase3-independent-restoration-source-pairs`；类别：`diagnostic`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅制备63000条原源配对和可选实际F0 clean特征；0优化，不训练G，不打开OE/V/目标。

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

## 本次范围

独立全量源配对制备，0优化；OE只作将来冻结诊断。原6300Ls+56700Us，不读y/V/OE/目标。新架构先前实际F0成本已通过，本run不重跑成本、不训练新版G，因此无正式识别预测。完成后独立核对63000ID/角色、一次RMS和与既有512源配对一致性。

24项聚焦检查PASS；唯一独立P0/P1审查PASS。实际代码 53410ce889adc94c53c1783741f970fbdaf51453。原始IQ/特征/私有配置不提交Git。正式新G训练须接好自动冻结测试后另登记30轮匹配矩阵。

## 完成证据

VERIFIED / ANALYZED。原 L_s6300/U_s56700 的完整 G10 MIX01 P0-clean 配对及独立 F0 clean fused160 缓存完成，执行 53410ce889adc94c53c1783741f970fbdaf51453。0 优化、0 TX 标签、0 OE/V/目标输入，未训练新版 G；OE 保持仅冻结诊断。

配对制备（含真实 2 条 L_s smoke）7.241917 秒，63000 条 F0 clean 缓存 2.309879 秒，完整 worker 10.528927 秒。F0 为实际原 Tri-SIG-TIME SSL / 3ff8 / E50-11100 / 232993 参数。缓存 IQ 两阵列 258048000 字节，F0 特征 40320000 字节；纯 IQ 优化 consumer 不打开指导阵列。共享缓存成本及 F0 原训练知识分别披露，不能称整个研究没有 F0 访问。

独立 0NN/0truth 核对：63000 条 ID/角色/RX/day 顺序一致、全部 IQ/特征有限；clean 复能量与 1 最大绝对差 2.07295787202e-08。与 r02 成本诊断重合的 512 条样本，其 clean、received、ID、角色、RX、day 六阵列逐位一致。原 raw ManySig 按 whole PKL 读取，包含其他角色字节与 TX 坐标命名，但仅索引原 L_s/U_s 包值；私有定位坐标不输入 G。63000 次原 raw RMS、2 次 canonical G10 审计、0 次信道后 RMS，首两条 P0 逐位复现。

真实源 smoke PASS，G 恒等初值、实际 F0 输入 VJP、0.5 梯度上限及模型状态保持。GPU 7 以原 1 个 compute 进程、22883 MiB 余量准入，新增本任务 1 个；worker 内 NVML 记录了自身 PID。首次独立读回时任务已自然退出，完整进度、smoke、manifest、结果及 controller 终态相互吻合；未为补遥测重跑。24 项聚焦检查及唯一独立 P0/P1 审查 PASS。

这只证明源训练数据已备好。尚无新 TF 的实际学习或正式性能结果，不能据此判断能恢复 clean 的几成、六景 3pp 或 Phase3 开放指标达标。下一步为独立纯 IQ / F0 指导、同初值和源顺序的从零单阶段 30 轮匹配组；自动冻结测试与独立评分已在实现核验中。默认 DURA 和完整目标保持。
