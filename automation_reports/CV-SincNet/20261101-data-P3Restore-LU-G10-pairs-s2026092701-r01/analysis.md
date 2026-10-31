VERIFIED / ANALYZED。原 L_s6300/U_s56700 的完整 G10 MIX01 P0-clean 配对及独立 F0 clean fused160 缓存完成，执行 53410ce889adc94c53c1783741f970fbdaf51453。0 优化、0 TX 标签、0 OE/V/目标输入，未训练新版 G；OE 保持仅冻结诊断。

配对制备（含真实 2 条 L_s smoke）7.241917 秒，63000 条 F0 clean 缓存 2.309879 秒，完整 worker 10.528927 秒。F0 为实际原 Tri-SIG-TIME SSL / 3ff8 / E50-11100 / 232993 参数。缓存 IQ 两阵列 258048000 字节，F0 特征 40320000 字节；纯 IQ 优化 consumer 不打开指导阵列。共享缓存成本及 F0 原训练知识分别披露，不能称整个研究没有 F0 访问。

独立 0NN/0truth 核对：63000 条 ID/角色/RX/day 顺序一致、全部 IQ/特征有限；clean 复能量与 1 最大绝对差 2.07295787202e-08。与 r02 成本诊断重合的 512 条样本，其 clean、received、ID、角色、RX、day 六阵列逐位一致。原 raw ManySig 按 whole PKL 读取，包含其他角色字节与 TX 坐标命名，但仅索引原 L_s/U_s 包值；私有定位坐标不输入 G。63000 次原 raw RMS、2 次 canonical G10 审计、0 次信道后 RMS，首两条 P0 逐位复现。

真实源 smoke PASS，G 恒等初值、实际 F0 输入 VJP、0.5 梯度上限及模型状态保持。GPU 7 以原 1 个 compute 进程、22883 MiB 余量准入，新增本任务 1 个；worker 内 NVML 记录了自身 PID。首次独立读回时任务已自然退出，完整进度、smoke、manifest、结果及 controller 终态相互吻合；未为补遥测重跑。24 项聚焦检查及唯一独立 P0/P1 审查 PASS。

这只证明源训练数据已备好。尚无新 TF 的实际学习或正式性能结果，不能据此判断能恢复 clean 的几成、六景 3pp 或 Phase3 开放指标达标。下一步为独立纯 IQ / F0 指导、同初值和源顺序的从零单阶段 30 轮匹配组；自动冻结测试与独立评分已在实现核验中。默认 DURA 和完整目标保持。
