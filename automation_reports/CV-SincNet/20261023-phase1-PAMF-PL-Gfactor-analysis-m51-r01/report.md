# PAMF普通PL×G数据权限：17契约三seed完整2×2

发布状态 **VERIFIED**；运行阶段 **QUEUED**，新增性能仍待完成。

51个既有普通PL F、17契约、三seed，新增51predict＋51独立score＋1 CPU2×2汇总，共103阶段；F/G训练及缓存新增均0。原raw和Sup/G-L/G-LU、PL/G-LU读出复用；共享既有G-L恢复IQ。PL F用过U，不称严格监督。

实际执行提交 `108d938f404b091099fc41972d9768365b38933c`，controller PID1558142。真实PL F与同契约G-L在3条L源IQ上无query smoke PASS，FP32缓存复制后的默认头logits逐元素一致。接管25健康worker，23条相同PID日志跨约155秒增长；旧job的配置、worker release与输出逐项完全保留，当前24 GPUworker实际归属已核实。其他任务继续。

r01控制器因未打包planner依赖在接管前退出，失败release与日志保留，新r02只修复运行依赖隔离，不更改模型/数据/科学矩阵。原external owner正常handoff退休，不是调度器故障；不要重启旧owner或失败r01。统一时延的旧riei失败仍由已完成r02替代，不重新报警或计时。

全部新row任务和七景决定冻结后独立truth-last；CPU聚合在51新评分及全部旧四格输入齐备后执行，报告每景/RX/TX/day、Accuracy/Macro-F1、净纠错、PL−Sup、LU−L和交互，保留负值。缓存分类耗时不是在线G+F时延，G祖先及共享缓存成本单列。

[固定方案](../../../docs/PAMF_PL_GFACTOR_READOUTS_20261023.md)，实际逐行源和预算见experiment.json，发布证据见launch_verification.json。IEEE设置已更新并编译目检13页；r06作为此前三核心完成时的不可覆盖快照保留，下次结果交付用新包。原整体ETA仍锚定香港10月23日01:20的18至30小时初估窗口，不按本次发布或小时检查重置。
