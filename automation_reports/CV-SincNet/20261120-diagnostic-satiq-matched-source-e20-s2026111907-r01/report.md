# 同报头 SatIQ 与 PAMF 第 20 轮全波束源验证

[配置](experiment.json) · [事件](events.jsonl) · [正式对照](../20261120-phase1-satiq-ccs2023-header-matched-sup200-s2026092701-r01/report.md) · [第 10 轮源验证](../20261120-diagnostic-satiq-matched-source-e10-s2026111907-r01/report.md)

两种方法均到固定第 20 轮（564,500 步）后，评分进程复用 46,785 个训练注册包以及 49,814 个已知源 V 包。每条信号面对全部 3,103 类。两组比较共同余弦原型，另保留 PAMF 原分类头。第 20 轮末步经固定计划保存，评分进程不会选权重，不访问独立 T。

调度器复用经验证的第 5 轮代码，增加固定轮次参数。本地验证第 5 轮默认调用兼容且第 20 轮检查点路径正确，缺少检查点且上游退出时失败，使用已解析的源数据路径，只有两组固定轮次结果均齐备才发布评分文件。

## 调度状态

**VERIFIED 已排队。** 2026-11-20T11:22:34+08:00 独立读回 N607 调度进程 PID 2497388、命令、工作目录与等待日志。PAMF 的第 20 轮检查点已存在，SatIQ 尚未到同轮。调度进程等待两份检查点齐备后才申请 GPU。未写评分，独立 T 保持留出。

- 调度日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261120-diagnostic-satiq-matched-source-e20-s2026111907-r01/dispatch.log`
- 结果 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-matched-source-e20-s2026111907-r01/source_e20_closed_set.json`
