# SF-TAPFT t3.norm＋D92组合实验r2报告

## 预登记

- run ID：`stage2_sf_t3_d92_combo_n2n10n20_rx20_1_s713101_m392002_20260828_r2_techfix1`
- 状态：`RUNNING`
- Git基线：在r1提交`89ca987260aea1b38224fc4fb70cc710437b5eb4`上修复真实checkpoint的identity backbone参数路径；本报告随修复提交冻结。
- 候选：`D0_T3_D92`、`S02_T3_D92`、`R3_DUALDELTA_T3_D92_INLOOP`。
- 矩阵：3候选×3场景×`N_new={2,10,20}`，共27格；每格输出`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。
- checkpoint：`ADV3B02_CORE90_SOFT_E200`。
- 数据：复用匹配`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`的既有Phase2包；旧类6类、K10；最大合法query分别为160、320、520条。
- D92：`D92-E0-NORF32`，RF32关闭，分类公式不变。
- 输入：固定checkpoint、old support、registered support、无truth/role query和data handle。
- 输出：不可覆盖run root下的t3-only delta、selection、四状态prediction、GNU time和GPU采样。
- GPU：N607资源检查后每卡最多2个训练任务；三个候选由三个子agent分别实现，主Agent统一冻结和发布。
- 停止规则：只对协议越权、错误输入/checkout、输出覆盖、确定性执行故障或prediction无法闭合进行定点技术停止；不得因低性能停止。
- 评分：全部27格prediction闭合后，独立scorer才连接truth。

## r1故障与定点修复

- r1真实checkpoint smoke证明模型参数为`id_backbone.t3.norm.weight/bias`；旧导出器只接受短路径，因而在产生artifact前技术停止。
- r2动态解析并严格允许两种完整二元组：真实identity路径或兼容测试短路径；拒绝domain backbone、混合前缀、缺项和额外参数。
- delta bundle保留真实模型路径，loader按同一路径应用；持久状态仍严格只有2个`t3.norm`张量，target head不持久化。
- 定点测试及三个组合聚焦测试19项通过；下一步为新提交真实checkpoint无query smoke。

## 发布与启动证据

- 冻结提交：`9f887cb0631f66682a4f5d8f2a1c788e77a7ff85`；GitHub远端分支OID独立读回一致。
- release：`stage2_sf_t3_d92_combo_9f887cb0.zip`；本地/N607 SHA256均为`df2464d8fa5e626be14984e5ffa5e2acac77c2eeb2586d1372beec0eabac672c`；远端编译和27格计划读回通过。
- 真实checkpoint smoke：`D0_T3_D92/leo_clear_weak`已完成；520步，墙钟13.71秒，最大RSS 1675588KB；1152个可训练元素全部实际变化；bundle 2520字节；真实持久参数为`id_backbone.t3.norm.weight/bias`；`query_rows_used=0`，未打开query truth/role。
- 当前适配：D0三个场景已完成；S02三个场景与R3九个场景/新类数组合共12个任务正在运行。GPU0–GPU2各2个任务，GPU3–GPU7分别1、1、2、1、1个任务，未超过每卡2个训练任务。
- 启动后读回：12个Python cmdline均绑定本run output root；8张GPU均出现对应进程，单进程初始显存约646–656MiB；异常指纹为空。
- 首轮并发SSH启动中13个请求已落地，但本地通道未立即退出；已定点关闭本地SSH且无残留TCP22连接。第14个请求明确未落地后仅补发该行，未重复启动其余任务。

## 待完成

等待15个适配artifact完整闭合，再发布27格prediction；全部prediction闭合后进行truth-last评分和最终分析。
