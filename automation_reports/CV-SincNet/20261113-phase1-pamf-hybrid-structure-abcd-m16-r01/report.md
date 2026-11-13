# Hybrid 主路径与融合辅助 A/B/C/D 对照

- run_id：`20261113-phase1-pamf-hybrid-structure-abcd-m16-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Main 的六景落后 seed 2026092701 和 10 类全部三个落后种子，每种子固定四组结构：A＝ordinary/raw 主路由、无额外辅助；B＝置信样本物理主路由、无额外辅助；C＝ordinary/raw 主路由、独立物理视图加融合层辅助；D＝置信样本物理主路由、相同融合层辅助。既有测试已曝光且用于挑种子，结果为探索性。

## 数据、seed与模型来源

维持原完整 L/U/V，不划出源域开发集；各行使用同配置同 seed Tri E20，独立续训 E10＋E20，固定末轮 E50。A 是保留 Tri 派生 U style、joint、跨 RX 外推、跨日关系与梯度 guard 的零额外辅助底座，不是普通 PL；已有 PL 和原 Hybrid 按 seed 配对复用。

## 执行与存储

固定同一物理输入强度 0.25 随 E21–E50 退火；C/D 融合辅助权重固定 0.1，不开展参数网格。C/D 辅助视图独立于主路径，额外前向冻结 BN 状态并隔离随机状态；A/B 不计算额外辅助。所有条件逐批记录源 U 样本编号、置信度、伪类别、RX/day 和视图预测，每 50 批同步；记录非零扰动和完整无标签损失。C/D 每 50 步以相同参数、AdamW 动量和物理裁剪创建仅用于诊断的基线影子更新，比较实时完整更新；A/B 没有辅助，辅助造成的差量按定义为零，不生成影子更新文件。

同一资源 owner 接管本任务的健康队列而不重启其 worker；按 GPU 总进程数 6、GPU 7 为 4 及实时显存准入。每行训练结束自动预测 raw 与 DURA-LU 的 clean 和六种 practical，再独立 truth-last 评分；四条件预测均冻结后独立计算 B/C/D 相对 A 按 TX/RX/day 的错变对、对变错和净变化。系统性技术失败停止新派发，保护已有结果及其他 owner。

## 结果与覆盖

尚无结果。按 seed 和各场景分别报告 A/B/C/D、原 Hybrid 与普通 PL；Main 仅一个选择性 seed，不可称作三种子均值。样本置信度与一致性不是无标签伪标签正确率。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
