# 六方向源机制诊断：冻结 CE30 三主种子

- run_id：`20261013-source-six-direction-mechanisms-manysig-m3-r01`
- group_id：`source-six-direction-mechanisms`；类别：`diagnostic`；阶段：`Phase1 source diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

按新报告分别证伪状态匹配、多相采样、RX更新迁移、跨TX共享变换、有向二阶和集合伪标签；0身份更新。

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
# 六个报告方向的固定源机制探索

用户在 2026-10-13 明确确认：所有新身份模型仍从头训练 30 轮。本次先执行报告要求的六项源机制诊断，固定三主种子 CE30 最终身份模型，仅前向或函数式虚拟更新，持久身份更新为 0。普通 cosine 分类头、160 维身份表示、无显式 CFO 坐标保持；内部原有频偏补偿仍存在，不能称为完全无频偏处理。最终 84% clean 三种子均值尚未达到。

统一数据为已有合法 L_s 6300、隐藏 TX 的 U_s 56700、只作健康诊断的 V 27000。身份模型此前见过全部 L_s；新诊断的物理 fold 只独立于其辅助拟合，不能称为身份模型未见样本盲测。接收 IQ 的状态描述符可能携带 RX/信道差异；跨 RX 的不同数据包只构成分布端点，不能声称相同发射波形配对。没有 target index、IQ、truth、prediction 或 score 访问。

|方向|固定诊断及对照|进入独立训练的源条件|
|---|---|---|
|状态匹配部分对齐|每 TX/day/RX 对，各侧 n=10、K=5、rho=.5，两个物理 ID 不交的重复；随机、贪心近邻、全局容量受限部分 assignment 同样正对数、权重、参考及负例；q 前 8 维匹配，lag1/4 维独立于 cost 验证|held L 验证维代价比随机下降至少 10%；每 TX/day 至少两个状态 bin 在至少三台 RX 重复；困难样本覆盖至少随机期望的一半；所有质量和容量约束成立|
|保留采样位置的多相路径|真实源 IQ 非循环 ±1/±2 位移；按累计 stride 标明共同有效区域与不可精确对齐的相位括号。最后 t2.pool 比较原池化、逐点 GN 池化、固定低通、受约束两相位 carry。保留原末端 160/cos|可逆/carry 误差≤1e-5、任意相位拼接反例≥1e-3；held L 原池化局部误差比 carry 至少高 .05；真实单位位移预测变化≥1%或实际 scale30 margin 改变量中位数≥.25|
|跨 RX 更新迁移|t1/t2 小 DW 参数块、alpha=.001/.01；三个 day、四个重复、完整 5×5 RX 迁移与四个同 TX/day 随机分组。物理 update/test 分离；固定 primary 为 t1、alpha=.01|真实 off-diagonal 损害率比随机至少高 10 个百分点，mean G 及 excess>1e-5；至少 2/3 day、3/4 repeat 同向。其余三 case 全报，不用 any-case 晋级|
|跨 TX 共享接收条件变换|共同坐标 rank2、B 谱范数≤.15、RX1 锚；三组双 TX ×三 day 留出，两次独立 L fold 拟合各 400 步。比较零动作、共享平移、共享低秩、同范数随机共同坐标、等幅位移、每 TX 平移参考|两次拟合各至少 6/9 留出块优于零与随机共同坐标，整体 MSE 各下降≥2%、覆盖≥80%、非零扰动、condition≤1.36。identity/inverse/composition 成立本身不算通过|
|有向时延二阶|固定中层时间中心化、K8、lag1/2/4；有向、静态 C0、时间打乱对照均映射回 160；fit0/fit1 独立 readout，fold2 检验。fit01/V 仅背景。倒序、因子2、lag0、C0 和零出口梯度检查|两次独立 fit 各较静态和打乱净救回≥3 包、较 joint160≥1 包；至少 4/5 RX 与 2/3 day 净救回非负；数学检查通过|
|集合伪标签|clean+.005 固定噪声，各 .95 累积概率最小前缀取并集；大小≤3，同 teacher/参与 mask/强噪声 .015/全批分母的 hard、soft、set 监督内容。L 先藏标签生成集合，U 只统计大小与稳定性|L fold2 模糊小集合≥20 且≥1%、top1 错误≥10、小集合真值覆盖≥95%、模糊错例包含率≥50%、全集≤10%。V 不参与判定，U 无准确率|

每个方向要求三主种子的固定条件均通过才进入新的独立从头 30 轮训练；未通过的方向保留全部结果及失败解释，不改阈值或选择性重跑。源条件只说明适用性，最终身份收益仍需后续同预算对照与冻结后的独立测试证据。不会把六种机制叠加后将无法归因的涨分当作各方向成功。

唯一 launch owner 为 root。一次代码/config 提交与上游 OID 核实、聚焦负例/数学测试、一次独立 P0/P1、真实三 checkpoint 无 query 首步、一次 N607 preflight、单 release archive 的一次本地/远端 SHA 对比、remote compile、PID/CWD/argv/GPU/log 检查。一个 GPU 进程、CPU 两线程，计入其他 owner 后 GPU 总进程最多 3、空闲显存至少 12000 MB；所有健康旧任务保持。不可覆盖输出根、预登记技术失败停止并保留产物，无低指标停止或自动重跑。命令、Python、CWD、路径、矩阵和预期产物见 experiment.json。

本阶段明确 source-only，不启动 target 测试。通过条件后的正式身份实验会另行固定矩阵、预算和匹配计算对照，逐 row 自动冻结→prediction→独立 truth-last→同 row 分析。旧九行条件 RX 探针未启动，因用户明确的新六方向报告而暂缓，未停止任何健康进程。


本地状态 LOCAL_VERIFIED：六项聚焦数学/输入负例/合成源入口检查、真实随机 native 架构与 identity-only 小参数块接口，以及 3×6 完整关闭/失败保存/不可覆盖的流水线 fixture 均 PASS。独立一次代码 P0/P1 审查 PASS。合成结果只用于组件核验；真实三 checkpoint 无 query smoke 是发布首步，真实源适用性与身份性能尚无新结果。V 未参与拟合或晋级，U 无真值。
