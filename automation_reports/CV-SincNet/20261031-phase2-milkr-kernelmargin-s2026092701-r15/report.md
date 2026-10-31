# 相同支持核的最大间隔判决对照

预登记R16。完整Tri-SIG-TIME SSL、DURA v1.0、神经默认头冻结；源CCA16原参数复用。当前support仍用R14 nested选择特征权重/核/tau/ridge配置，但最终分类明确替换为precomputed C-SVC，属于目标函数变体，不声称原MILKR ridge未改。执行命令、环境/CWD、模型/数据来源、owner、GPU、不可变路径、stoprule见experiment.json。

C-SVC用当前K标签及相同核Gram训练，OVO优化/OVR decision_function输出全部注册类分数，最终沿原lexicaltie argmax。不使用query真old/new角色、配额、联合分配或query统计/拟合。全局C候选0.1/1/10/100，只从源选一个；target每head仅拟合该C。tol1e-4、max_iter100000，fit_status非0直接技术失败，类别顺序必须0..C−1。源ridge控制精确R14。sklearn已独立核实：本地1.7.2、N6071.7.0，不跨版本pickle，远端源真实执行先于target。

源L6300物理/既有ep91/92模拟12600配对；五heldRX×两episode×K1/5/10/20×old_count2/3/4/5×四路。shadowCCA16拟合排除heldRX，支持前K≤20、query固定后32/class，物理互斥。new/old源比例0.2至2，目标N20/6=3.33及26类复杂度为外推；F已见源六类，不能作为真novel确认。C-SVC四C加ridge控制共3200个源H，F/G更新0、无U/V真值。

源门槛：四路平均H≥ridge+0.5pp、两S Q0各≥ridge−0.5pp、各K四路均值≥ridge−0.5pp、两S Q1>Q0。通过者取源H最高，平局小C；无通过则source-only停止target0。通过才首RX1-1/P01/MIX01/seed2026092701的全16KN四路80行，全补集query；匹配原版/R14各80行复用。全部预测固定后独立truth-last。目标仍两S Q0较原版≥5pp、Q1>Q0和Q0旧新各≥原版−1pp；仅先导通过才相同参数其余六RX确认，完整逐格保留，不降低5pp。

本候选唯一独立P0/P1审查PASS；合成held类识别、全类输出、query批次独立/系数不变、selection JSON、ridge控制精确测试PASS。真实checkpoint无query smoke为launcher第一步，通过即继续。源GPU4、targetGPU0一worker，实际显存余量检查；其他owner不动，技术失败保留产物不盲重试。

## 用户验收澄清（源阶段运行中）

用户明确选择“每个K/N格都至少+5个百分点”。均值通过不能判定goal完成或继续扩展；两种support每个K/N分别要求Q0较匹配原版≥5pp、Q1>Q0，四路相对原版的逐格+5也完整检验。源候选/C网格/矩阵/种子/算法/预算均不变，健康源进程不热改；此项仅更新科学验收与晋级。原已发布快照保留，不抹去当时规则。后续不得以先导或七RX均值掩盖单格不足。

## 源端实际结果

**VERIFIED**：真实checkpoint无query smoke PASS，五RX/两epoch/四K/四oldcount/四路/五classifier控制共3200源H完成。source CCA参数原样复制，目标support/query/预测/评分0，所属进程全部退出。sklearn实测1.7.0，未跨版本pickle。执行commit `da91d2bbfa24d48d8e18a536e91c7a609321551b`；release传输、compile与PID/CWD/argv/GPU/log增长均已核实。

| C | 源四路H | 较ridge差 | S0/S1 Q0差 | 各K最小差 | 通过 |
|---:|---:|---:|---|---:|---|
| 0.1 | 79.395% | +0.017pp | -1.094, +0.471pp | -0.666pp | False |
| 1.0 | 79.107% | -0.271pp | -1.094, -0.422pp | -0.976pp | False |
| 10.0 | 79.148% | -0.229pp | -0.694, -0.486pp | -1.160pp | False |
| 100.0 | 79.148% | -0.229pp | -0.694, -0.486pp | -1.160pp | False |

没有C通过预登记源门槛，正常source-only科学停止，不降低阈值访问目标；它不能作为ridge优化成功证据。用户最新要求每K/N格≥5pp，原R14仅6/16 raw格满足，goal仍ACTIVE。
