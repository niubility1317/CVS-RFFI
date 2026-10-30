# PAMF 同记录视图学习后的冻结身份结构诊断

physical-open 两组均已完成并未晋级。主 P_LOE 源未知 R_U16.4127%、目标 R_U20.1389%；目标已知接收95.8333%但A_K69.2708%，原F0闭集到开放损失3.6458个百分点。训练目标实际产生梯度，不等于未知设备结构得到改善。本轮先区分表征结构和成员评分，不加训练预算、调阈值或更换主模型。

## 固定对象、数据与复用

冻结 `20261030-phase3-star-physical-open-s2026092701-r01` 的 P_L、P_LOE，实际代码31fa172946aeaecd2d3f1e784aee7366b7b5bee1、各2220新增/累计13320更新；原F0 E50/11100祖先、六类默认头和常量C0结构保持。每个模型导出同次完整unit fused160与unit open64，不重训、不另训头、不更新原型／阈值／buffers。

F0 baseline精确复用已完成 `20261030-phase3-star-open-feature-diagnostic-s2026092701-r01/features/F0.npz` 的 original F0 unit fused160。其旧export代码55977eec2cf0c2bd09d7c6856179f919e9f0923d，parent_F0身份和同source novel manifest须与本轮两模型一致，IDs/RX/day顺序逐值相等。不重做F0前向，不把旧score或truth用于基线生成。

输入只有既有 source novel3150 条G10P1：19-15、19-16、7-1各1050，source五RX（1-19、18-2、19-2、2-19、3-19）×三日（2021-03-08/15/23）×70/设备/域。TX只独立scorer可见；导出只见opaqueID/RX/day及同received经冻结DURA的缓存P1。源接收机已见、三未知设备已有历史开发暴露，不是新RX/新TX盲测。

source L/V/OE/Us、targetU/Q、正式RX24-18均新增0读；G加载/前向0，所有postPhase1新IQ仍G10星地P1，不新读clean、P0值或重抽信道。新F前向2×3150=6300真实行、两个真实checkpoint各2条合成G10 P0 noquery smoke共4合成行；F0 baseline3150 features复用。独立物理记录只有3150，不能把两模型或五表征算成更多记录。

## 五表征、先选择后评分

固定五种：F0.fused160、P_L.fused160/open64、P_LOE.fused160/open64。全部特征完成后，独立CPU在每种表征分别固定all-other、cross-RX、cross-day、cross-RX-and-day四套1NN索引。所有候选排除自身，以FP64 cosine、blocked128及opaqueID字典序处理精确ties。真TX、真实K、角色配额不能进入近邻选择；全五种四套索引全部冻结后，才连接唯一source novel truth。

评分报告真实TX宏命中率及逐RX/day/TX、跨RX/day同类／异类平均cosine gap、有效秩／参与秩、TX/RX/day单因素组间方差比例。真零向量保留以诊断塌缩；无候选／缺必需类保持N/A。方差比例非正交分解、非可加、非因果；高秩或较高命中率也不自动证明可拒识。

这些是同池参考检索与描述性几何，不是无标签分类、发现真实类数、未来匿名注册A_N或全Phase3成功。主P_LOE身份不变，控制或baseline不能按结果替换主。没有新的参数／checkpoint／阈值选择，也不据本轮source真值更新任何状态。

## 诊断后的研发方向

若开放表征仍保留明显跨RX身份结构、拒识却失败，下一机制优先检查评分如何利用结构；若结构明显退化或RX差异增加，先研究表示的域不变约束。二者是诊断解释，不是自动晋级或从已有控制挑最高分。新的学习机制须独立预登记；本轮不扩正式RX/多seed、不改既有目标或把检索率充作匿名注册准确率。

## 执行

唯一新run/output root，代码Git身份、聚焦协议负测、唯一直接P0/P1审查、最小预登记、一次N607 preflight、单release archive比较／compile、首步两个真实checkpoint无query smoke、独立PID/CWD/argv/GPU/log增长、全部特征和近邻固定后独立truth-last。仅一个GPU导出worker（actualfree≥4096MB、每卡≤2新增own），随后CUDAempty/BLAS1 CPU scorer；foreign不整卡排除、不停止/修改/重启。

非法祖先／角色、非有限、输出碰撞或无法产生合法特征时保留，不自动重试；UNKNOWN先核原PID/root。诊断负结果不触发停止他人或更改数据。白名单外设计审批／报告审查／签名链NONBLOCKING。精确命令、环境、路径、矩阵与产物在run登记。完整跨RX星地拒识、零目标标签适应、发现和未来匿名注册目标保持ACTIVE。
