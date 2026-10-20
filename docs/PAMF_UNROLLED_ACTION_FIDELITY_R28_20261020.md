# R28：三步 U-Net 的作用／噪声／末级先验源域诊断

持续目标保持：独立恢复器的单一 IQ 供原 PAMF 全部分支使用；恢复训练 TX 标签／F／target 信息为0，利用原 L6300/U56700；正式 low urban≥70%，其他五受损环境分别距原 F50/seed raw clean≤5个百分点。R27 补测全部结束，完整结果及负结果保留；本轮不使用其中的目标排名选择模型或改参数。

本轮检验 R17 原报告 A17-08 已明确留下的未证实项：三步展开中，替换估计作用与噪声代理是否改善源身份保留。R06/R14 仅覆盖 coarse／R10 单步 prior；R25 是冻结源特征统计；R22 解析约束及 R23 coarse 联训均不能回答此问题。固定历史 R17 `unroll_unshared`（原 source-selected 行），不改用 R27 中较高目标分数的 shared 行。只读同一原1800 source V、R07 已缓存的七视图和条件作用，零新训练／噪声抽样／持久状态拟合，0 target。

|行|作用|方差代理|输出|
|---|---|---|---|
|native_estimated|原估计|原估计|原三步控制|
|known_action_estvar|已知条件作用|原估计|三个 innovation 和 MAP 都替换作用|
|est_action_simvar|原估计|原模拟器代理|三个 logvar 输入和 MAP 都替换方差|
|known_action_simvar|已知条件作用|原模拟器代理|同时替换|
|prior_only|原估计|原估计|保持前两步原轨迹，返回末级 prior、跳过末级 MAP 输出|

四条完整轨迹均为三次原 Core9、每步2次 warm-start CG、原 SCALE²；控制与末级 prior 共享计算但不改变输出。已知作用含真实 AGC/lock/FIR/main/image，属于不可部署 oracle。模拟器方差仍是白输入等效实噪声代理，不是完整真实协方差。已训练 prior 与原估计作用共适应，替换作用也改变 innovation，替换方差改变 logvar 条件；因此该反事实不是全模型理论上界，也不把变化全部归因于单一物理参数。

真实2条L_s／固定全 checkpoint／原F50无query smoke 为入口第一步，检查新控制与原 forward 一致；原F用 no_grad，不能在 inference_mode 内构造依赖参数版本的缓存。之后一个N607 GPU worker 在原7场景保存全部5条IQ/features/logits，全部完整有限且ID匹配后独立CPU scorer才打开源V标签。默认头不拟合。源V只作诊断，无反向传播或持久状态更新。三郊区、三urban、clean、TX/RX/day、MacroF1、修复／损伤、波形／FFT和配对clean特征均保留。原F50有源域身份监督，分析不称整条链完全无监督。

沿用 R14 的预声明机制门槛：已知作用＋模拟器代理相对本轮原三步控制的六场景平均NMSE≤.95倍、low urban≥+1pp、mid urban≥+.5pp、clean相对原输入下降≤.1pp、低urban任一TX变化≥−1pp，才支持优先研究作用估计。其他替换及末级prior解释作用／噪声／最后一致性影响。任何结果都不晋级oracle、不自动访问target、不修改旧源资格或健康任务；目标70/5pp及逐细节身份真实性继续保留。

代码来源为原R17 commit `cd97744884b5e460926e48c8b7efb7d78c6da0f4`，严格加载原末轮E20/4940全状态；R07/R02完整原7/63/30祖先及0TX/F/target训练来源沿用已核实结论。原R10仅提供同ID原F的identity/coarse参考，不作为本轮prior。epoch/data/角色不改，不重复数据验证或建立hash链。

N607普通账户、新不可覆盖root `20261020-diagnostic-restoration-unrolled-action-F50-s2026092701-m5-r28`；GPU池0/1/5/7，最多1条本任务GPU worker，每卡全owner cap2、空闲≥12GB，共享capacity4锁保持到CUDA PID可见。非有限／错误来源／输出碰撞／无法合法预测时保留产物、停止派发、无自动重试；健康其他owner不动。精确commit、命令、环境/CWD、路径、seed和预期artifact见登记。

`REJECTED_EXTRA_GATE`：通用 brainstorming 的额外设计审批、独立spec审批不属于项目八项最小流程；按用户明确的持续研究授权和最小流程推进。该记录不放宽科学数据、checkpoint来源或独立评分权限。
