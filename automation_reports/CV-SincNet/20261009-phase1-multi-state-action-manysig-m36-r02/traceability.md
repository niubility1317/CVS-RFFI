# 多解耦状态条件作用：设计逐项追溯

设计来源：用户附件 `39674558-b411-4b15-8434-f44981c3a6da/已粘贴的文本.txt`。本次授权要求逐项完成，不将第三阶段隐式延期。三个阶段顺序为源作用对照、联合条件作用、从零身份训练及冻结后独立测试。LT保留诊断是报告明确范围，非遗漏。

所有新文件属于 `experiments/cvs_multi_state_action/`。身份骨干维持旧门控＋cosine与原clean＋LEO拼接；作用审查复用四个既有合规multi E200，正式身份矩阵从零训练，禁止读取历史目标成绩作选择。

| ID | 报告章节 | 验收要求 | 目标文件 | 状态 | 验证证据 | 说明 |
|---|---|---|---|---|---|---|
| S01 | 1/11 | 四seed、源物理包独立拟合/审查，说明骨干曾见源包 | runner.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 同一物理包跨视图同角色 |
| S02 | 4 | 真实p、解析配对p、神经配对p三类对照 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | L稳定复数LS，T加权相位LS与条件数 |
| S03 | 5 | 零增量恒等、允许偶部、0/+p/-p诊断 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不将-p视为FIR精确逆 |
| S04 | 3/6 | D(h,p)与D(h,s,p)等预算对照 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 作用码和接收样本状态分开 |
| S05 | 6 | 一阶与一阶＋二阶作用对照 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 状态相关方向，非固定全局子空间 |
| S06 | 6 | 全349维对照精确29维＋学习320维 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 真实固定前端响应 |
| S07 | 5/9 | 损失能量下限、归一化z及所有竞争类margin | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 冻结G/C参数但保留输入梯度 |
| S08 | 6 | 模20圆周delay差及路由切换诊断 | action_checks.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 仅模板路由不称物理时延 |
| S09 | 8 | own/同TX异包/匹配crossTX/不匹配crossTX/shuffled增量 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | RX/day/质量条件明确 |
| S10 | 8 | own＋cross联合拟合、匹配样本曝光及多增量日程 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不用cross替代own |
| S11 | 8 | 辅助模型TX交叉留出 | runner.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 身份训练保留全部源TX |
| S12 | 5/11 | 学习曲线、梯度/稳定性、预算末未收敛解释 | actions.py/receiver.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 200步非能力上界 |
| S13 | 7 | R逐包作用后聚合G/各竞争类margin | receiver.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不用G(mean h)冒充mean G(h) |
| S14 | 7 | R跨TX共享或leave-TX-out描述码 | receiver.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不等同共享参数 |
| S15 | 7 | 独立描述/源包/目的统计分袋＋重复分袋 | receiver.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 物理ID不交 |
| S16 | 7 | RX有向对×条件其他TX均值强基线 | receiver.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 与zero/centroid对照 |
| S17 | 7 | day组成匹配、统计估计误差与稳定性 | receiver.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 无逐包跨RX反事实声明 |
| S18 | 7/9 | 真实目的RX固定类别锚点进入训练目标 | receiver.py/runtime.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不止诊断数值 |
| S19 | 11 | 单套L/T/R联合clean＋源LEO并报告分层 | runner.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 另有分别拟合对照 |
| S20 | 9 | learned/精确virtual/real梯度与完整native参照 | runtime.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 分E/G范数、余弦与参考语义 |
| S21 | 9 | virtual只更新G/C与同时更新E/G对照 | runtime.py/design.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 真实IQ正常更新E/G |
| S22 | 9 | 真实CE独立权重、源可靠性仅控制学习作用 | runtime.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不用聚合skill充当逐包可靠性 |
| S23 | 9 | 分支增多不增加总辅助预算 | runtime.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 记录额外视图/梯度/解析成本 |
| S24 | 9 | 在线参考漂移、版本隔离、贡献实际过期验证 | runtime.py/receiver.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 固定参考检查不能代替在线 |
| S25 | 11 | native、real_views、unified、L、LT、LTR六项身份对照 | design.py/source.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 全部固定E200从零、四seed |
| S26 | 11 | 原生伪标签U隐藏、旧门控cosine、拼接信道保持 | runtime.py/design.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 每4身份step最多32标注源样本 |
| S27 | 10 | LT四角/顺序/交换残差诊断、无扩容 | actions.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 不添加learned LT网络 |
| S28 | 5/11 | 参数、计算/时间/显存/状态/曝光成本 | runtime.py/runner.py | verified | 实际CVS CPU/GPU定向检查及源码审查；见run/evidence | 未测FLOPs写N/A |
| S29 | 11 | 预登记源判定规则、所有固定对照冻结后预测 | dispatch.py/evaluate.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 不读测试结果选模 |
| S30 | 11 | 全矩阵独立truth-last测试与同row分层指标 | evaluate.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | clean、三LEO弱场景及已有六场景 |
| S31 | 全文 | 新run独占输出、唯一owner、每GPU最多4进程 | design.py/dispatch.py | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 不修改健康旧任务 |
| S32 | 全文 | 逐项反向审查、完整报告/登记/Git推送核验 | register.py/本文件 | implemented | evidence/local checks及独立定向复查；真实源/身份结果待本run产物 | 仅实际证据可标verified |

当前状态：实现中。科学无收益不构成技术失败；源校准权重为零的分支必须如实报告，不宣称已有效约束身份。若直接正确性问题导致后续无法运行，保留所有已完成证据并明确阻塞项。

## 本地验证与解释边界

24项实现已通过对应本地验证；8项已实现并进入正式执行流程，等待真实数据或最终评分证据。deferred/rejected/blocked均为0。实现是报告核心公式与对照的直接落实；质量匹配使用明确标记的log-IQ功率代理（非校准SNR），R当前学生对冻结参考目的分布属于显式教师分布监督。固定200步不声称能力上界。

已执行action_checks CPU/GPU、receiver_checks CPU/GPU、runtime_checks CPU/GPU、control checks、实际native smoke九臂及R路径交叉fixture、compileall、登记validate --launch-ready。最高剩余风险：辅助作用改善与可靠性准入能否转化成E200识别收益；只能由本轮冻结后的独立测试回答。
