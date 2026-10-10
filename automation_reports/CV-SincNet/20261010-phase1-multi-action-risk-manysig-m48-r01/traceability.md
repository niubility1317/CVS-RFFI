# 下一版多解耦：真实动作提案与判别分布风险

来源：用户附件70624102-bbe2-49c5-b74c-3e38240426a7/已粘贴的文本.txt。用户2026-10-10明确本轮底座为纯CE、关闭半监督和星地增强；报告第七节原生配方据此覆盖。三model seed 2026092701/02/03，scratch E200，保持现有数据角色。健康旧实验不修改。

新包：experiments/cvs_multi_action_risk。所有要求先登记后实现；不能以存在类/参数代替实际路径证据。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D01 | II.1 | 逐指标观测次数、缺失null、稀疏原始快照 | runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不改历史训练 |
| D02 | II.1 VII.1 | E和G/C加权L/T/R/总辅助梯度及实际优化器更新 | runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 权重含可靠度 |
| D03 | II.2 | R available/fit/identity/audit/unique IDs分开 | receiver.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不重复解释候选数 |
| D04 | II.3 | 恢复既有源审计全量结构化产物/曲线/准入 | historical.py/report | verified | historical_recovery.json:316 files VERIFIED | 不重训历史 |
| D05 | IV.1-2 | 物理语义、CFO源交叉加性诊断与交互残差 | physics.py/diagnostics.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不宣称纯硬件分解 |
| D06 | IV.3 V.3 | T物理单位、周期相位、CFO及随机过程分头 | physics.py/actions.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 曲率单列人工压力 |
| D07 | IV.4 | 变动质量代理/不确定性，替代归一化功率 | physics.py/receiver.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不称真实SNR |
| D08 | V.1 | 状态条件一阶+对称二阶、exact29、分离优化 | actions.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 推理仅身份路径 |
| D09 | V.2 | 顺序轮换、条件边替换任务、真实中间及实际链式误差 | actions.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 公共顺序日程 |
| D10 | V.4 | 小候选困难提案、范围、随机保留、真实核验/降准入 | actions.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | p停止梯度，E走真实IQ |
| D11 | VI.2 | R五维margin均值/收缩协方差/下尾及估计误差 | receiver.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 小袋按唯一物理ID |
| D12 | VI.2 | R三袋分离、otherTX、day/condition、版本年龄、TX留出与directed mean | receiver.py/diagnostics.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 保留已落实约束 |
| D13 | VI.3 | R逐包CE+CVaR尾部，只更新G/C，小作用/平滑/可靠性 | receiver.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 无raw-z身份硬对齐 |
| D14 | VII | clean纯CE底座，无teacher/U/satellite无用计算 | design.py/source.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 用户明确覆盖旧配方；作用参考为独立冻结快照 |
| D15 | VII.1 | 固定分支系数的贡献消融与明确替代的等预算控制 | design.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 移除不重分配 |
| D16 | VII.2 IX.1 | 随机CE/一致性、精确G-only、L/LT/LR/LTR、R均值/分布 | design.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 每配置三seed |
| D17 | IX.1 | shared同作用形式/输入/损失/曝光，参数量报告 | actions.py/design.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不复用旧通用残差unified |
| D18 | VII.3 | L-only与L+U同无标签损失入口、禁止U标签/RX/day | source.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 仅独立作用拟合消融 |
| D19 | VIII | 独立源V压力/组合/RX-day-TX/macro/尾部/已施加标记 | validation.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不更新任何状态 |
| D20 | VIII | 冻结源规则clean保留+压力+困难组，噪声内简化 | validation.py/dispatch.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 固定对比矩阵全测；源排名不决定目标访问 |
| D21 | IX.2 | 先源机制诊断：真/解析/学习参数、条件链/TX留出/R尾/提案真实风险/噪声残差 | diagnostics.py/dispatch.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不以有限预算判容量上限 |
| D22 | IX.3 | 第四网络需源证据触发 | report | deferred | 明确按设计条件不触发 | 不新增交互/噪声网络 |
| D23 | IX.4 | exact29/角点缓存复用与分项资源耗时 | actions.py/runtime.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 不宣称隔离速度收益 |
| D24 | 发布 | 独占输出、三seed、scratch合法性、原生完整日志 | design.py/source.py/register.py | verified | CPU/CUDA focused and native integration checks; report/evidence | E200预算不缩减 |
| D25 | 发布 | 每row冻结→clean+6practical→独立评分→汇总登记 | evaluate.py/dispatch.py | verified | CPU/CUDA focused and native integration checks; report/evidence | 评分不回流 |
| D26 | 发布 | 本地负测/原生smoke、独立P0P1、Git/远端读回 | checks.py/publish.py/report | implemented | 本地与独立审查通过；待远端发布读回 | 唯一launch owner/root |

实现细节、量化系数和矩阵将在首次远端训练前冻结。诊断不创建额外审批；技术错误保留产物，不因低性能停止或选择性重跑。

## 验证与解释边界

D01–D21、D23–D25共24项已验证实现可达性；D22按设计条件延后1项（第四网络未触发）；D26远端启动待读回。正式源诊断/身份训练/目标结果仍待运行，不能把本地验证解释为效果已证实。

独立审查发现并修复R单包协方差令准入恒零问题；21项CPU/CUDA回归包含16次单包到达后oracle非零准入。源审计316份完整结构化文件已恢复，历史梯度归一化错误明确保留解释限制。

预算说明：每辅助调用最多32个新源L包；R另外最多重编码32个缓存审计IQ，前向/缓存/物理ID均单独计数。U仅在独立作用拟合消融使用，其TX未知，不声称U维持TX0留出。
