# 原生DR＋完整EG实施与验收

状态：LOCAL_VERIFIED / NO_EXPERIMENT_LAUNCH。用户授权本地实现、配置准备、全面验收；不启动实验。
基线：358a087a。工作树：E:/type10-7/code/snapshots/native_dr_eg_prepare_20260914_wt。
设计源：用户附件pasted-text.txt；按本对话13项实施计划执行。科学权限以E:/type10-7/项目.md为准。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| T01 | 2.1/4.4 | 配置三开关、实际生效、多seed | joint_config.py/runtime.py | verified | strict配置负测、19文件解析读回、默认入口不调用train | adaptive按T12后置 |
| T02 | 2.2/4.1 | 原生DR和明确的DR-off普通U目标 | dr_objective.py/objective.py | verified | E1/21/80实际模型双场、独立P1复审 | 额外扩展关闭；U底层domain/self不变 |
| T03 | 3.2/7.1 | 完整EG事务、一次提交 | solvers.py/runtime.py | verified | AdamW独立参考、异常回滚、真实模型两场 | 仅合成输入，无正式训练 |
| T04 | 2.3 | 实际除数记录/复用，原模式对照 | joint_normalization.py/dr_objective.py | verified | L/U实际除数、梯度和一次状态提交 | 保持L/U递推 |
| T05 | 2.8/4.3 | 接受步时钟、随机流、阶段 | tickets.py/runtime.py | verified | 14阶段边界、同seed初始张量、跨seed同输入、LR回归 | 固定课程下origin/live严格相等 |
| T06 | 2.4/2.5 | A/B探针及门控观测 | joint_diagnostics.py | verified | 真实模型合成特征probe A/B、状态/RNG恢复 | B是当前L批次跨TX诊断，非独立采集保证 |
| T07 | 2.6/6.3 | 路由漏斗、有效监督量 | muse_ssdg.py/dr_objective.py | verified | 实际route trace、受控H/P非零梯度、校准crossfit记录 | N关闭，不为实际低置信样本填配额 |
| T08 | 2.7/6 | 六层指标 | joint_diagnostics.py/joint_metrics.py | verified | 分项梯度、表征、分组/F1/配对数学检查、评分入口编译 | 实际完整数据评分与性能留待运行 |
| T09 | 4.1/7.2 | R0四格三seed配置 | configs/native_dr_eg | verified | 12份R0实例解析、无启动副作用 | launch=false |
| T10 | 7.3 | adv=0两组配对 | configs/native_dr_eg | verified | 6份配置、L/U adv CE关闭定义、U GRL梯度测 | 保留U z_dom/self |
| T11 | 4.2/4.3 | R1逐项消融配置及机制 | joint_config.py/dr_objective.py | verified | 9承接模板、R1实际logit .1/κ .5/GRL .5配置、ramp测试 | 后续承接不擅自选择source赢家 |
| T12 | 5/7.4 | 梯度预算和动态EG | 研究规格 | deferred | 依赖固定EG结果 | 按计划后续研究，不混入R0/R1 |
| T13 | 7/8 | 验收与版本交付 | 本报告和acceptance artifacts | verified | 53项PASS、配置/编码/编译读回；Git交付另见delivery.json | 不启动、不宣称性能成立 |

## 交接
已完成：实现、18配置、R1模板、53项本地验收。12项verified、1项deferred、0项rejected、0项blocked。verified仅指本次获授权的代码/准备/本地验收范围，不含正式实验性能。

详细证据：experiments/adv3b02_xuc/acceptance/native_dr_eg/README.md、acceptance.json、tests.xml、artifact_readback.json。代码路径表中均相对experiments/adv3b02_xuc/code/cvsrffi/xuc_fusion，solver位于game_tracking、muse_ssdg位于cvsrffi根。

最高风险剩余项：真实ManySig/source物理角色、N607环境和44,400步闭合尚未执行验证。不要自动启动，等待用户后续指示。恢复任务应先读本报告及交付commit，不重建或重复发起实验。
