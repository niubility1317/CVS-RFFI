# NEXT-R5 FA-RDCE3→qKNN Target125实现追溯

状态：`LOCAL_VERIFIED / P0=0 / P1=0 / NOT_LANDED / NOT_LAUNCHED`

|ID|需求|目标文件/证据|状态|验证|
|---|---|---|---|---|
|TR-01|完整5receiver×5seed×5slice矩阵|`stage2_next_r5_fa_target125_matrix.py`|completed|125/375/1500/1350/150计数测试通过|
|TR-02|K5/K10使用同一闭式FA|`stage2_next_r5_fa_target125_core.py`|completed|K5/K10同公式、无K特异参数测试通过|
|TR-03|K1完整旁路FA|core/runtime|completed|fit调用0、logit/state/prediction/resource exact alias测试通过|
|TR-04|FA只由REG0 old support拟合并在REG1同对象复用|core/runtime|completed|REG0 fit与REG1 same-object reuse测试通过|
|TR-05|R1 signed unit后无ReLU/二次归一化|core/runtime|completed|direct qKNN与无二次归一化测试通过|
|TR-06|Target专用6-old-class Phase1聚合资产|asset builder|completed|D106 strict tap直连、98×6聚合、零Target输入测试通过|
|TR-07|复用sealed Target125 received-IQ与checkpoint|runtime/adapter|implemented; remote smoke pending|真实checkpoint no-truth smoke为N607首个动作|
|TR-08|support/query物理ID、capsule/split/row/scene/K绑定|runtime|completed|顺序、交集、row与asset/checkpoint/method-lock负测通过|
|TR-09|query零fit/update/selection/truth/role/quota/global reassignment|全部入口|completed|签名、ledger和forbidden-field负测通过|
|TR-10|prediction封存后独立truth-side score|adapter/CLI|completed locally|1500 coverage、DA同query、truth query-ID顺序和REG0 old子序列闭合|
|TR-11|四状态中文主名称与REG0 new/H=`N/A`|matrix/artifact/scorer|completed|schema、指标可用性与四状态差分测试通过|
|TR-12|最小发布门|报告、Git、review、preflight|in progress|独立审查`P0=0/P1=0`；待Git提交、N607预检和runner handoff|

本地验证：六个Python入口`py_compile`通过；四份聚焦测试共`14 passed`；`git diff --check`通过。独立Terra/max审查结论为`P0=0，P1=0`，准予进入Git冻结和N607发布，但不构成性能结论。
