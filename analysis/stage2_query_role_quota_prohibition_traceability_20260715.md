# Stage2逐样本角色与类别配额禁令追踪

## 范围

2026-07-15用户确认：对任一待识别样本，系统事先不知道其属于旧类还是新类。Phase2/Phase3目标域query推理必须逐样本在全部已注册类别上自主决策；真实old/new角色、query批次类别数量、每类quota及由这些信息驱动的Hungarian分配均被禁止。

## 需求追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|RQO-01|用户指令|将禁令提升为Phase2/Phase3通用协议，不局限于极轻型模式|`E:\type10-7\项目.md`、`docs/source_controls/PROJECT_PROTOCOL.full.md`、`docs/PROJECT_PROTOCOL.md`|verified|根协议与Git摘要均明确禁止query真实角色、批类别数、quota、排序/分块与配额重排|地面Phase1伪标签class quota不在本禁令范围内|
|RQO-02|用户指令|正式runner只允许逐样本全注册类决策，并拒绝role-partition adapter与类别配额Oracle|`paper_reproduction/cvs_aligned/cvs_method_runner.py`、`paper_reproduction/scripts/run_cvs_publication_matrix.py`|verified|配置预检、runner validate与artifact验收三层拒绝；正式调用不再传入old query数量或每类query数量|未标注query-query transductive图本身不等于角色/quota Oracle，不在本次通用禁令内；极轻型模式仍按资源协议禁用dense query图|
|RQO-03|用户指令|历史Oracle配置与launcher必须明确不可启动、不可晋升、不可排名|两个legacy Oracle JSON、两个Oracle shell launcher、`analyze_qknnv42_strict_dual125.py`|verified|JSON为`launchable=false`；两个launcher在任何资源操作前`exit 2`；排序仅包含协议有效light行，Oracle另存无效附表|历史文件继续用于来源审计|
|RQO-04|报告声明边界|`87.74%`与`83.25%`等role/quota Oracle结果只保留为协议无效历史上界，不进入正式比较|报告生成器、完整历史轻量报告、MLP报告、完整对比报告|verified|正式重建后headline=3、core=3、非Oracle K-shot=10；Oracle附表=1+5且均为`protocol_valid=false`、`eligible_for_ranking=false`|不改写原始metrics|
|RQO-05|回归验证|新增测试证明正式runner拒绝角色Oracle、类别配额和role-partition adapter，同时合法非Oracle路径继续可用|`tests/test_cvs_proposed_stage2_runner.py`、`tests/test_cvs_publication_matrix.py`及相关定向测试|verified|`ssr-gpu`下py_compile通过；55项定向pytest全部通过；5份JSON解析通过|Windows本地bash执行因启动超时未作为动态证据；已验证两个脚本的首个业务动作均为阻断并`exit 2`|

## 遗漏风险

- 只在文档写禁令，但runner仍接受`legacy_role_quota_oracle`。
- 只把Oracle标为diagnostic，benchmark或launcher仍能生成新结果并被后续摘要误纳入排名。
- 把Phase1源域伪标签的class quota误删；本次禁令只针对部署阶段target query决策。
- 仅禁止角色分路，却保留利用全批类别数量、query排序或标签分块推断角色/配额的等价信息泄漏。

## 完成统计

- verified：5
- deferred：0
- rejected：0
- blocked：0

本次实现与用户明确禁令保持严格一致：正式路径禁止真实角色、批类别数、每类quota及等价配额重排；Phase1源域伪标签quota不受影响；既有Oracle只保留历史审计artifact。
