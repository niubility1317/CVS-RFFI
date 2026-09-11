# CORE90V2设计实施与机制开启再审查

本次请求：再次核对设计报告、实施计划及机制/参数实际开启情况。审查起点为`2a80889a274734483452c5a651345179f8d0c9be`，不启动新正式训练，不重跑已完成的长计时。

依据：[设计原文](design_refs/CORE90_GAME_V2_DESIGN_REVIEW_20260911.txt)、[实施计划](CORE90_GAME_V2_REPAIR_IMPLEMENTATION_PLAN_20260911.md)、[此前验收记录](CORE90_GAME_V2_IMPLEMENTATION_ACCEPTANCE_20260911.md)。本轮将代码可达、配置选择和运行激活分别核对；此前的19项本地验收结论不代替本次复核。

## 结论

再次审查发现2项P1、4项P2，共6项实际偏差或静默配置问题，均已修复。此前“完成本地验收”不能理解为没有遗漏；本次保留原始缺陷复现、先失败后通过的测试记录，以及独立定向复核。没有发现本轮尚未处理的P0/P1。

21份F1/强普通候选配置按真实CLI解析、与manifest逐项比对通过，开启组合符合计划：F1故意关闭恢复审计、控制器和能力课程。它验证求解器×对抗因子，不是把V2全部机制同时开启。没有启动新的正式E200，也没有在N607替换现有release。

## 发现与修复追踪

|ID|设计/计划来源|要求|位置|状态|验证及影响|
|---|---|---|---|---|---|
|A01|设计§10.1；计划B3|可靠低gap才能授权校正|controller.py:GameControllerV2.decide|verified|P1：历史gap=1使lag_exit=.5，当前未收敛HIGH/gap=.2原会CORRECT；现必须typed LOW才可CORRECT，typed HIGH只能进入追赶判断|
|A02|计划F1|18行对抗×求解器因子开关正确；统一关闭审计|config.py/build_core90_game_matrix.py及持久化JSON|verified|修前/修后21份实际CLI解析及manifest匹配均通过，未改变因子矩阵|
|A03|计划A—E|审计、控制、课程、B8和遥测实际可达|runtime及独立模块|verified|独立审查、受控实际入口动作测试、B8单步真实网络及遥测读回；不声称真实source必然触发控制|
|A04|计划F2/F3/G5/G6|区分已实现工具与未派发的正式对照|矩阵、donor、release与运行产物|verified|配置状态仍CONFIGURED_NOT_LAUNCHED；本次没有正式训练、target访问或远端部署|
|A05|计划B3/B5|零head预算不连带禁用EG校正|runtime_control.py/controller.py|verified|P2：correction＋max_extra_head=0原定标永远None；现零head合法且禁止CATCHUP，CORRECT由独立质量/预算控制|
|A06|计划E5|连续3次进入门槛确认才升级|curriculum.py|verified|P1：1次enter＋2次中间带原会升级；现中间带保留滞回ready但清零连续enter计数，必须3次连续合格新观测|
|A07|计划E3|课程不变不能仅因E91清历史|runtime.py|verified|P2：实际E90/E91两步入口复现；能力课程不再误reset，固定课程仍reset，真实loss/阶段变更保护保留|
|A08|计划E4|current能力测量使用实际难度|runtime_control.py/step_context.py|verified|P2：固定E41/E131原均误用level0；现读取实际epoch的p/场景比例。固定课程next=current，明确无自适应候选，不假装连续level|
|A09|计划A—E开启前提|矛盾的no-audit与依赖审计机制不能静默接受|config.py|verified|P2：no-audit＋control/capability/H1/J1原可静默不执行；现解析时报错。独立B8不受影响，负head预算也在入口拒绝|

本次专项追踪：verified=9、deferred=0、rejected=0、blocked=0。原21项设计追踪仍为19项完成本地范围、R19/R20两项按计划暂缓；R15/R17/R21中明确的donor正式对照、远端release、E200与新评分仍未执行。上述“verified”均以各行写明的范围为限。

## 当前机制/参数是否开启

以下为持久化JSON经`parse_args`解析后的值，不是只读README，也不是实际训练结果。[完整解析与阶段权重](../local_artifacts/core90_v2_reaudit/config_activation_after_fix.json)。

|配置组|主求解器|身份域对抗|B8实现|恢复审计/控制器|课程|
|---|---|---|---|---|---|
|V2_A×3seed|simultaneous|关闭|不适用|关闭/关闭|旧固定课程|
|V2_B×3seed|simultaneous|原调度，lambda_adv=.35|不适用|关闭/关闭|旧固定课程|
|V2_C×3seed|head_lookahead|关闭|head_grad_only|关闭/关闭|旧固定课程|
|V2_D×3seed|head_lookahead|原调度，lambda_adv=.35|head_grad_only|关闭/关闭|旧固定课程|
|V2_E×3seed|extragradient|关闭|不适用|关闭/关闭|旧固定课程|
|V2_F×3seed|extragradient|原调度，lambda_adv=.35|不适用|关闭/关闭|旧固定课程|
|3个强普通source候选|simultaneous|原调度，lambda_adv=.35|不适用|关闭/关闭|旧固定课程|

公共参数：scratch、E200、lr=2e-4（强普通候选为1e-4/2e-4/4e-4）、head_lr_ratio=1、FP32、`game_deterministic=true`。实际对抗权重沿用历史warmup：开启对抗的行E1为.245，E79以后为.35；关闭对抗行全阶段为0，不能把参数lambda_adv=.35写成从E1始终使用.35。卫星CE系数.68且E80才计入loss，卫星一致性系数0；U和EMA开启，U训练从E131开始。原有其他loss未删除，FISHR仍是既有解析logit-gradient proxy。

`game_telemetry_interval=250`仍开启，第一步及后续采样步记录角色梯度/位移；B8采样步实际构造完整目标wrapper并产生component telemetry。它与`game_no_audit`关闭的恢复/能力审计不同。graph_reuse虽然已通过此前有界FP32验收，但当前F1选择的是D1，不是graph；这符合计划先采用已验收实现的范围。

裸CLI默认是V2、ordinary、control=off、curriculum=fixed、审计开启、B8实现reference、确定性开关false。它不代表F1完整配置。更重要的是，矩阵生成器的旧名称`B8/S4/C2`仍显式生成V1以保留历史复现；直接调用这些名字不会自动启用新恢复探针或新课程。新版机制实验必须显式选择V2和对应control/curriculum，不能拿旧row名称当激活证据。

选中source探针为lr=.002/120步，但通用parser保持40步；21份F1配置的审计已关闭，因此该默认不影响它们。未来机制实验需将选中的两项参数显式合入其配置，或在使用该实验JSON时追加`--game_probe_lr .002 --game_probe_steps 120`。CLI一次只读取一个`game_config_json`；重复传两个JSON不会合并，不能用第二个三字段probe文件覆盖完整实验配置后仍声称保持原row。

## 实际训练接线与未证明部分

入口是`code/SSDG/train_core90_game.py`→真实parser→`runtime.train`。V2且审计开启时才创建`V2Coordinator`；两时钟分别消费观测，至少3个有效source观测并满足定标起始步数后才有控制器/课程。声明开启不保证动作非零：恢复未知、方向迁移失败、身份保护不满足、未完成定标或预算不足时，应保留NORMAL或当前课程。

动作门控现保留估计对象语义：HIGH仅支持可恢复空间与追赶候选；LOW且完整、代表性的方向证据可靠才可校正。历史source候选仍为`TRANSFER_FAILURE`，本轮没有重新拟合、挑选或消除该失败。能力课程仍使用同一fit-only身份头观察clean/current/next，独立于lag；连续3次进入条件、250步冷却、每次.1上限和连续实际场景混合均保留。

独立B8复核使用真实CORE90网络、受控合成输入、E131目标、CPU FP32的一个更新：reference/D1/graph模型状态最大绝对差均0，前向调用5/5/3（该fixture含上下文准备计数），遥测均AVAILABLE。此前确定性CUDA9组长计时和18个120步终点核对不因本次纯控制/课程修复重跑；本轮没有修改B8求解器算术路径。该证据不外推真实长期控制激活或识别性能。

精确性边界未改变：经验gap是有界源训练子集的优化空间，不是精确最优反应；H1仍是条件近似，J1仍是局部诊断；图复用只验收FP32。最重要的剩余科学风险仍是可靠测量在真实训练过程中能否迁移、触发有用动作并带来配对收益，不能由本次修复或单测回答。

## 本次验证与交付

本轮5份绿色JUnit记录共76个去重用例全部通过，没有未解决失败；这是分阶段聚焦回归，并非一次全集成命令。覆盖控制器、V1兼容、真实runtime动作与恢复、课程、实际固定能力视图、配置组合、矩阵和确定性入口。原红测与一次E91测试fixture的range替换错误均保留，fixture修正后才确认生产缺陷，未把测试装置错误冒称生产问题。

实际命令均使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8`，包括：

```text
-m pytest code/tests/test_game_tracking_controller_v2.py code/tests/test_game_tracking_activation.py code/tests/test_game_tracking_runtime_v2.py -q
-m pytest code/tests/test_game_tracking_controller_v2.py code/tests/test_game_tracking_audit_control.py -q
-m pytest code/tests/test_game_tracking_curriculum_v2.py -q
-m pytest code/tests/test_game_tracking_runtime_v2.py code/tests/test_game_tracking_coordinator_v2.py code/tests/test_game_tracking_curriculum_v2.py code/tests/test_game_tracking_activation.py -q
-m pytest code/tests/test_game_tracking_config_v2.py code/tests/test_game_tracking_matrix.py code/tests/test_game_tracking_determinism.py code/tests/test_game_tracking_runtime_v2.py -q
code/scripts/audit_core90_v2_activation.py --output local_artifacts/core90_v2_reaudit/config_activation_after_fix.json
```

每次pytest实际均另指定对应`--junitxml`路径。[控制缺陷独立报告](../local_artifacts/core90_v2_reaudit/control_review.md)、[课程/B8独立报告](../local_artifacts/core90_v2_reaudit/solver_curriculum_review.md)、[控制修复独立复核](../local_artifacts/core90_v2_reaudit/control_fix_followup.md)、[课程修复定向复核](../local_artifacts/core90_v2_reaudit/fix_followup.md)、[去重测试清单](../local_artifacts/core90_v2_reaudit/verification.json)。证据保存在其原始相对目录并纳入Git，避免搬移诊断脚本后破坏ROOT解析。

只提交本轮代码、测试、审查报告和相应证据；保留无关`report_core90_target_test.py`。Git提交及独立远端OID读回结果见本任务最终交付消息。
