# D105-CBRC-MB4+LPO-RC-qKNN单seed Target25实验报告

状态：`PHASE1_R6_SYSTEMIC_TECHNICAL_FAILURE / D105-FTU2_ARCHIVE_VERIFIED / R7_LOCAL_PREREGISTERING / TARGET25_NOT_STARTED / NO_D105_PERFORMANCE_RESULT`

## 1.实验标识

|字段|值|
|---|---|
|experiment ID|`d105_cbrc_lporc_target25_s713102_20260731_r1`|
|日期|2026-07-31|
|operator|主agent负责整合、数据和结果分析；terra-max功能agent负责非重叠模块；N607另设唯一terra-max runner|
|协议|`p2_min_v1`|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`；单seed研发screen；未完成时不得报告性能；通过也不等于多seed稳定或`PROMOTABLE`|
|GitHub|不push、不上传；仅本地Git版本化|

## 2.目标与假设

目标是在D62和D92的同一row协议上验证四臂`M0/M_DA/M_HEAD/M_JOINT`。共享DA只读取不可变Phase1聚合bundle和当前row合法target old/new support；HEAD只读取当前臂target support和冻结qKNN lock。假设D105的非等距低维表示修正能够提高旧类和弱类地板，LPO-RC能够改善新类注册，两者联合不牺牲任一任务。

硬目标：

|slice|注册后旧类准确率|最低旧类准确率|新类准确率|
|---|---:|---:|---:|
|K10/new5|≥92%|≥85%|≥92%|
|K10/new10|≥92%|≥85%|≥90%|
|K10/new20|≥92%|≥85%|≥86%|

K5/new20相对matched K10/new20的`A_old/F_old/N/H`下降均不得超过5pp。K1/new20相对同row冻结D92必须满足`ΔH≥2pp、ΔF_old≥2pp、ΔA_old≥0、ΔN≥0`且old+new总正确数严格增加。

## 3.冻结矩阵与比较对象

```text
receivers={20-1,3-19,7-14,7-7,8-8}
seed=713102
slices={K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}
scenarios={leo_clear_weak,leo_low_elev_weak,leo_rain_weak}
arms={M0,M_DA,M_HEAD,M_JOINT}
25 jobs × 3 scenarios × 4 arms = 300 scenario-arm pairs
每个pair含before(S_B)与after(S_C)，共600个state prediction surfaces
```

K5的support physical IDs必须是同receiver、同scenario、同capsule的K10/new20子集，query root必须逐scene完全相同。比较对象为同row D62、D92和SVRN/r4.2；D91只保留其单development cell边界，不冒充Target25或125。

## 4.当前本地实现与验证

|文件/组件|用途|当前状态|
|---|---|---|
|`stage2_d105_cbrc.py`|共享DA|本地独立复审`P0=0、P1=0`|
|`stage2_lpo_rc_qknn.py`|纯support HEAD|本地独立复审`P0=0、P1=0`|
|`stage2_d105_four_arm.py`|四臂集成|本地独立复审`P0=0、P1=0`|
|`stage2_d105_feature_tap.py`|真实checkpoint同IQ单次`z_id/z_dom/hidden/pre_relu`tap|真实checkpoint随机IQ smoke通过|
|`stage2_d105_target25_runner.py`|25job计划、四臂before/after预测封存、truth-side score及300-pair/600-state覆盖|R4统一回归通过|
|`stage2_d105_target25_launcher.py`|真实row context、GPU绑定和逐row执行|R4统一回归通过|
|`stage2_d105_target25_inputs.py`|签名D92/D81 authority和真实package到25行plan/context的唯一prepare入口|R4本地真实结构fixture通过|
|D105 Phase1 bundle/authority|source-only压缩知识、签名D102撤销和可信外部formal seal|R4统一回归通过；生产D105 signature未生成|
|D105 query evaluator|D92 sealed row package到四臂预测|R4闭包与预检顺序回归通过|

已核验checkpoint：

```text
E:\type10-7\automation_reports\CV-SincNet\d105_feature_tap_real_checkpoint_smoke_20260731\input\best_joint_safe_ssdg.pth
SHA256=2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98
```

真实checkpoint tap smoke：

```text
status=PASS
rows=3
z_id=[3,160]
z_dom=[3,160]
hidden=[3,320]
pre_relu=[3,160]
relu_exact=true
state_unchanged=true
query_truth_read=false
labels_or_roles_passed=false
artifact_sha256=65e67bb763aeb1d2eb20b7f577923745d2f91532c83159902a7667123950604a
```

## 5.N607输入、环境与拟定落地面

2026-07-31 14:24 HKT只读preflight曾通过：主机`dell-DSS8440`，项目根可见，GPU0–7均空闲。该状态不是当前资源保证；正式release前仍需由唯一runner再次执行preflight并记录即时占用。

|输入|N607路径|
|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|cache root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`|
|authority root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_cbrc_lporc_target25_s713102_20260731_r1`|
|source/CWD|`<run-root>/source/code`|
|main log|`<run-root>/logs/target25.log`|
|PID/exit|`<run-root>/logs/target25.pid`、`<run-root>/logs/target25.exit`|
|GPU计划|GPU0–7，每GPU最多1个本run worker；不超过项目默认每GPU两个训练实验|

现有D92/D81签名authority为`formal_launch_authority=false`，因此本run只能使用唯一prepare入口生成`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`的plan/context。该声明、authority envelope root和`formal=false`必须贯穿prediction、score与summary，不能重标升级。

精确服务器命令、Git commit、源码包哈希、Phase1 bundle/validator/runtime/method-lock哈希将在R2复审`P0=0、P1=0`且本地提交后写入；在此之前不得落地或启动。

## 6.预期artifact与完整性

- 不可覆盖的run manifest、25个row log和25个exit；
- 25个row prediction artifact，每个包含3scenario×4arms×before/after；
- 完整prediction manifest，300个scenario-arm pair receipt和600个state prediction receipt唯一；
- prediction全部封存后才允许生成truth-open event；
- 25个truth-side row score和完整score manifest；
- 300个scenario-arm paired score、600个state score、逐类count/correct、old/new同row指标；
- K5/K10嵌套receipt、K1逐值恒等receipt；
- 汇总JSON/CSV、资源审计、异常指纹、coverage和归档哈希。

只有完整日志、25/25成功row、300/300 pair、600/600 state预测和600/600 state score全部同键闭合，状态才可进入`ARTIFACTS_COMPLETE→ANALYZED`。

## 7.健康停止与成功标准

性能高低不得触发停止。仅当发生P0协议/安全错误，或至少两个不同row在产生预测前出现同一标准化确定性异常指纹时，停止后续派发并终止已证明属于本run的进程树。失败run保留全部partial artifact，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得覆盖、恢复或重标为性能实验。

技术成功条件：25/25 row退出0、300/300 pair、600/600 state预测、600/600独立state score、query零fit/update、K5嵌套、K1恒等、所有hash/receipt/只读门通过。性能成功条件按§2硬目标判定。

## 8.风险与完成后检查

|风险|处理|
|---|---|
|D102 Phase1 held已`FALSIFIER_REJECT`|禁止复用其REJECT bundle；D105必须生成并独立验证新bundle|
|D92 runtime不含`pre_relu`|使用D105专用checkpoint-bound feature tap，不从归一化`z_id`反推|
|support代理过拟合|不按support准确率晋级；只读完整Target25独立query score|
|旧类均值掩盖弱类|报告全部旧类count/correct、`F_old`、row-floor、receiver/scene|
|只完成联合臂或after state|300-pair/600-state覆盖门直接拒绝|
|单seed偶然性|通过仅记`TARGET25_SCREEN_PASS`；fresh confirm seed另行预注册|

完成后重点检查四臂简单效应`E_DA/E_HEAD`、交互项`I`、旧类错误流、弱类地板、K5衰减和K1净新增正确数，再决定晋级或下一轮DA/HEAD修订。

## 9.seed713102严格配对基线

已从N607只读回收D62、D92完整125汇总的`row_metrics.csv`，并分别筛选`seed=713102`的25行；SVRN-qKNN-BCRR由同seed的25个原始评分文件重算。SSH/SCP结束后本地无残留`ssh.exe`。

|来源|本地SHA256|行数|同seed行数|
|---|---|---:|---:|
|D62|`5b737206b0ebb222443996e71e76cdd2209929ae77ec3f6295688ae649b5807d`|125|25|
|D92|`bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62`|125|25|
|SVRN-qKNN-BCRR|25个原始`diag_cosine_score.json`逐文件解析|25|25|

基线联合表已写入Git承载面`analysis/d105_target25_baseline_seed713102.md`。当前可验证结论是：D92在K10/new10、K10/new20和K5/new20提高注册后旧类并降低遗忘，但新类小幅下降；K1与D62逐值一致；SVRN在全部切片显著更差。D105尚无性能数据，不能加入排序。

## 10.R1拒绝、R3作废与R4闭包修复

R1独立release审查结论为`NO-GO / P0=1 / P1≥4`，原因包括unsigned Phase1 authority自述、缺少真实D92→plan/context入口、runtime闭包遗漏launcher/CLI和D102拒绝仅靠名称。R1未提交、未落地、无性能结果。

R2/R3完成authority和执行链后，独立复核在N607落地前发现两条清单外动态执行路径：tap-cache导入legacy exporter，query evaluator经通用`checkpoint_loading`进入训练栈。第一次54文件修复后，R4 reviewer又发现`model_dual_cvsincnet.py`会在来源校验前探测清单外`model_modified.py`。因此旧45文件R3 review、旧`r2`运行预登记及中间54文件哈希全部作废，且均未落地。

当前R4修复：

- 固定Ed25519信任根、独立review receipt、时间窗、nonce防重放和完整formal authority artifact；
- D102r6真实内容identity签名撤销；
- 唯一Target25 prepare CLI，从现有签名diagnostic authority和真实封存package派生25行，不接受调用者自报physical ID/root/registry；
- 独立`TARGET25_PREPARE`签名域精确绑定matrix、plan、context、prepare receipt、Git commit、run ID、候选锁及N607 nonce ledger identity；非dry-run prediction在执行前消费nonce，dry-run/score只验签；
- Phase1和Target25均以“本机账本绝对路径＋run ID＋签名域”重算跨主机ledger identity，拒绝替换账本路径、run或签名域；
- development claim不可升级，K5嵌套、query root一致、25/300/600覆盖和600预测后才开truth；
- R3只完成N607受控落地，canonical loader在py_compile/checkpoint/detach前发现24/54项manifest与Git archive字节不一致；pipeline pid/log/exit始终不存在，终态为`LANDED_PRELAUNCH_HASH_MISMATCH / NO_PERFORMANCE_RESULT`；
- 根因为Windows工作树CRLF与Git blob LF不一致；`.gitattributes`现固定`*.py`/`*.sh`为LF，54个runtime文件与当前Git blob逐项SHA一致，并新增CRLF负测；
- repaired candidate runtime SHA256=`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`；
- repaired candidate method lock SHA256=`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`；
- `ssr-gpu`统一回归212项通过；同代真实checkpoint无query R6 smoke收据SHA256=`a954896a5b3e3db91334ac564d967705568c892b5d2b7c6dbe42111a03d7c76c`；query fit/update=0/0、Target访问=false、性能计算=false。
- 修复提交`46a65b3af2621d23bcc0a34631f45c8be17af4dd`的精确Git archive SHA256=`d313243c79eab306f988abadf67c2e207d380dba633f39a04e2cc63ffae7ed7a`；解包副本的canonical loader、54 pyc、9帮助面和真实checkpoint无truth smoke均通过，archive smoke SHA256=`a915eb66c4df926e6f738a4de636026fa29cb9bf3968c5fb6a15007ffc47ce84`。

R4的`LOCAL_RELEASE_GO`已被跨平台发布字节P0作废。R5已在修复提交的精确Git archive上独立完成54文件canonical loader/编译、真实checkpoint安全加载策略、动态依赖guard、8项关键测试、9个帮助面和archive smoke，最终结论`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`，receipt SHA256=`65f8f211c01b8b72b4f4d7a385d9c1747b16dae9f14bddd32457ccf2f402c822`。

新的Phase1运行曾预登记为`d105_phase1_sourceheld_d23469ba_20260731_r4`，源码只允许使用提交`d23469ba54afe00c284aa9b78b025def2b22fc43`的精确Git archive，新run root为`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_d23469ba_20260731_r4`。预登记时该run尚未落地；只有完成Phase1 source-held闭环、独立审查、离线authority签名和formal seal，才允许启动Target25。D105仍无任何Target性能数据。

R4随后在全部预启动门通过后唯一detach一次，但首个`tap-cache`在任何strict tap或预测前以`TypeError: expected np.ndarray (got numpy.ndarray)`退出。exit=1，所有Phase1正式artifact为0，未访问Target、未启动Target25、未执行authority签名或seal。R4永久关闭为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整交接SHA256=`f362f5051a71d0dd88552a815c2a82680b6157a537ef1ebc36b1d8e720a3811a`。只有完成本地修复、回归、独立审查、提交和新run预登记后，才可再次申请Phase1 release。

NumPy/Torch边界已在Phase1输入、Target25查询输入和feature tap输出三面完成最小修复，提交=`9f608e8be72024f00f1497cf6bddb9fb77e28201`。统一回归216/216通过，R7独立审查`P0=0、P1=0、P2=2`；新runtime/method=`8940e05f…85425`/`f36a0c6c…4864e`。该提交的精确Git archive SHA256=`dd85491e…9a67`，解包副本54/54、54 pyc、9帮助面和真实checkpoint无truth smoke全部通过，smoke收据SHA256=`fdea3e39…9a07`。

新的Phase1 R5预登记为`d105_phase1_sourceheld_9f608e8b_20260731_r5`，预登记提交=`27fccbfc1d49599a4c9e5e82d301780b02fbad37`，新run root为`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_9f608e8b_20260731_r5`。它尚未落地，且不得读取或复用R4的任何运行内容。只有R5完整source-held闭环、独立component审查、离线authority签名和formal seal全部通过，才允许启动Target25。D105仍无任何Target性能数据。

R5随后通过全部远端prelaunch门并唯一detach一次，但首个`tap-cache`以exit=2结束，完整日志仅报告`strict tap must expose byte-bound z_id/pre_relu and z_dom`。所有Phase1正式artifact仍为0，无Target、Target25、authority或seal操作。R5永久关闭为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，handoff SHA256=`5f390e0220d5168948a7a1cf4a2e964dfc3961cfdcdcb667a9db99a77fcd88ab`。第四次release前必须先完成三轮正式回顾和新根因闭包。

R3/R4/R5三轮正式回顾已完成。真实checkpoint复现证明Phase1接入的GRB旧tap只前向`id_backbone`，`z_id/pre_relu`有效但`z_dom=None`；D105专用双backbone tap在同一模型、同一IQ上正确输出`z_dom=[2,160]`。下一技术候选`D105-FTU1`只统一正式特征出口并补齐入口级真实checkpoint回归，不改变CBRC、LPO-RC、qKNN、Target25矩阵、seed或性能门。Target25继续保持未启动，只有新Phase1资产链完整通过后才可进入。

`D105-FTU1`现已在提交`a0bdbba6`落地：真实checkpoint 195 tensors下`z_id/pre_relu/z_dom=[2,160]`，fresh进程调用前后旧GRB模块均未加载，10文件223/223回归通过，runtime/method=`873879aa…9214`/`7d336627…4848`。R10独立审查为`LOCAL_CODE_REVIEW_GO / P0=0 / P1=0 / P2=2`。这只是Phase1本地技术闭环；精确archive smoke、新run预登记、Phase1 source-held资格和formal seal仍未完成，Target25保持未启动。

精确archive smoke现已通过：提交`a0bdbba6bfb56c45682e0c2bde95aa622a68f101`的archive SHA256=`99fd633c78070b940064ca6e95ca9072427457058cab96c3a61e584c7991c0b4`，54/54文件四面一致、54/54独立编译、真实checkpoint严格双分支前向、9/9帮助面、8/8 FTU1定向和223/223冻结回归全部通过。R6已冻结为`d105_phase1_sourceheld_a0bdbba6_20260731_r6`，本地预登记提交=`814d3b1d51ce764e67c1125492886fb1a4f6b03e`，仅用于重新生成Phase1 source-held证据；Target25继续保持未启动、无D105性能结果。只有R6完成component/score/gate、独立component审查、离线authority签名与formal seal后，才能另行预登记Target25落地。

R6唯一detach后在8400行strict tap与reference dual的数值parity门失败，prediction、truth、score、gate、component和formal asset均为0，因此永久关闭为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。全量差分将根因闭合为参考固定256补零合同与R6可变batch=128合同不一致；异常严格集中在最后80行partial batch。`D105-FTU2`只修复固定256容量、末批补零/切回和receipt绑定，不改模型、方法或阈值。Target25继续保持未启动。

FTU2本地实现现已通过238/238统一回归和独立`GO / P0=0 / P1=0 / P2=0`审查，runtime/method更新为`8797de12…a498`/`9a87e51d…61e6`，提交=`2d948ce981b9008522f825cfe6d868bce08cb624`。该结论只允许进入精确archive smoke；Phase1尚无新formal asset，Target25继续保持未启动。

FTU2精确archive已通过54/54、9/9帮助面、238/238回归、真实checkpoint固定256和8400批合同验证。R7冻结为`d105_phase1_sourceheld_2d948ce9_20260731_r7`，预登记提交=`632fd9f0e1324d14cb9d489b92b71259e9ac29fe`，只用于重新完成Phase1 source-held闭环；首次8400行reference parity仍是N607硬门。Target25继续保持未启动。

## 11.R7 Phase1状态同步

R7的8400行fixed256 strict tap与reference parity已通过，但source-held derive-gate在两个合法负min_net_correct字段上被非负计数validator误拒，exit=2。prediction、truth-open和score虽已生成，但gate、component、authority与formal asset均未闭合；R7永久标记为STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT，不得读取未闭合评分作为性能结论。

因此Target25仍为未启动状态：没有Target输入、prediction、truth-open、score、authority、nonce消费或任何Target性能行。只有后续本地修复gate validator语义并通过独立审查、提交和新的不可覆盖Phase1 release后，才可重新评估Target25启动资格。
