# D105 Phase1 source-only压缩知识与source-held门实验报告

状态：`PHASE1_R5_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / RETROSPECTIVE_REQUIRED / NO_PERFORMANCE_RESULT`

## 1.实验标识与目标

|字段|值|
|---|---|
|experiment ID|`d105_phase1_sourceheld_asset_20260731_r1`|
|日期|2026-07-31|
|operator|主agent整合；terra-max Phase1功能agent实现；正式N607由另设唯一terra-max runner执行|
|协议边界|仅Phase1 source-only；`target_rows=0`、`query_rows=0`|
|目标|从真实ADV3B02 checkpoint和经验证的source weak-IQ生成D105严格`pre_relu/z_dom`tap，完成receiver-held K1/K5/K10、class-LOCO、TX probe与INT8量化门，生成class-free/receiver-free聚合bundle并独立封存|
|性能边界|本run只决定Phase1资产是否具备formal Phase2输入资格，不产生Target性能结论|
|GitHub|不push、不上传；仅本地Git版本化|

## 2.核心假设

D105共享DA所需的Phase1知识可以压缩为不含类名、receiver名、physical ID、原始IQ或逐样本特征的INT8/FP16聚合状态。该状态只有在完整source-held非劣门、低TX可预测性、量化一致性、独立复审与外部authority seal全部闭合后，才允许作为Phase2不可变输入。

D102的`PHASE1_HELD_FALSIFIER_REJECT` bundle、held结论和其formal=false资产一律禁止复用。可重用的只有独立SHA绑定的原始source cache、selection salt、真实checkpoint和通用读取实现；D105必须重新产生自己的strict tap、预测封存、truth-open、独立score、gate、component与seal。

## 3.预登记输入

|输入|N607路径|SHA256/状态|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|source validation cache set|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|selection salt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|dual archive parity reference|`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8/output/archive/phase1_singleobs_dual_feature_archive.npz`|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；仅作字节一致性参考，不直接转封为D105资产|

运行环境拟定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，GPU0执行严格tap和source-held预测。正式release前由唯一runner再次执行只读preflight并记录即时GPU占用。

## 4.不可变生命周期

```text
source weak-IQ
  -> D105 strict tap archive/receipt（无Target、无query）
  -> source-held prediction manifest（无truth，只读）
  -> truth-open receipt（绑定prediction manifest SHA）
  -> independent score artifact
  -> gate recomputation
  -> non-formal aggregate component
  -> independent review P0=0/P1=0
  -> external authority seal
  -> formal D105 Phase1 asset/runtime handle
```

缺少任一步、文件可写、哈希漂移、D102 REJECT lineage、truth先于prediction、held覆盖不完整或任一门失败时，必须保持`formal_phase2_eligible=false`，不得供Target25加载。

## 5.冻结source-held覆盖与门

- 每个source receiver均覆盖K1、K5、K10 receiver-held预测；
- 每个source receiver×class覆盖class-LOCO；
- receiver-held和class-LOCO均要求D105相对M0的balanced accuracy、最低类准确率和净正确数不劣；
- 每个receiver完成TX probe，最大balanced accuracy≤25%；
- INT8相对FP32 top1一致率≥99.5%，大margin flip为0；
- 不持久化原始IQ、clean IQ、逐样本FP32 sidecar、source replay或类/receiver/physical ID；
- 独立代码复审要求`P0=0、P1=0`。

这些门只授予Phase1资产资格，不等于D105四臂Target性能或`PROMOTABLE`。

## 6.拟定N607输出

|类型|路径|
|---|---|
|旧run ID|`d105_phase1_sourceheld_8f08a46f_20260731_r2`；从未落地，因45文件闭包遗漏而作废|
|失败run ID|`d105_phase1_sourceheld_2eaa1b11_20260731_r3`；永久停止于预启动哈希门|
|失败run Git提交|实现`2eaa1b11b4d720673fa999025939058918efc63d`；预登记`bf9406429cb5540e785bc3e61434c682ab548bb9`|
|失败run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_2eaa1b11_20260731_r3`；保留且不得覆盖、恢复或复用|
|source snapshot|`<run-root>/source`|
|input receipts|`<run-root>/input`|
|strict tap|`<run-root>/output/strict_tap`|
|prediction manifest|`<run-root>/output/source_held_prediction`|
|truth-open与score|`<run-root>/output/source_held_score`|
|component|`<run-root>/output/component`|
|formal sealed asset|`<run-root>/output/formal_asset`|
|main log/PID/exit|`<run-root>/logs/pipeline_stage1.log`、`pipeline_stage1.pid`、`pipeline_stage1.exit`|
|GPU|`cuda:0`；仅strict tap执行backbone，source-held矩阵为冻结解析执行|
|R5 repaired candidate runtime|`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`；54文件，全部与当前Git blob字节一致|
|R5 repaired candidate method lock|`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`|
|R4 review receipt|历史SHA256=`af312331c81c638240c8d8245f69513d4c9f8bf63fe4b8adff0f1e63e414fc51`；被N607预启动字节不一致证据作废|
|D102 revocation manifest/signature|`99393aa21b30cc654ba784ecf8a60b1ac8497e67d7fdefc3ee12872133293734`/`53d138b36f7688431d364f2e6291e86aefb9c9fc3e84697e288f0aa813b81c58`|
|失败run stage1 launcher|`run_d105_phase1_stage1_2eaa1b11.sh`；SHA256=`6d540868f3347633846ffce92a2bc424e3f63593fc9a0b16333de3f7aad938de`；从未执行|

旧`r2`和失败`r3`的命令、路径与launcher均不得执行。`r3`原预登记命令为：

```bash
RUN=/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_2eaa1b11_20260731_r3
nohup bash "$RUN/input/run_d105_phase1_stage1_2eaa1b11.sh" \
  >"$RUN/logs/pipeline_stage1.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$RUN/logs/pipeline_stage1.pid"
```

该命令从未执行：`pipeline_stage1.pid/log/exit`始终不存在。失败run只完成Git archive、公开input和source落地；canonical loader在py_compile、checkpoint加载与detach前发现24/54项哈希不一致。完整交接SHA256=`cc969b37754280d9a78bbb7ff09aceacbd458809b90b2790a7770f6448136b23`。修复后必须新建run ID和launcher。

### 6.1 R5批准的Phase1 R4运行预登记

|项目|冻结值|
|---|---|
|新run ID|`d105_phase1_sourceheld_d23469ba_20260731_r4`|
|源码Git archive提交|`d23469ba54afe00c284aa9b78b025def2b22fc43`；必须直接由该提交生成精确archive，不得从工作树打包|
|预登记Git提交|`03e3ff67003f16b6c39596a521f5bfdf0401850c`；包含本报告、Target报告、追踪和launcher，不改变上行源码archive身份|
|新run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_d23469ba_20260731_r4`；落地前必须证明不存在|
|candidate runtime|`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`；54文件|
|candidate method lock|`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`|
|R5独立review receipt|`analysis/d105_release_review_r5_20260731.md`；SHA256=`65f8f211c01b8b72b4f4d7a385d9c1747b16dae9f14bddd32457ccf2f402c822`；`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`|
|D102 revocation manifest/signature|`99393aa21b30cc654ba784ecf8a60b1ac8497e67d7fdefc3ee12872133293734`/`53d138b36f7688431d364f2e6291e86aefb9c9fc3e84697e288f0aa813b81c58`|
|R4 stage1 launcher|`run_d105_phase1_stage1_d23469ba.sh`；SHA256=`b72fef60ab14ec86e9b53cc3355d07ac50598a3d6788000b4962726c89271d35`；5624B；LF-only；`bash -n`通过|
|GPU|`cuda:0`|
|fresh-run retry|`NO`；任何技术失败均关闭本run，修复后必须新run ID|

R4只能在直接N607 preflight、即时GPU/进程/磁盘盘点、run root不存在、源码archive结构安全、archive内54/54 manifest字节一致、远端canonical loader、54文件独立pyc编译、真实checkpoint加载策略记录和launcher `bash -n`全部通过后detach一次。源码archive与launcher必须分别同步：源码严格来自`d23469ba54afe00c284aa9b78b025def2b22fc43`，launcher严格采用上表字节。

唯一预登记启动命令为：

```bash
RUN=/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_d23469ba_20260731_r4
nohup bash "$RUN/input/run_d105_phase1_stage1_d23469ba.sh" \
  >"$RUN/logs/pipeline_stage1.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$RUN/logs/pipeline_stage1.pid"
```

R4已按§8永久关闭，上述命令与launcher不得再次执行。

### 6.2 NumPy/Torch边界修复后的Phase1 R5预登记

|项目|冻结值|
|---|---|
|新run ID|`d105_phase1_sourceheld_9f608e8b_20260731_r5`|
|源码Git archive提交|`9f608e8be72024f00f1497cf6bddb9fb77e28201`；只允许精确Git archive|
|预登记Git提交|`27fccbfc1d49599a4c9e5e82d301780b02fbad37`；包含报告、验证artifact和launcher，不改变源码archive身份|
|新run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_9f608e8b_20260731_r5`；落地前必须证明不存在|
|candidate runtime|`8940e05f9fdf92d7735bba1570bb3239ee210313ecbbeffa3511b62e21685425`；54文件|
|candidate method lock|`f36a0c6c4ee832b34cd98ed7664ec87707a4dbb1559c7c9b4b05dd13fbf4864e`|
|R7独立review receipt|`analysis/d105_numpy_boundary_fix_review_r7_20260731.md`；SHA256=`558053a1c0352fda8226a8d52adbf4252b685b54cb4fc6e8b4c5e64607fab842`；`GO / P0=0 / P1=0 / P2=2`|
|精确Git archive|SHA256=`dd85491e96f1cb9ea14e967694db91aec590e42273a2556492221af982ee9a67`；242851840B；4754项；单根、无链接/逃逸/重复|
|archive验证收据|SHA256=`c206f1d89dba723b6b6e70c5f67362cb9c4dd877b6918587d74700ed22addd74`；54/54、54 pyc、9帮助面、旧桥AST=0|
|archive真实checkpoint无truth smoke|SHA256=`fdea3e395b15d34ba7968037aa9a54ca835ec643e669c66de6854c8c3ff69a07`；400步、K1恒等、query fit/update=0/0、Target/performance=false|
|D102 revocation manifest/signature|`99393aa21b30cc654ba784ecf8a60b1ac8497e67d7fdefc3ee12872133293734`/`53d138b36f7688431d364f2e6291e86aefb9c9fc3e84697e288f0aa813b81c58`|
|R5 stage1 launcher|`run_d105_phase1_stage1_9f608e8b.sh`；SHA256=`83bc12edd5db9b177e6f38fe589c1d25d9a7bea2da366e29fd27306303bef56c`；5624B；LF-only；`bash -n`通过|
|GPU|`cuda:0`|
|fresh-run retry|`NO`；R5若技术失败则永久关闭并新建run|

R5必须重新执行direct preflight、即时资源盘点、run root不存在、远端archive和54/54字节门、54文件独立pyc、checkpoint加载策略及launcher检查。R4旧root、source、input、launcher、PID、日志和output均不得读取为R5输入或复用。

唯一预登记启动命令为：

```bash
RUN=/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_9f608e8b_20260731_r5
nohup bash "$RUN/input/run_d105_phase1_stage1_9f608e8b.sh" \
  >"$RUN/logs/pipeline_stage1.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$RUN/logs/pipeline_stage1.pid"
```

run-specific脚本按`tap-cache→predict-source-held→open-truth→score-source-held→derive-gate→build component`固定顺序执行，任一步失败即退出并以不可覆盖方式写`pipeline_stage1.exit`。脚本不封存formal asset；stage1结束后必须先回收完整component/score/gate，由独立审查确认source-held门，再由离线authority绑定N607 nonce ledger identity、run ID、commit和component签名。生产私钥不进入N607。

## 7.健康停止与判定

性能高低不得触发提前停止。仅P0协议/安全错误、输出覆盖风险、错误checkout/hash、truth顺序违规或至少两个不同held row在预测前出现同一标准化确定性异常指纹时停止本run。停止后保留全部partial artifact，并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

技术完成要求：strict tap、prediction manifest、truth-open、score、gate、component、authority seal和formal validation全部同哈希闭合，完整日志可读，退出码0。若source-held性能门失败但技术闭合，则状态为`COMPLETED_SOURCE_HELD_REJECT / NO_TARGET_LAUNCH`，不得生成formal seal或启动Target25。

## 8.当前进展

- 已完成真实checkpoint随机IQ feature-tap smoke，checkpoint字节和`pre_relu/z_dom`形状闭合；
- 已完成strict source tap→无truth只读prediction→truth-open→独立score→重算gate→aggregate component闭环；
- R1独立release审查发现unsigned authority自述可将component升级为formal，判为`P0=1`并`NO-GO`；R1未提交、未落地；
- R3审查后、N607落地前发现tap-cache动态导入两个legacy exporter，query evaluator动态调用通用`checkpoint_loading`训练栈；因此旧45文件闭包、旧R3 review和旧`r2`运行预登记全部作废，均未落地；
- R4修复把tap-cache选样、salt读取和最小checkpoint重建收回D105闭包，query evaluator改用相同的受绑定模型工厂，并在构模前完成包、authority、split、同IQ、Phase1资产、qKNN、checkpoint与device预检；
- 正式authority仍使用固定`somph_runtime_trust`Ed25519公钥和D105专用signature domain；formal asset完整保留authority envelope、detached signature、独立review receipt、签名D102 revocation manifest及signature；
- formal authority签名绑定N607预先创建的nonce ledger identity；该identity由N607账本绝对路径、run ID和签名域规范化产生，离线签名端只接收摘要，封存端必须在消费nonce前以本机路径重算一致；
- D102r6的bundle manifest、payload、seal、content root、method lock、runtime、held score和tap archive真实SHA均进入内容撤销项，改名副本不能绕过；
- R4独立审查进一步发现模型工厂在来源校验前探测清单外`model_modified.py`，判为新P0；现已改为确定性导入清单内`model.py`并增加负测，中间锁与R4 smoke随即作废；
- R3在N607只完成受控落地，随后canonical loader发现24/54项manifest期望SHA与Git archive实际字节不一致；未执行py_compile、checkpoint加载或detach，pipeline三件套始终不存在，终态为`LANDED_PRELAUNCH_HASH_MISMATCH / NO_PERFORMANCE_RESULT`；
- 根因是冻结manifest使用Windows工作树CRLF字节，而Linux部署使用Git blob LF字节；本地已把全部54文件与Git blob逐项对照，32个受影响工作树文件仅存在CRLF→LF差异，未发现非行尾内容漂移；
- `.gitattributes`新增`*.py text eol=lf`，54个runtime文件全部规范为LF；新增回归要求Python/Shell属性固定且runtime中无CRLF；
- repaired candidate runtime SHA256=`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`，54/54项与当前Git blob SHA一致；method lock SHA256=`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`；
- `ssr-gpu`统一回归212项全部通过；R5独立技术复核最终结论`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`，receipt SHA256=`65f8f211c01b8b72b4f4d7a385d9c1747b16dae9f14bddd32457ccf2f402c822`；
- 同代真实checkpoint无query R6 smoke收据SHA256=`a954896a5b3e3db91334ac564d967705568c892b5d2b7c6dbe42111a03d7c76c`；400个source-held meta step完成，K1恒等成立，query fit/update=0/0，Target访问=false，性能计算=false；
- LF/manifest/test/report修复提交=`46a65b3af2621d23bcc0a34631f45c8be17af4dd`；由该提交生成的精确Git archive共4747项、242800640B、SHA256=`d313243c79eab306f988abadf67c2e207d380dba633f39a04e2cc63ffae7ed7a`，单一`source/`根且无链接或异常成员；
- 在上述archive解包副本内，canonical runtime/method loader、54文件独立pyc编译和9个CLI/关键子命令帮助面全部通过；同一解包副本的真实checkpoint无truth smoke SHA256=`a915eb66c4df926e6f738a4de636026fa29cb9bf3968c5fb6a15007ffc47ce84`；
- R5 reviewer在隔离解释器中以真实checkpoint验证`weights_only_with_explicit_safe_globals`、195 tensors、eval=true、factory/backbone均来自archive；SSDG、通用`checkpoint_loading`、paper路径、`model_modified`和两个legacy exporter均未导入，8项关键guard/query-tamper测试通过；
- 2026-07-31 14:24 HKT只读N607 preflight通过，8张RTX 3090均空闲；盘点结束后本地无残留SSH进程或到N607/bridge的ESTABLISHED连接；
- 上述GPU状态仅是历史只读盘点，正式release前必须重新preflight；
- R4独立审查的本地代码结论被跨平台发布字节P0作废；R5已把Git archive字节同一性作为硬门并达到最终`P0=0、P1=0、P2=2`；
- 新Phase1 R4已用非覆盖run ID、精确源码提交、独立launcher、一次detach和禁止retry规则完成预登记，但尚未落地；
- R4在全部远端预启动门通过后唯一detach一次，PID=`2726125`，首个`tap-cache`在生成strict tap前以exit=1退出；完整日志992B，归一化异常指纹=`TypeError: expected np.ndarray (got numpy.ndarray)`，位置`stage2_d105_phase1_bundle.py:1658`；
- R4的strict tap、prediction、truth-open、score、gate和component均为0，未访问Target、未启动Target25、未签名或seal；运行后runtime仍54/54一致，GPU、run进程与SSH均已清理；
- R4永久关闭为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重启、修补、覆盖或复用；完整交接SHA256=`f362f5051a71d0dd88552a815c2a82680b6157a537ef1ebc36b1d8e720a3811a`；
- 已用受检`frombuffer→reshape→clone→to`输入桥和`tolist→float32`输出桥同时修复Phase1与Target25正式执行面，未改变方法或协议；修复提交=`9f608e8be72024f00f1497cf6bddb9fb77e28201`；
- `ssr-gpu`统一回归216/216通过；R7独立复核`P0=0、P1=0、P2=2`，54文件编译和canonical runtime/method闭合；
- 修复提交的精确archive SHA256=`dd85491e…9a67`，4754项；解包副本54/54、54 pyc、9帮助面和旧桥AST扫描通过；真实checkpoint无truth smoke 400步通过，收据SHA256=`fdea3e39…9a07`；
- 新Phase1 R5已用新run ID和独立launcher预登记，提交=`27fccbfc1d49599a4c9e5e82d301780b02fbad37`，尚未落地；D105仍无Phase1 formal asset或Target性能数据。
- R5通过全部远端prelaunch门后唯一detach一次，PID=`2770709`，首个`tap-cache`以exit=2结束；完整日志313B/4行，异常=`strict tap must expose byte-bound z_id/pre_relu and z_dom`；
- R5的strict tap、prediction、truth-open、score、gate和component仍全部为0，未执行Target、Target25、authority或seal；终态GPU、run进程和SSH均清理；
- R5永久关闭为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重启、覆盖或复用；handoff SHA256=`5f390e0220d5168948a7a1cf4a2e964dfc3961cfdcdcb667a9db99a77fcd88ab`；
- R3/R4/R5构成连续三轮技术release探索，第四轮前必须完成项目目标、会话索引、完整日志、协议边界和下一候选的正式回顾。

## 9.R7发布门

|门|当前状态|
|---|---|
|统一代码/负测|216 passed|
|正式执行闭包|54文件；commit`9f608e8b`精确archive内54/54 SHA一致；D105可执行旧NumPy桥AST=0|
|精确Git archive|SHA256=`dd85491e…9a67`；4754项；单根、无链接/逃逸/重复；canonical loader、54 pyc、9帮助面PASS|
|真实checkpoint无query smoke|精确archive=`fdea3e39…9a07`；400步、K1恒等、query fit/update=0/0、Target/performance=false|
|candidate runtime/method lock|`8940e05f…85425`/`f36a0c6c…4864e`；archive canonical loader通过|
|可信外部authority签名|代码闭合；尚未生成生产signature|
|签名D102 revocation|生产内容撤销manifest/signature已本地生成并用固定公钥验签；私钥未进入工作树|
|R4独立release复审|已被N607预启动Git archive字节不一致P0作废|
|R5独立release复审|最终`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`；receipt SHA256=`65f8f211c…2c822`|
|R7 NumPy/Torch边界修复复审|`GO / P0=0 / P1=0 / P2=2`；receipt SHA256=`558053a1…fab842`|
|本地Git提交|LF/manifest/test/report修复=`46a65b3af2621d23bcc0a34631f45c8be17af4dd`；R5 review/docs源码archive提交=`d23469ba54afe00c284aa9b78b025def2b22fc43`|
|R4预登记提交|`03e3ff67003f16b6c39596a521f5bfdf0401850c`；包含独立launcher及运行边界|
|N607 R3 landing|失败于预启动哈希门；无detach、无性能；run永久封存|
|N607 R4预登记|`d105_phase1_sourceheld_d23469ba_20260731_r4`；新run root；源码只取`d23469ba`精确archive；尚未落地|
|N607 R4终态|唯一detach=1；exit=1；首个tap-cache在零prediction前触发PyTorch/NumPy对象边界异常；全部正式artifact为0；`NO_PERFORMANCE_RESULT`|
|N607 R5预登记|`d105_phase1_sourceheld_9f608e8b_20260731_r5`；全新run root；尚未落地|
|R5预登记提交|`27fccbfc1d49599a4c9e5e82d301780b02fbad37`|
|N607 R5终态|唯一detach=1；exit=2；首个tap-cache严格输出合同失败；全部正式artifact为0；`NO_PERFORMANCE_RESULT`|

生产私钥不得进入Git、报告、N607或formal asset。若无法获得与固定公钥匹配的独立签名，必须保持`NO_TARGET_LAUNCH`，不能退回unsigned JSON。

54文件闭包覆盖正式Phase1/Target25执行面、5个CLI、最小CVSincNet模型构造以及真实feature-smoke实际可达依赖；任何新增本地可执行依赖都必须先扩展清单并重新冻结。

R4的两个P2为：`model.py`仍产生`torch.cuda.amp.autocast`弃用警告；旧PyTorch缺少`safe_globals`时保留仅对精确SHA绑定checkpoint开放的兼容反序列化分支。两者不改变当前本地结果；唯一N607 runner必须在预检中记录PyTorch版本和实际checkpoint加载策略。
