# D105 Phase1 source-only压缩知识与source-held门实验报告

状态：`PHASE1_R7_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / D105-FTU3_ARCHIVE_VERIFIED / R8_LOCAL_PREREGISTERING / NO_PERFORMANCE_RESULT`

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

## 10.R3/R4/R5三轮正式回顾

回顾已完成并记录于`analysis/d105_r3_r5_release_retrospective_20260731.md`。项目会话索引已刷新为1181条，并检索D105三轮、当前目标、D62/D91/D92/SVRN和Target25历史；同时重新读取2026-07-20版`项目.md`及R3/R4/R5完整handoff和pipeline日志。

真实checkpoint最小复现确认：Phase1当前`_strict_forward`实际调用GRB旧tap，该路径只前向`id_backbone`，得到`z_id/pre_relu=[2,160]`且字节绑定通过，但`z_dom=None`；同一模型、同一IQ调用D105专用`extract_d105_feature_tap`时得到`z_dom=[2,160]`。复现JSON为`analysis/d105_r5_strict_tap_real_checkpoint_reproduction_20260731.json`。R5根因因此闭合为“正式Phase1接错特征出口”，不是数据、CUDA、数值或方法性能失败。

下一候选冻结为`D105-FTU1`：统一Phase1与D105正式双backbone tap，保持`z_id/pre_relu`字节绑定和`dom_backbone.feat_imp→dom_enhancer`域特征来源；增加真实checkpoint入口级Phase1 strict-tap/export回归和字段级fail-closed诊断。不得删除`z_dom`、伪造域特征、只改报错后release、远端修补R5或跳过Phase1启动Target25。

协议复核保持：LEO弱观测唯一、Phase2无clean/source、query零fit/update且逐样本全注册类决策、无role/quota/global reassignment、无类ID专属规则。后续Target25必须同时报告同row before/after旧类、`seen_new_acc`、`H_old_new`、逐旧类准确率、floor和forgetting；D105当前仍无任何性能结果。第四次release需等`D105-FTU1`实现、真实checkpoint本地闭环、独立`P0=0/P1=0`审查、Git提交和新run预登记全部完成。

## 11.D105-FTU1本地实现闭环

`D105-FTU1`已在提交`a0bdbba6`实现并提交。Phase1 `_strict_forward`现在唯一调用D105专用同IQ双backbone tap，`z_dom`严格来自`dom_backbone.feat_imp→dom_enhancer`；不存在GRB旧tap运行时导入或identity-only fallback。正式export将hook标志、`z_id/z_dom/pre_relu`的dtype、形状、有限性、ReLU绑定和execution path拆为字段级fail-closed门。

|本地门|结果|
|---|---|
|checkpoint形状正向入口/export|通过；旧GRB helper被置为必失败但调用数仍为0；一行IQ strict-tap archive与直接tap一致|
|字段破坏负测|7/7在创建输出目录前拒绝|
|SHA钉定真实checkpoint|195 tensors；`z_id/pre_relu/z_dom=[2,160]`；ReLU parity、finite和hook exact均通过|
|fresh进程旧GRB导入|调用前=false；调用后=false|
|统一回归|10文件223/223通过|
|54文件runtime|54/54哈希与内存编译通过|
|candidate runtime/method|`873879aad707fd2407b7645de45daa68fec1d3537feaf9fd57fe98b3ab059214`/`7d33662750b160fce82217dace9e1933aa8e43ea2a0df19f59e28adcf8bb4848`|
|R10独立审查|`LOCAL_CODE_REVIEW_GO / P0=0 / P1=0 / P2=2`；收据SHA256=`31ebec822064b4db7a3e5f4d419ee0ce8c4a493bb454ef1c0629c265164b8831`|

本地实现闭环不授权N607。下一门是从提交`a0bdbba6`生成精确Git archive，并在其解包副本中复跑54文件canonical loader/编译、帮助面、SHA钉定真实checkpoint Phase1 strict-forward/export no-truth smoke及旧路径不可达检查。archive smoke和新run预登记完成前不得同步或启动。

R4的两个P2为：`model.py`仍产生`torch.cuda.amp.autocast`弃用警告；旧PyTorch缺少`safe_globals`时保留仅对精确SHA绑定checkpoint开放的兼容反序列化分支。两者不改变当前本地结果；唯一N607 runner必须在预检中记录PyTorch版本和实际checkpoint加载策略。

## 12.D105-FTU1精确归档与R6预登记

提交`a0bdbba6bfb56c45682e0c2bde95aa622a68f101`的精确Git archive已完成独立本地验证。archive SHA256=`99fd633c78070b940064ca6e95ca9072427457058cab96c3a61e584c7991c0b4`，大小242913280B；4763个成员均在单一`source/`根下，无路径逃逸、链接、特殊成员或重复路径。54/54项满足`Git blob=archive=解包=runtime manifest`，LF和独立编译均54/54通过。

归档内真实checkpoint验证加载195个state tensor，`z_id/pre_relu/z_dom=[2,160]`，dtype、有限性、ReLU绑定和`eager_forward_hook`全部通过，fresh进程前后旧GRB模块均未导入。单行source-only export和receipt闭环通过，同时被既有最小34行formal聚合门正确拒绝；因此它只证明入口技术健康，不形成formal asset。CLI9/9、FTU1定向8/8、D105/LPO-RC回归223/223均exit0。总验证JSON SHA256=`78543dbb00d2ba3381d6e10b9808ebe751e8355d351261fa8c284cbe44c2ba30`，中文handoff SHA256=`95b4df18c212b959c49942e01c0fcd8a2484fa2387372bc69e4d47b12f3a9441`。

第四次release冻结为：

|字段|R6冻结值|
|---|---|
|run ID|`d105_phase1_sourceheld_a0bdbba6_20260731_r6`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_a0bdbba6_20260731_r6`|
|源码提交|`a0bdbba6bfb56c45682e0c2bde95aa622a68f101`|
|源码archive|SHA256=`99fd633c78070b940064ca6e95ca9072427457058cab96c3a61e584c7991c0b4`；242913280B|
|candidate runtime|`873879aad707fd2407b7645de45daa68fec1d3537feaf9fd57fe98b3ab059214`|
|candidate method lock|`7d33662750b160fce82217dace9e1933aa8e43ea2a0df19f59e28adcf8bb4848`|
|launcher|`run_d105_phase1_stage1_a0bdbba6.sh`；SHA256=`7f23f6e9bc8038a859962fed4b8fbb6ab63a805301ac9dbae7310a51be36e28d`；5624B；LF-only；`bash -n`通过|
|本地预登记提交|`814d3b1d51ce764e67c1125492886fb1a4f6b03e`|
|GPU|`cuda:0`；正式preflight后才可确认可用|
|fresh-run retry|`NO`；任何技术失败永久关闭R6并使用新run ID|

R6必须由一个专用terra-max实验子agent作为唯一launch owner执行。它重新完成direct N607 preflight、即时资源盘点、run root不存在、archive/54文件/pyc/checkpoint/launcher门和实际strict-forward预启动验证后，才允许唯一detach一次。R6不得读取或复用R3、R4、R5的source、input、PID、日志或output；不得远端修补、调参、因性能停止、访问Target或启动Target25。Phase1 stage1闭合后仍须回收完整component/score/gate，由独立审查确认source-held门，再由离线authority绑定N607 nonce ledger identity、run ID、commit和component签名，之后才可能封存formal asset。

## 13.R6 N607终态：reference dual parity技术失败

R6已由唯一runner启动一次后关闭为\`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT\`。run ID=\`d105_phase1_sourceheld_a0bdbba6_20260731_r6\`，主PID=\`2817802\`，首个\`tap-cache\`子PID=\`2817808\`，\`pipeline_stage1.exit=2\`。主PID的CWD、cmdline和\`CUDA_VISIBLE_DEVICES=0\`均绑定R6 run root与冻结launcher；没有第二次detach、重试、远端修补、调参、Target访问或Target25启动。

远端预启动全部通过：direct普通账户preflight、run root不存在、GPU0空闲、4个input SHA/大小、4763成员archive安全、54/54canonical runtime、source外pyc54/54、cache/salt/checkpoint/reference SHA、R2生产checkpoint strict-forward、9/9帮助面以及launcher LF/\`bash -n\`。R2终检记录195 tensors、torch=\`2.1.0+cu121\`、\`legacy_pickle_exact_frozen_sha_only\`、\`z_id/pre_relu/z_dom=[2,160]\`、float32、finite、ReLU绑定、\`eager_forward_hook\`和旧GRB未导入。

预检中旧\`verify_loader_and_real_checkpoint.py\`曾把1行source-only技术导出送入正式聚合读取，得到预期的\`strict tap feature rows drift\`；它不在冻结\`source/\`、R6 input或launcher/CLI入口，且已被R2终检替代。实际archive定义\`DOMAIN_DIM=32\`，\`StrictTapRows\`的最小聚合行数为\`DOMAIN_DIM+2=34\`；因此R2正确把1行拒绝记录为防误晋级边界，而不是运行时失败。

唯一正式启动写出8400行strict tap后，在reference dual byte-parity guard停止。日志指纹为\`strict D105 tap/reference dual archive parity failed\`。三项metadata完全一致，但\`z_id\`差=\`1.9073486328125e-05\`、\`z_dom\`差=\`0.0019412636756896973\`，均超过固定\`1e-5\`阈值。reference SHA=\`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0\`，R6 strict-tap SHA=\`68c08c85b2fdd7429444f9c9e92859f5e412076165d2e36d4fc634702ae1f5c6\`。这是确定性入口技术故障，不是性能或数据协议结论。

|候选/run|机制|strict tap|prediction|truth-open|score|gate|component|formal asset|性能结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|\`D105-FTU1/R6\`|Phase1双backbone strict tap＋reference dual parity guard|\`5\`文件、8400行|\`0\`|\`0\`|\`0\`|\`0\`|\`0\`|\`0\`|\`NO_PERFORMANCE_RESULT\`|

主/子进程均已自然退出，GPU0终态0%/1MiB且无compute process；本地SSH清理完成。完整回收交接见\`retrieved_d105_phase1_sourceheld_a0bdbba6_20260731_r6_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT/handoff.md\`，主日志SHA256=\`e11c6054150e9cf008890dfd61cca707b2b3afc16d356b8b288f01e01f0c1789\`。R6不得重试；下一轮必须先在本地闭合reference dual archive与当前生产tap的数值parity来源，重新独立审查、提交并使用新run ID。

## 14.R6固定批容量根因与D105-FTU2

全量8400×160差分已确认，reference dual以固定capacity=256生成，共33次调用；R6以batch=128直接前向，最后一次实际shape=80。`z_dom>1e-5`恰好只发生在最后80行，前8320行最大差仅`1.9073486328125e-06`；`z_id`唯一超过`1e-5`的行也位于该末批。根因是旧固定256补零TorchScript wrapper与新可变批eager路径的执行合同差异，不是checkpoint、输入顺序或D105-FTU1特征语义改变。

下一修复冻结为`D105-FTU2`：Phase1 tap固定以256容量执行，不足256行的末批零填充后前向并切回真实行；receipt增加capacity、调用数、末批真实/填充行；`tap-cache`拒绝非256；257行和8400行边界及批形状敏感fake model进入回归。参考archive、checkpoint、模型、feature tap和全8400行`max_abs<=1e-5`门均不改变。详见`analysis/d105_r6_batch_contract_root_cause_20260731.md`。

## 15.D105-FTU2本地实现与独立审查

FTU2现已完成固定256分块、末批零填充/切回、receipt v2绑定和`tap-cache`严格早门。257行形成2次256前向且末批1＋255；8400行形成33次256前向且末批208＋48。非法`128`、`256.0`、`np.int64(256)`和`True`均在任何外部路径访问前拒绝。真实SHA绑定checkpoint在1/208/256真实行下，相对独立256零填充reference的`z_id/pre_relu/z_dom`最大差≤`1.91e-6`。

统一10文件回归238/238通过。更新后的54文件canonical runtime SHA256=`8797de12f035db609aeb6f453f096571f216d0d514d6705344e763f5ec63a498`，method lock SHA256=`9a87e51de4d775ff2ea05e59654afaa62844edaf2def942d8f73c8e289ea61e6`。独立R11最终裁决为`GO / P0=0 / P1=0 / P2=0`；首轮1个P1和3个P2均已关闭。详见`analysis/d105_ftu2_implementation_validation_20260731.md`与`analysis/d105_ftu2_review_r11_20260731.md`。

本地实现提交=`2d948ce981b9008522f825cfe6d868bce08cb624`。该提交不授权N607。下一门是从该提交生成精确archive，在解包副本中复跑54/54、CLI、真实checkpoint固定256 tap和完整8400行reference parity技术smoke；通过后才能创建新run ID和唯一runner。

## 16.D105-FTU2精确archive与R7预登记

提交`2d948ce981b9008522f825cfe6d868bce08cb624`的精确archive已独立验证通过。archive SHA256=`e58240a0a358893c0c90ce0b3cb9c202eed9e6907272fa0d587d160f3fb8ec23`，242964480B、4770成员；成员安全、54/54四方SHA、LF、独立pyc、canonical runtime/method、9/9帮助面和238/238回归全部通过。真实checkpoint在1/208/256真实行下相对独立256零填充reference的三路max_abs=0，8400批合同闭合为33次forward、末批208＋48。

R7冻结为：

|字段|冻结值|
|---|---|
|run ID|`d105_phase1_sourceheld_2d948ce9_20260731_r7`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_2d948ce9_20260731_r7`|
|source commit|`2d948ce981b9008522f825cfe6d868bce08cb624`|
|R7预登记提交|`632fd9f0e1324d14cb9d489b92b71259e9ac29fe`|
|archive|SHA256=`e58240a0a358893c0c90ce0b3cb9c202eed9e6907272fa0d587d160f3fb8ec23`；242964480B|
|runtime/method|`8797de12f035db609aeb6f453f096571f216d0d514d6705344e763f5ec63a498`/`9a87e51de4d775ff2ea05e59654afaa62844edaf2def942d8f73c8e289ea61e6`|
|launcher|`run_d105_phase1_stage1_2d948ce9.sh`；SHA256=`95081f1e20aabc7f89a970b667bae223926949dde26edd0a0e660acd8157406a`；5624B；LF=123；CRLF=0；`bash -n`通过|
|GPU|`cuda:0`；正式preflight后才可确认|
|fresh-run retry|`NO`|

R7必须使用全新root和唯一terra-max runner，不得读取或复用R3—R6的source、input、output、PID或日志。首次正式tap必须对全部8400行通过reference parity；否则R7技术失败并永久关闭。只有tap、prediction、truth-open、score、gate和component完整闭合后，才进入独立component审查；Target25仍禁止启动。

与R6启动脚本逐行比较只存在两个冻结差异：run root从R6改为上述R7全新root，tap-cache的`--batch-size`从128改为256。其余命令、参考archive、门阈值、输入和阶段顺序不变。

## 17.R7 N607终态：derive-gate整数门技术失败

R7由唯一runner按冻结方案detach一次后关闭为STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT。run ID=d105_phase1_sourceheld_2d948ce9_20260731_r7；source commit=2d948ce981b9008522f825cfe6d868bce08cb624；R7预登记commit=632fd9f0e1324d14cb9d489b92b71259e9ac29fe；证据绑定HEAD=20ec6291436213c66daace27f6f0b9572c25e6fc。main PID=2857134，CWD、cmdline、CUDA_VISIBLE_DEVICES=0和launcher均由launch_receipt绑定到R7新root；进程自然退出，pipeline_stage1.exit=2。

预启动门全部通过：direct普通账户preflight、GPU0空闲、run root初始不存在、archive/launcher/revocation四输入SHA与大小、4770成员安全解包、54/54 Git blob=archive=extract=manifest、54/54 LF与隔离pyc、9/9 CLI help、外部cache/salt/checkpoint/reference SHA，以及真实checkpoint的生产bridge fixed256小型合同smoke。后者在1/208/256行均以固定256容量运行，对独立bytearray/frombuffer reference的三路max_abs均为0，195 tensors、eval、state不变和旧GRB未导入均通过；它是source-only技术预检，不产生formal asset或性能结论。旧外部helper的直接torch.from_numpy在N607 NumPy/Torch配对下失败，已标为非生产bridge预检缺陷，未改变冻结source或launcher。

唯一正式tap-cache已完成8400行reference parity：fixed_256_zero_pad_then_slice_v1，forward=33，末批=208真实行+48填充行，reference z_id/z_dom max_abs均为0；strict tap为source-only，target_rows=0、query_rows=0、raw/clean IQ均未保留。随后prediction、truth-open、score各生成一份不可变artifact；derive-gate尚未创建输出即以source-held derived gate integer drift退出，component、authority、seal、formal asset和Target25均未执行。

|候选/run|机制|strict tap|prediction/truth/score|gate/component|Target25|结论|
|---|---|---|---|---|---|---|
|D105-FTU2/R7|Phase1 fixed256双backbone strict tap与source-held闭环|8400行，33次forward，reference parity通过|1/1/1份已回收|0/0|未启动|NO_PERFORMANCE_RESULT|

失败门精确复核仅输出结构与类型，不披露性能值：整数guard共9个字段，类型漂移=0；receiver_held_min_net_correct和class_loco_min_net_correct两项为合法负整数，却被该validator的非负计数条件拒绝，failed_integer_guard_field_count=2。immutable row计数为scored prediction=63、truth score=63、tx probe=7；target_rows=query_rows=0。该失败是gate validator语义与min_net_correct允许负值之间的确定性技术不一致，不是性能结论。

完整证据已回收至retrieved_d105_phase1_sourceheld_2d948ce9_20260731_r7_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT/artifacts。主日志SHA256=8680bf8aef479e59cc4ec7b3dc8e7588c371a0339c0d8a25a0af256a9b7ff5be；strict-tap SHA256=6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578；receipt SHA256=27ee9275b3b89ef78c4ed61349d87f9491274efacd12e6028002aa9a6faed67f。回收副本与远端21项哈希逐项一致，GPU0终态0%/1MiB且无compute process，本地SSH/SCP已清理。

R7不得重试、恢复、覆盖或重新解释为性能实验。后续只能先在本地修复并测试min_net_correct的合法整数域与gate validator语义，完成独立审查和新Git提交后，以新的不可覆盖run ID重新申请release；在完整Phase1 component、独立审查、离线authority和formal seal之前，Target25继续保持未启动。

## 18.D105-FTU3本地修复与审查

R7回收artifact的结构/类型复算确认，7个普通计数字段均为原生非负`int`，只有`receiver_held_min_net_correct`和`class_loco_min_net_correct`是合法负`int`。`D105-FTU3`只把这两个字段从非负计数组拆出并要求原生`int`；其他计数、schema、gate计算、阈值、方法和生命周期不变。

新增15项回归覆盖负原生整数放行、`bool/float/np.int64`拒绝、7个普通计数负数拒绝，以及负证据组件保持`DIAGNOSTIC_STATUS`并被formal seal拒绝。统一10文件共253项执行到100%，无失败或错误；canonical runtime/method loader通过54/54成员。新runtime SHA256=`5de5926bbb2e9fd78b2f3315ec6e109964ddd6216ebe4f75e428b6b9f6bf11bc`，method lock SHA256=`7345f81e88588c46ad453eb315786306f28291478a5eaddce618ef7ee6998ecd`。独立增量复审已实跑组件/封印拒绝case并核对54文件身份，最终结论=`GO / P0=0 / P1=0 / P2=0`。

FTU3实现提交=`230c6cbc9149250ca0303ca240945d0e0992360e`。它仍只有本地技术证据，不改变R7的`NO_PERFORMANCE_RESULT`。下一门是该提交的精确archive复核；通过前不创建新N607 run、不签名、不封装formal asset、不启动Target25。详见`analysis/d105_ftu3_gate_signed_int_fix_20260731.md`。

## 19.D105-FTU3精确archive与R8预登记

实现提交`230c6cbc9149250ca0303ca240945d0e0992360e`的精确archive已独立验证通过：SHA256=`16d57519cfa15d9929a38282217b0a2e2908e5c92e8b42672dae1537386855c7`，243005440B。archive成员安全、54/54四方SHA、LF、独立编译、canonical runtime/method、9/9帮助面、15项FTU3定向、10文件253项统一回归、真实checkpoint fixed256三档和8400批形状合同全部通过。独立archive审查=`PASS / P0=0 / P1=0 / P2=0`。

R8冻结为：

|字段|冻结值|
|---|---|
|run ID|`d105_phase1_sourceheld_230c6cbc_20260801_r8`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_230c6cbc_20260801_r8`|
|source commit|`230c6cbc9149250ca0303ca240945d0e0992360e`|
|archive|SHA256=`16d57519cfa15d9929a38282217b0a2e2908e5c92e8b42672dae1537386855c7`；243005440B|
|runtime/method|`5de5926bbb2e9fd78b2f3315ec6e109964ddd6216ebe4f75e428b6b9f6bf11bc`/`7345f81e88588c46ad453eb315786306f28291478a5eaddce618ef7ee6998ecd`|
|launcher|`run_d105_phase1_stage1_230c6cbc.sh`；SHA256=`db0757789fa4b3a4155e793c28a2d7c76926248b59cd51c3758cf93364a3cdc9`；5624B；LF=123；CRLF=0；`bash -n`通过|
|GPU|`cuda:0`；正式preflight后确认|
|fresh-run retry|`NO`|

R8启动脚本相对R7只有全新run root一个差异；输入、固定256合同、reference、阈值和阶段顺序不变。R8不得读取或复用R3—R7任何run内容；完整8400行reference parity仍是首tap硬门。只有gate和component实际落盘后才能读取其正式状态；若component为诊断拒绝，则不得签名、封装或进入Target25。
