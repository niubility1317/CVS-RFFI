# D105 Phase1 R7 N607终态交接

状态：STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT
运行ID：d105_phase1_sourceheld_2d948ce9_20260731_r7
唯一runner：terra-max
fresh-run retry：NO
Target25/authority/formal seal：未执行

## 冻结身份

|字段|值|
|---|---|
|source commit|2d948ce981b9008522f825cfe6d868bce08cb624|
|R7预登记commit|632fd9f0e1324d14cb9d489b92b71259e9ac29fe|
|证据绑定HEAD|20ec6291436213c66daace27f6f0b9572c25e6fc|
|source archive|source_2d948ce9.tar；242964480B；e58240a0a358893c0c90ce0b3cb9c202eed9e6907272fa0d587d160f3fb8ec23|
|runtime/method|8797de12f035db609aeb6f453f096571f216d0d514d6705344e763f5ec63a498 / 9a87e51de4d775ff2ea05e59654afaa62844edaf2def942d8f73c8e289ea61e6|
|launcher|run_d105_phase1_stage1_2d948ce9.sh；95081f1e20aabc7f89a970b667bae223926949dde26edd0a0e660acd8157406a|
|remote run root|/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_2d948ce9_20260731_r7|

## 发布与预启动

- direct普通N607账户preflight通过；8张RTX3090均空闲，GPU0用于本run，发布前后无其他compute process。
- 四个冻结输入的SHA、大小、launcher LF=123/CRLF=0和bash -n均通过。
- archive安全门为4770成员、单一source根、链接/特殊成员/重复/路径逃逸均为0。
- 54/54 Git blob=archive=extract=runtime manifest，54/54 LF、54/54隔离pyc，source pyc/cache=0；runtime/method绑定通过。
- 9/9 CLI help通过。
- 真实checkpoint生产bridge fixed256小型smoke通过：1/208/256行均固定256容量，三路reference差均为0，195 tensors、eval、state不变、旧GRB未导入；只使用合成source-only IQ，Target/query/performance/formal asset均为false。
- 旧外部直接torch.from_numpy helper在N607上触发expected np.ndarray(got numpy.ndarray)，因其绕过冻结生产frombuffer bridge而仅记为预检helper缺陷，不代表R7 launcher失败。

## 唯一detach与终态

main PID=2857134。launch_receipt绑定CWD、cmdline、CUDA_VISIBLE_DEVICES=0、launcher和R7 root。launcher自然退出，pipeline_stage1.exit=2；没有重试、远端修补、调参、Target访问、Target25、authority或formal seal。

正式tap-cache完成8400行：fixed_256_zero_pad_then_slice_v1，33次forward，末批208真实行加48填充行，reference z_id/z_dom max_abs均为0。strict tap为source-only，target_rows=query_rows=0，raw/clean/received IQ均未保留。

|阶段|已生成数量|状态|
|---|---:|---|
|strict tap|5文件，8400行|通过reference parity|
|prediction|1份，63个scored-row结构+7个tx-probe结构|已生成，不作性能结论|
|truth-open|1份|已生成，prediction后开启|
|score|1份，63个truth-row结构+7个tx-probe结构|已生成，不作性能结论|
|derive-gate|0|失败于整数guard|
|component/formal asset|0/0|未生成|

错误指纹：source-held derived gate integer drift。

精确门诊断仅保留字段名和计数：整数guard字段共9个，类型漂移=0；违反非负条件的字段数=2，字段为receiver_held_min_net_correct与class_loco_min_net_correct。它们是合法负整数，validator却把它们按非负计数处理。未输出任一性能值。此为确定性gate validator语义故障，不是性能或数据协议结论。

## 回收与完整性

回收目录：retrieved_d105_phase1_sourceheld_2d948ce9_20260731_r7_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT/artifacts。

|证据|SHA256|
|---|---|
|launch receipt|d37fbc93a422413c13317c421e41c2416722a3b298c978fd7130ec7da93b2ddd|
|pipeline log|8680bf8aef479e59cc4ec7b3dc8e7588c371a0339c0d8a25a0af256a9b7ff5be|
|pipeline exit|53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3|
|runtime preflight|79f6e88e3cadb57c5107d84a5e2e9530bb234c919b4807e283da0d0166a6f3c3|
|production bridge smoke|6380577a2c52e24b4fe5a0ee2af2f60afaeba6fd9bdd6ab6c526c9318c3772ec|
|strict tap archive|6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578|
|strict tap receipt|27ee9275b3b89ef78c4ed61349d87f9491274efacd12e6028002aa9a6faed67f|
|prediction manifest|4d196a727eeac32de7f66f934c6a903fd0c23df41892142e5d3295a55fd581cb|
|truth-open receipt|a800d13588f5622dcce3a1109912ee20e71e4121a69b943f7df3f311d499ed18|
|score artifact|c23ee793c2acb9106d0a4042e607e5cabaadb2d0f252637fed93ee94270844c5|

远端与本地21项选定证据SHA逐项一致。GPU0终态0%/1MiB，无R7进程或compute process。本地ssh.exe/scp.exe和N607/bridge TCP22均已清理。

## 后续门

R7永久关闭，不得恢复、覆盖、重试或把partial score解释为性能。下一轮必须先在本地修复并回归min_net_correct的合法整数域与gate validator语义，完成独立审查和新Git提交，然后使用新的不可覆盖run ID重新发布。仅当新的Phase1 component完整、独立审查、离线authority和formal seal全部成立时，Target25才可能进入预登记。
