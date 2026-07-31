# D105-FTU4 TX负结果诊断回执实现

## 1.问题与目标

R8在`tx_probe_gate_pass=false`后仍进入通用`RXIDMetaBias4Bundle`构造与序列化。底层bundle按既有合同拒绝failed-TX wire，外层却把该合法负结果包装为“quantization closure后无法序列化”的技术错误，导致gate已经闭合但component没有落盘。R8永久保持`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / ARTIFACTS_INCOMPLETE / NO_PERFORMANCE_RESULT`；本轮不复用、恢复或重标R8。

`D105-FTU4`只修复这一生命周期缺口：当且仅当合法derived gate的`tx_probe_gate_pass is not True`时，构建器在聚合与通用RXID wire之前输出独立、无wire、不可部署的诊断回执。TX通过的component/formal路径不变；TX通过但receiver/class门失败时，既有`D105_PHASE1_COMPONENT + DIAGNOSTIC_STATUS`路径也不变。

## 2.实现合同

新artifact使用schema=`cvs.phase1.d105.cbrc.tx_probe_diagnostic_receipt.v1`、kind=`D105_PHASE1_TX_PROBE_DIAGNOSTIC_RECEIPT`、status=`DIAGNOSTIC_STATUS`，并固定`formal_phase2_eligible=false`、`deployable_wire_present=false`。目录精确allowlist只有5项：

1. `d105_phase1_tx_probe_diagnostic.manifest.json`；
2. `d105_phase1_tx_probe_diagnostic.manifest.sha256`；
3. `d105_source_held_gate.json`；
4. 签名D102 revocation manifest；
5. D102 revocation detached signature。

manifest只保留candidate/protocol身份、checkpoint/runtime/method/candidate lock、strict-tap SHA、source-held gate SHA与aggregate-safe summary、formal missing、D102 manifest/signature SHA绑定，以及`source-only/target_rows=0/query_rows=0`和无raw/clean/source row/handle/ID的fail-closed标志。它不包含wire、receiver aggregate、量化数组、源行、receiver/class/physical句柄或可部署bundle identity。

专用`load_d105_phase1_diagnostic_receipt`验证目录与成员非符号链接、精确allowlist、manifest canonical字节、SHA256 seal、精确字段集合、candidate/runtime/method绑定、source-held gate canonical字节及摘要、`tx_probe_gate_pass=false`、TX missing token、签名D102 revocation和全部不可部署标志。返回值为冻结的`D105Phase1DiagnosticReceipt`；其manifest与gate均为只读映射，不含`RXIDMetaBias4Bundle`。

普通`load_d105_phase1_asset`、`validate_d105_phase1_asset`、runtime handle和formal seal均拒绝该目录。seal在读取authority envelope、signature、independent review、消费nonce或创建formal输出前检查诊断manifest并退出。二次bundle rebuild与serialize异常现已拆分：真实量化/类型重建故障仍报告技术错误，真实serialize故障使用独立文案，二者都不会生成诊断回执。

## 3.测试闭环

新E2E从不可变synthetic prediction、truth-open和score证据构造TX失败。测试把`_build_aggregate_parameters`替换为必失败断言，构建仍成功输出5成员诊断目录，证明负结果分支发生在聚合之前。覆盖面如下：

|验证面|结果|
|---|---|
|无wire与无聚合|通过；`deployable_wire_present=false`，目录无`d105_phase1_aggregate.wire`，聚合函数调用数为0|
|专用loader|通过；canonical、seal、gate、D102、allowlist、只读映射均闭合|
|普通/formal/runtime拒绝|普通loader、formal validate、runtime handle和seal全部拒绝|
|nonce与输出边界|seal拒绝前后nonce ledger成员及SHA不变，formal输出不存在|
|篡改负测|伪装kind、manifest字节篡改、塞wire、删除TX missing token、翻status、非法allowlist、gate篡改、D102签名篡改全部拒绝|
|成员symlink负测|5个diagnostic成员逐一替换为真实文件symlink，专用loader全部拒绝；本机无skip|
|canonical重封绑定负测|修改gate summary、checkpoint/runtime/method/strict-tap SHA及D102 manifest/signature SHA后重算canonical manifest与seal，7/7全部拒绝|
|非覆盖|对同一输出目录二次build拒绝|
|错误分类|第二次bundle build故障与serialize故障使用不同异常文案，均无诊断输出|
|FTU4定向|23项执行到100%，全部通过|
|Phase1 bundle整文件|187项执行到100%，全部通过|
|统一回归|固定10个D105/LPO-RC测试文件共276项，执行到100%，全部通过|
|runtime闭包|54/54 core SHA一致，54/54文件内存编译通过|

统一回归唯一警告是既有`code/model.py:693`的`torch.cuda.amp.autocast`弃用警告。本轮未修改`rxid_metabias4_bundle.py`及其failed-TX不可serialize合同，也未修改CLI行为。

## 4.独立复核P2关闭

独立复核首轮结论为`GO / P0=0 / P1=0 / P2=2`。P2-A要求逐成员symlink拒绝证据；新增5个参数实例，覆盖diagnostic manifest、seal、source-held gate、D102 manifest和D102 signature。Windows主机成功创建并验证5个真实symlink，因此没有触发权限skip，专用loader逐项以symbolic-link门拒绝。

P2-B要求证明“攻击者修改manifest并重算合法canonical seal”仍不能绕过语义绑定。新增7个参数实例分别修改`source_held_gate_summary`、checkpoint/runtime/method/strict-tap SHA、D102 manifest SHA和D102 signature SHA；每个case均先验证修改后的manifest仍为canonical字节且seal与新manifest SHA一致，再确认专用loader拒绝。两项P2均由实际测试关闭，最终复核状态为`GO / P0=0 / P1=0 / P2=0`。

## 5.canonical身份

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d105_phase1_bundle.py`|`91931cb3893cb902a7eef1e509d209b232d2769225b012c9a0027c978a3ced39`|
|`tests/test_stage2_d105_phase1_bundle.py`|`9a677ff0a045a6816f469fc3d4c0e55cedb672cefc4e6bf1e6fb1fa7da0e0253`|
|`configs/d105_candidate_runtime_manifest_20260731.json`|`9b1887e64851851be8a81118a3b3728cd94517de6c9ae275f8574764cb30c38e`|
|`configs/d105_candidate_method_lock_20260731.json`|`7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08`|

runtime manifest相对FTU3只改变`cvsrffi/stage2_d105_phase1_bundle.py`一个core hash；其他53个core hash、entrypoints、checkpoint、candidate和protocol均不变。method lock只更新runtime manifest绑定，方法参数与阈值逐字段不变。

## 6.状态与边界

本地状态为`LOCAL_REVIEW_GO / P0=0 / P1=0 / P2=0`。冻结设计无偏差；实现没有扩大到性能计算、target/query访问、formal seal创建、authority签名或Stage2运行。`E:\type10-7`根目录不是Git仓库；本文件的Git承载版本位于当前工作树，根目录`analysis`仅保存同步镜像，尚未形成独立版本历史。

本轮`NOT_N607_AUTHORIZATION / NO_PERFORMANCE_RESULT`，且按任务要求尚未commit或push。下一门是主agent独立审查当前diff；只有审查关闭P0/P1并由主agent决定提交后，才能讨论精确archive或新的非覆盖run ID。
