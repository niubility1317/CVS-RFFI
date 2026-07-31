# D105 R3独立release审查收据

> 已作废：后续R4审查发现tap-cache、query evaluator和模型工厂存在清单外动态执行路径。本收据对应的45文件闭包与`LOCAL_RELEASE_GO`不得用于N607发布或性能声明。

状态：`LOCAL_RELEASE_GO（同版本本地实现）`

审查日期：2026-07-31
审查人：独立release reviewer
审查对象：`D105-CBRC+LPO-RC-qKNN`R2冻结工作树
协议：`p2_min_v1`

本收据只签核本地实现、冻结配置和本地技术证据。GO在主agent将本收据、代码、配置、报告镜像和smoke镜像以同一文件集合完成本地Git提交后生效。提交前保持`NOT_LANDED`；本收据不授权N607落地、Phase1 formal asset、Target25性能分析或`PROMOTABLE`声明。

## 1.最终结论

|等级|数量|结论|
|---|---:|---|
|P0|0|未发现协议、信任、数据权限、query泄漏、覆盖闭合或执行入口绕过缺陷。|
|P1|0|未发现阻止同版本本地release的实现、测试、散列、报告镜像或可追溯性缺口。版本化提交是本GO的生效动作，尚未授权服务器操作。|
|P2|0|未发现需要延后本地release的非阻塞实现问题。|

结论为`LOCAL_RELEASE_GO`。它只说明D105 R2可以进入“本地提交后交由唯一server runner预检”的状态；不等于Phase1 source-held门已经通过，也不等于D105在Target25上产生了任何性能结果。

## 2.冻结身份与规范加载

|对象|独立复核结果|
|---|---|
|candidate runtime manifest|`639c16dd6a70620ca99fa960acb9e988aeba3cea92edcb7a9a158b26a6d958b5`|
|candidate method lock|`37dd03fcdb7cb01e6e545def11711b0c9c9ad35e3d505d75c18f314cb3ef3576`|
|authority模块文件|`stage2_d105_phase1_authority.py`SHA256=`fe81a728bcd8e1047a40069b9d9954aed2af1c89b98633489ccf2b922b4364bd`|
|checkpoint|`best_joint_safe_ssdg.pth`SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|canonical loader|runtime和method lock均成功加载；45个成员的当前散列与manifest逐项一致。|
|Target25入口闭包|45个成员重新计算得到的映射与runtime loader观测映射完全一致。|

method lock明确固定：CBRC为4轮IRLS、old/new任务各0.5、K1零系数和FP16部署；Student-t qKNN为`nu=3.0`、`p_eff=12`、`gamma=1.0`、`h0=0.35`、scale prior=2.0、比例范围[0.5,2.0]、temperature=0.85、`int8_fp16_scale`；Target25为25个outer row、300个scenario-arm pair、600个state prediction surface，且声明永久为`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`与`formal_launch_authority=false`。

`stage2_d105_target25_inputs.py`和`stage2_d105_query_evaluation.py`都把D92索引内的qKNN lock逐字段对照上述候选锁。手工改写Target25 index内的`qknn_lock`、运行时锁或候选锁都会在prepare或预测前失败，不能把开发态锁改成formal声明。

## 3.运行闭包、测试和入口验证

正式执行闭包包含40个递归可达的`cvsrffi`模块和5个CLI，共45个文件。AST测试检查集合精确性；逐项删除或内容漂移均应导致身份验证失败。45个文件均已在`ssr-gpu`中通过`py_compile`。

以下10个D105测试文件实际运行通过，共182项：

|测试面|项数|
|---|---:|
|CBRC|16|
|LPO-RC-qKNN|18|
|四臂|6|
|feature tap|2|
|Phase1 authority|5|
|Phase1 bundle与运行闭包|99|
|query evaluator|15|
|Target25 inputs|6|
|Target25 launcher|5|
|Target25 runner|10|

5个正式CLI的`--help`均退出0：`build_d105_phase1_bundle.py`、`sign_d105_phase1_authority.py`、`run_d105_four_arm_real_feature_smoke.py`、`prepare_d105_target25_inputs.py`和`run_d105_target25.py`。`sign-authority`、`sign-target25-prepare`、`predict`和`score`四个关键子命令参数面也均退出0。

45文件闭包覆盖正式Phase1/Target25执行面和5个入口。真实checkpoint smoke脚本本体在清单内，但其训练专用helper不属于正式Target25预测依赖；因此45文件不被表述为覆盖全部smoke传递依赖。这是已记录的范围边界，不影响正式预测执行面。

## 4.信任、撤销和唯一prepare复核

Phase1 formal asset使用固定`somph_runtime_trust`Ed25519信任根和独立签名域。authority envelope绑定组件、checkpoint、runtime/method lock、独立review、D102撤销manifest、Git commit、有效时间窗、run ID、nonce和预先创建的账本identity。`compute_d105_nonce_ledger_identity`以本机解析后的账本绝对路径、run ID和签名域计算身份；消费端复算并比对后，才以排他创建标记消费nonce。

Target25使用独立`TARGET25_PREPARE`签名域。预测和评分CLI在加载plan前重算并验签matrix index、prepare receipt、plan、context、Git commit、run ID、候选runtime/method lock和账本identity。非dry-run预测在打开执行面前消费nonce；dry-run和score仅验签、不重复消费。测试覆盖了错误签名域、错误签名键、过期、绑定篡改、formal重标、替换账本路径和nonce重放。

包加载路径要求signed path-free authorization、package root、detached seal、authority commit、dataset authority root及receiver/seed/stage/registration/K一致。四个包的authority commit必须唯一且等于六个split authority receipt；共享的v2 prediction-context builder同时被prepare和真实evaluator调用。这样不会再出现手写plan/context、包与authority平行验证或prepare/evaluator上下文散列不一致的路径。

D102r6真实内容身份已逐项核对并在签名撤销逻辑的fixture中使用：manifest=`0690f2ab19560a54c96599ffc59a56fd31786f48ac2f05659414d8c29ff0da64`、payload=`440ff82a1f74b67078f699eaca86e85b9739d574721ccfb460a423ff97cc93d4`、seal=`cdcfceb5a31e3409ccea137fe116347f2214640e6514b080d442e7a193a0db59`、content root=`16b9a8388c612509e4b220f2883fcd92187e1de0e4236ef25e2ef72a472a48b7`、checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`、method=`9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f`、runtime=`e1b21bee74941dfb550b67698a75f485937bc39431ed7859baaa20d44a4899f3`、held=`01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2`、tap=`c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1`。改名副本不能依靠文件名绕过拒绝。

生产私钥、生产D105 formal authority signature和生产D102 revocation signature均尚未生成；这不是本地实现缺陷，而是后续Phase1外部authority和服务器阶段的必要前提。缺失时必须保持`NO_TARGET_LAUNCH`。

## 5.同代真实checkpoint smoke与报告镜像

同代无truth smoke凭据为`E:\type10-7\automation_reports\CV-SincNet\d105_cbrc_lporc_local_smoke_20260731_r3\real_feature_no_truth_smoke.json`，文件SHA256=`cc08c4891b8c9112fc37dc9c752f7f53f99e4a3b83df22195f3f58e48696ef5f`。其状态为`DEVELOPMENT_ONLY_REAL_CHECKPOINT_DERIVED_NO_TRUTH_SMOKE_PASS`，checkpoint字节已核验，完成400个source-held meta step，K1的`M_HEAD=M0`和`M_JOINT=M_DA`均逐值成立；query fit/update=0/0，query truth和query标签均未传给predictor，state receipt不变，Target访问=false，性能计算=false，formal Phase2 evidence=false。

根报告与snapshot镜像的字节散列一致：

|报告|根报告SHA256|snapshot镜像SHA256|
|---|---|---|
|D105 Target25 R2报告|`827118c9d45fe2845be75be6ed1aece88bf99836d4bc4b2e0492fa563e5d7eb1`|`827118c9d45fe2845be75be6ed1aece88bf99836d4bc4b2e0492fa563e5d7eb1`|
|D105 Phase1 source-held报告|`32cbef4dba14f176546b19a343e503686f811871d95eb26f04ce962e95f77115`|`32cbef4dba14f176546b19a343e503686f811871d95eb26f04ce962e95f77115`|

两份报告、设计冻结和目标文档均已更新为45文件、R2修复、签名prepare、nonce ledger、182项回归和本地证据边界；它们不报告D105 Target性能。

## 6.版本管理与下一步

`git diff --check`无空白错误；仅出现两份已跟踪Markdown在Git触碰时可能由LF转换为CRLF的提示，不构成diff检查失败。本收据创建时工作树仍待主agent统一stage；主agent应将本收据、报告与smoke镜像以`git add -f`纳入同一笔本地提交，并在提交后记录commit SHA。不得push或上传。

提交完成后，唯一server runner才可按既定流程重新N607预检、精确同步和hash/compile检查。此后仍必须先完成真实D105 Phase1 source-held预测、独立truth-open score、gate、外部签名和formal seal；只有这条资产链通过，才允许执行完整25job/300pair/600state的Target25开发screen。任何后续性能结论必须来自完整、同row、独立评分的artifact，不能由本收据、smoke或局部测试推断。
