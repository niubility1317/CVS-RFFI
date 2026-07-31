# D105-CBRC-MB4＋LPO-RC-qKNN实现追踪

## 1.状态

- 设计基线commit：`776dc6a4`
- 当前阶段：`LOCAL_IMPLEMENTED / LOCAL_TESTED / REAL_CHECKPOINT_DERIVED_FEATURE_SMOKE_PASS / INDEPENDENT_IMPLEMENTATION_REVIEW_GO`
- N607：未预检、未同步、未创建远端目录、未启动
- 性能证据：无
- D104：保持`PAUSED_BEFORE_LANDING / NO_PERFORMANCE_RESULT`

## 2.功能分工与实现

|部分|实现|功能边界|
|---|---|---|
|共享DA|`code/cvsrffi/stage2_d105_cbrc.py`|单一support-only四维CBRC系数；K1固定零系数；query只读变换|
|分类HEAD|`code/cvsrffi/stage2_lpo_rc_qknn.py`|K≥2物理LOO零和class bias；K1精确返回base logits|
|四臂集成|`code/cvsrffi/stage2_d105_four_arm.py`|M0、M_DA、M_HEAD、M_JOINT；不引入新估计器|
|真实特征smoke|`code/scripts/run_d105_four_arm_real_feature_smoke.py`|核验真实checkpoint字节和既有checkpoint派生tap/dual特征；source-held、无truth、无Target|

四臂归因固定如下：

|arm|表示|HEAD|
|---|---|---|
|M0|base canonical z_id|base Student-t qKNN|
|M_DA|同一D105 state变换|base Student-t qKNN|
|M_HEAD|base canonical z_id|LPO-RC-qKNN|
|M_JOINT|与M_DA同一D105 state|与M_HEAD同代码、同配置的LPO-RC-qKNN|

## 3.独立审查

|对象|首轮问题|修复|当前裁决|
|---|---|---|---|
|DA|bundle authority/receipt、query校验开销、资源闭合|新增自验证validator receipt、payload原位校验和完整资源账本|`P0=0/P1=0/P2=0，GO`|
|HEAD|递归不可变、state wire、workspace上界、rank1置换持久测试|补齐deep-copy/tamper、确定性workspace上界和rank1置换回归|`P0=0/P1=0/P2=0，GO`|
|四臂集成|整批root检查导致外部分块被拒绝|改为完整query root一次核验，入口内部按`chunk_size`分块|`P0=0/P1=0/P2=0，GO`|

DA审查GO只证明本地实现G0，不证明外部Phase1 validator签名权威或性能。HEAD审查GO不等于release GO。

## 4.本地验证

环境：`ssr-gpu`

覆盖：

- DA/HEAD/四臂模块及smoke脚本`py_compile`；
- 四臂完整批、3行分块、逆序2行分块逐值一致；
- K1满足`M_HEAD=M0`和`M_JOINT=M_DA`的数组及字节恒等；
- query root、state receipt、bundle payload、INT8 bank、HEAD bias篡改fail-closed；
- DA、HEAD、基础Student-t qKNN和RXIDMetaBias4 bundle相邻回归。

最终稳定版结果：10个指定D105测试文件共`182 passed，0 failed`；45文件正式执行闭包全部`py_compile`通过；5个正式CLI及`predict/score/sign-authority/sign-target25-prepare`关键参数面退出码均为0。

## 5.真实checkpoint派生特征无truth smoke

凭据：

`E:\type10-7\automation_reports\CV-SincNet\d105_cbrc_lporc_local_smoke_20260731_r3\real_feature_no_truth_smoke.json`

凭据SHA256：`cc08c4891b8c9112fc37dc9c752f7f53f99e4a3b83df22195f3f58e48696ef5f`

|项目|结果|
|---|---|
|checkpoint字节SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`，核验通过|
|tap archive SHA256|`c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1`|
|dual archive SHA256|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|Phase1训练|400 meta steps|
|source-held receiver|`1-1`|
|support/query|6类K1 support=6；无truth query=354|
|DA状态|`ACTIVE`|
|query fit/update|0/0|
|K1恒等|`M_HEAD=M0`、`M_JOINT=M_DA`，逐值和hash均相同|
|state receipt|计分前后不变|
|性能计算|false|
|Target访问|false|

本次smoke绑定冻结runtime SHA256=`639c16dd6a70620ca99fa960acb9e988aeba3cea92edcb7a9a158b26a6d958b5`和method lock SHA256=`37dd03fcdb7cb01e6e545def11711b0c9c9ad35e3d505d75c18f314cb3ef3576`，耗时121.047秒，stderr为0。它复用已封存的checkpoint派生真实特征，没有重新从原始IQ执行backbone前向；它是实际checkpoint字节＋真实派生特征的机械闭环证据，不是Target性能、正式Phase2 row或完整runner证据。45文件正式执行闭包包含该smoke入口本体，但不声称覆盖smoke专用训练helper的全部传递依赖；正式Target25预测不依赖这些helper。

## 6.历史性能定位

同口径历史对比见`analysis/stage2_baseline_comparison_20260731.md`。当前D105没有任何性能行，所有相对D62、D91、D92和SVRN-qKNN-BCRR的数值增益均为`UNKNOWN`。

## 7.放行边界

进入N607 Target25前仍需：

1.独立release复审达到`P0=0、P1=0`并形成同版本review receipt；
2.本地Git提交和文件SHA登记；
3.由唯一实验release子agent执行N607预检、精确同步、hash/compile和不可覆盖run-root；
4.先完成Phase1 source-held预测、独立truth-open score、gate、外部签名和formal seal；只有该资产链通过后，才准备并执行完整Target25。

在上述条件全部完成前，不得把本地landing、smoke或历史对比描述为D105性能成功。
