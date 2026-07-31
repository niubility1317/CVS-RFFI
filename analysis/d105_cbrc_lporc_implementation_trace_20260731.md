# D105-CBRC-MB4＋LPO-RC-qKNN实现追踪

## 1.状态

- 设计基线commit：`776dc6a4`
- 当前阶段：`R7_LOCAL_RELEASE_GO / ARCHIVE_SMOKE_PASS / PHASE1_R5_PREREGISTERED / NOT_LANDED`
- N607：R3停止于预启动Git archive字节门；R4通过全部预启动门并唯一detach一次，但首个tap-cache在零预测前触发PyTorch/NumPy对象边界异常，exit=1
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

旧45文件闭包在独立审查中被作废：tap-cache可动态进入两个legacy exporter，query evaluator可经通用`checkpoint_loading`进入训练栈；首次54文件修复后又发现模型工厂会在来源校验前探测清单外`model_modified.py`。这些问题均在N607落地前发现，未产生远端状态或性能结果。

R4的211项本地结论在N607发布前被跨平台字节P0作废：冻结manifest基于Windows工作树CRLF，Git archive部署为LF，导致24/54项远端SHA不一致。pipeline pid/log/exit始终不存在，未执行py_compile、checkpoint加载或任何性能流程。

当前R5修复把`.py`/`.sh`固定为LF，并要求全部54个runtime文件无CRLF；工作树与当前Git blob 54/54项SHA一致。10个指定D105测试文件共`212 passed，0 failed`。repaired runtime SHA256=`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`，method lock SHA256=`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`。

修复提交`46a65b3af2621d23bcc0a34631f45c8be17af4dd`生成的精确Git archive为4747项、242800640B、SHA256=`d313243c79eab306f988abadf67c2e207d380dba633f39a04e2cc63ffae7ed7a`。单根/成员类型检查通过；解包副本的canonical loader、54文件pyc和9个帮助入口通过，真实checkpoint无truth smoke SHA256=`a915eb66c4df926e6f738a4de636026fa29cb9bf3968c5fb6a15007ffc47ce84`。

R5独立发布审查最终结论为`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`，receipt SHA256=`65f8f211c01b8b72b4f4d7a385d9c1747b16dae9f14bddd32457ccf2f402c822`。新的非覆盖Phase1运行预登记为`d105_phase1_sourceheld_d23469ba_20260731_r4`：源码只取提交`d23469ba54afe00c284aa9b78b025def2b22fc43`的精确Git archive；launcher `run_d105_phase1_stage1_d23469ba.sh` SHA256=`b72fef60ab14ec86e9b53cc3355d07ac50598a3d6788000b4962726c89271d35`，5624B、LF-only、`bash -n`通过；预登记提交=`03e3ff67003f16b6c39596a521f5bfdf0401850c`。该run尚未落地。

R4远端实际PyTorch=`2.1.0+cu121`，checkpoint安全加载走已审查的`legacy_pickle_exact_frozen_sha_only`分支并闭合195个tensor。唯一detach PID=`2726125`后，首个`tap-cache`在`stage2_d105_phase1_bundle.py:1658`的`torch.from_numpy(batch)`触发`TypeError: expected np.ndarray (got numpy.ndarray)`；完整日志992B、exit=1，strict tap/prediction/truth/score/gate/component全部为0。运行后runtime仍54/54一致，无Target、Target25、authority或seal操作，GPU/进程/SSH均清理。R4终态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，handoff SHA256=`f362f5051a71d0dd88552a815c2a82680b6157a537ef1ebc36b1d8e720a3811a`。

修复提交`9f608e8be72024f00f1497cf6bddb9fb77e28201`以受检buffer复制桥替换D105正式Phase1/Target25的旧NumPy C-API桥，runtime/method SHA256=`8940e05f9fdf92d7735bba1570bb3239ee210313ecbbeffa3511b62e21685425`/`f36a0c6c4ee832b34cd98ed7664ec87707a4dbb1559c7c9b4b05dd13fbf4864e`。统一回归216/216通过，R7独立审查`P0=0、P1=0、P2=2`。精确Git archive SHA256=`dd85491e96f1cb9ea14e967694db91aec590e42273a2556492221af982ee9a67`；4754项，54/54、54 pyc、9帮助面通过；archive真实checkpoint无truth smoke收据SHA256=`fdea3e395b15d34ba7968037aa9a54ca835ec643e669c66de6854c8c3ff69a07`。新Phase1 R5预登记提交=`27fccbfc1d49599a4c9e5e82d301780b02fbad37`，尚未落地。

## 5.真实checkpoint派生特征无truth smoke

凭据：

`E:\type10-7\automation_reports\CV-SincNet\d105_cbrc_lporc_local_smoke_20260731_r6\real_feature_no_truth_smoke.json`

凭据SHA256：`a954896a5b3e3db91334ac564d967705568c892b5d2b7c6dbe42111a03d7c76c`

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

本次smoke绑定冻结runtime SHA256=`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`和method lock SHA256=`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`，耗时39.547秒。它复用已封存的checkpoint派生真实特征，没有重新从原始IQ执行backbone前向；它是实际checkpoint字节＋真实派生特征的机械闭环证据，不是Target性能、正式Phase2 row或完整runner证据。

## 6.历史性能定位

同口径历史对比见`analysis/stage2_baseline_comparison_20260731.md`。当前D105没有任何性能行，所有相对D62、D91、D92和SVRN-qKNN-BCRR的数值增益均为`UNKNOWN`。

## 7.放行边界

进入N607 Target25前仍需：

1.独立release复审达到`P0=0、P1=0`并形成同版本review receipt；
2.本地Git提交和文件SHA登记；
3.由唯一实验release子agent执行N607预检、精确同步、hash/compile和不可覆盖run-root；
4.先完成Phase1 source-held预测、独立truth-open score、gate、外部签名和formal seal；只有该资产链通过后，才准备并执行完整Target25。

在上述条件全部完成前，不得把本地landing、smoke或历史对比描述为D105性能成功。
