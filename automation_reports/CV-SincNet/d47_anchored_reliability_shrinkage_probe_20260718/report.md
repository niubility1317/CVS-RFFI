# D47正部锚定可靠度收缩探针报告

## 1.身份与目标

- 实验ID：`d47_anchored_reliability_shrinkage_probe_20260718`。
- 操作者：Codex`/root`。
- 当前状态：`PRE_REGISTERED_NOT_RUN`。
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折。
- query sealed；不访问confirmation seeds，不生成125结果；当前不访问N607。

D46证明类级support inner-LOO可靠度可以改变真实决策并提高seen-new与最低new，但类间估计噪声同时放大旧类遗忘。D47保持B20、full/3-block LDA、canonical gauge、support RMS、int8生命周期、数据capsule、outer folds和比较门不变，只把D46逐类log-odds向D45全局锚点作无可调超参的正部矩收缩。目标是在保留D46新类收益的同时恢复rain与总体旧类稳定性。

## 2.机制与统计含义边界

对匿名类`c`和inner fold`r`，从合法support-held交叉熵构造：

`d_c,r=CE_block,c,r-CE_full,c,r`，`dbar_c=mean_r(d_c,r)`，`s_c²=Var_r(d_c,r)`。

D46类观察log-odds为`z_c=K×dbar_c`，其within-class log-odds方差代理为`u_c=K²×(s_c²/K)=K×s_c²`。令`mu=mean_c(dbar_c)`、`zbar=mean_c(z_c)=K×mu`。D45使用所有`C×K`个inner held样本，因此其全局锚点必须是`z0=C×mu`；`z0`与`zbar`在`C!=K`时不同，必须分别持久化，禁止把`K×mu`误当D45。

类间异质性采用固定正部矩估计：

`tau²=max(0,Var_c(z_c)-mean_c(u_c))`。

若`tau²=0`，固定`a_c=0`；若`tau²>0,u_c=0`，固定`a_c=1`；其余`a_c=tau²/(tau²+u_c)`。最终：

`zpost_c=(1-a_c)×z0+a_c×z_c`，`w_full,c=sigmoid(zpost_c)`，`w_block,c=1-w_full,c`。

`tau²=0`时精确退回D45权重公式；`a_c=1`时该类精确到D46权重公式。这里是公式级端点，不预先宣称candidate state字节等价，真实state关系必须由同一运行实测。该构造只称`positive-part anchored reliability shrinkage`，审计声明固定为`eb_inspired_deterministic_shrinkage_not_calibrated_posterior`。由于类间样本少且inner LOO折重叠，本探针不得把它描述成校准后的经验贝叶斯posterior或不确定性区间。

## 3.协议边界与特殊K

公式不读取class ID、TX、old/new角色、receiver、handle、场景、outer-held或query；无temperature、clip、阈值或权重扫描。support label仅用于合法的support监督拟合和inner可靠度计算。每个query仍独立对全部注册类argmax，无truth、role Oracle、class quota或global reassignment。

K1固定1:1等价回退。K2只有full/block逐fold逐类CE在数值容差内完全相等时才允许1:1，否则fail closed。sigmoid若因极端log-odds在FP64舍入到0或1也fail closed，不以事后clip掩盖。before state必须在首次new support读取前物化且不可变。

## 4.资源口径

D47复用D46主体计算：B20为2016个trainable parameters、20 epoch、20 optimizer steps；LDA inventory在K>1时为`4K+4`，可靠度评分与类级仿射融合MAC沿用D46精确公式。D47对已经持久化的`C×K`标量证据计算矩和权重，不能把新增计算记为0。每个before/final state的保守MAC-equivalent上界拆为：`6KC`覆盖fold evidence和一/二阶矩，`16C+8`覆盖跨类矩及正部收缩，`8C+8`覆盖post-logit、sigmoid和端点检查；两state合计`0 if K1 else 6K(C_old+C_all)+24(C_old+C_all)+32`并计入`estimated_adaptation_macs`。该上界对任意整数`K>=2`锁定，K1不执行矩代数且为0。新增LDA fit=0、新增optimizer step=0、新增query state=0、query sidecar=0。最终仍只持久化一个int8/FP16 query state。host FP64 covariance peak继续标记未实测，不能以CUDA峰值替代。

## 5.预注册晋级门

先继承D42全部协议、lifecycle、source、ground、state、resource、artifact、聚合、floor、逐场景、forgetting、joint、量化和混淆门。D47还必须同时满足：

- 聚合seen-new和最低new不低于D46的`84.67%/73.33%`；
- rain after-old不低于D42的`78.33%`，rain forgetting不高于D42的`10.00pp`；
- 相对D46至少改变1个final held预测；
- before/final int8-FP32 argmax变化与margin翻转均为0。

若`tau²=0`导致D47退回D45、全部final预测与D46相同、任一通用门失败或量化翻转，D47直接记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；不得事后添加第二arm、temperature、clip或shrinkage扫描。即使所有门通过，本探针仍是强制identity、禁止full-K10 refit的开发探针，只能进入另行正式候选实现与封闭开发验证，不能直接生成125或宣称正式性能。

## 6.文件、版本与计划命令

- 探针：`code/scripts/probe_d47_anchored_reliability_shrinkage.py`。
- 共享helper最小扩展：`code/scripts/probe_d46_classwise_loo_reliability_fusion.py`仅增加可选策略回调，默认D46路径不变。
- 单测：`tests/test_probe_d47_anchored_reliability_shrinkage.py`及D42–D46回归。
- 追溯：`analysis/d47_anchored_reliability_shrinkage_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d47_anchored_reliability_shrinkage_probe_20260718\anchored_reliability_shrinkage`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，device=`auto`。
- runtime：`E:\type10-7\code\snapshots\d41wt`；探针在本轮预注册提交的detached clean worktree运行。
- 输入：D18 receiver`20-1`/seed`713101`/K10-new5密封capsule及D22 component manifest、D19 class binding、D18 before/final enrollment seals。

根目录`E:\type10-7`不是Git仓库；代码、测试、追溯和正式报告进入`github_publish/CVS-RFFI-repo`，只暂存本轮精确文件；根目录只保留报告镜像。真实命令及所有输入hash将在预注册提交后写入本报告，输出目录必须预先不存在。

## 7.本地验证

- D47+D46定向测试：首轮`23 passed`；修复独立复核发现的2项P1、2项P2和1项P3后最终为`37 passed`，exit0。
- D42–D47继承链：修复前`90 passed`；最终`104 passed`，exit0。
- py_compile：通过。
- `C!=K`、complete-pooling D45公式权重端点、no-shrinkage D46公式权重端点、零异质性、手算部分收缩矩、标签置换、K1/K2完整链、稳定sigmoid、非零标量资源重算、K1/2/5/8/10/20上界常数、integrated fit/verifier和核心字段tamper拒绝均有测试。
- pytest结束后本机`pytest-current`出现既知`WinError 5`临时目录清理噪声，不影响测试退出码和结论。

## 8.待完成

独立代码复核、预注册提交、detached clean worktree、105行真实运行、完整日志解析、D42/D45/D46同row对照、逐场景/逐类/混淆/量化/资源/artifact闭包，以及最终晋级或拒绝判定尚待完成。
