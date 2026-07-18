# D48一次性OOF-head margin残差探针报告

## 1.身份与目标

- 实验ID：`d48_one_shot_oof_margin_residual_probe_20260718`。
- 操作者：Codex`/root`。
- 当前状态：`PRE_REGISTERED_NOT_RUN`。
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折；每个outer fit实际K8。
- query sealed；不访问confirmation seeds，不生成125结果；当前不访问N607。

D45与D47最终决策完全相同，D46只在low-elev改变2条final argmax；三者共同保留rain O3旧类失效，说明继续平滑full/block权重无法触及主瓶颈。D48回到D45稳定的global LOO融合state，只增加一组由合法support inner-held margin统一生成的一次性class intercept residual，目标是同时修复低margin旧类和新类，而不使用类ID或old/new角色。

## 2.锁定算法顺序

完整继承D45 frozen-outer-B20 head-only LOO：full与3-block组件先canonical，各inner-train fold分别计算RMS；D45 global weight只由组件inner-LOO class-balanced CE计算。对inner fold`r`和匿名true类`c`：

`q_c,r,j=w_full×delta_full,c,r,j/s_full,r+w_block×delta_block,c,r,j/s_block,r`。

这里必须使用inner-train组件RMS，不能混用完整support RMS。随后：

`margin_c,r=q_c,r,c-max_{j!=c}(q_c,r,j)`；

`m_c=mean_r(margin_c,r)`，`mbar=mean_c(m_c)`；

`beta_raw,c=mbar-m_c`，`beta_c=beta_raw,c-mean_c(beta_raw)`。

完整support D45 affine state仍以完整support组件RMS和同一个global weight合成。只把`beta_c`一次性加入其intercept，再做canonical class-centering并进入既有residual-int8 coefficient/FP16 intercept编译。coefficient必须逐bit不变。禁止beta回流RMS、weight、margin、LDA或B20，禁止第二次beta或迭代到收敛。

## 3.协议与声明边界

support label用于选择true logit并排除true类后的`max_other`，属于合法support监督。每类使用同一mean/zero-sum公式，标签或support rank置换时beta和prediction列同步置换。无class ID表、old/new角色、receiver、scene、handle、outer-held、query、temperature、clip、threshold或扫描。max-other并列只使用相同最大值，不按class ID改变算法，tie仅审计计数。

本方法称`support-supervised one-shot OOF-head margin residual`。global weight与beta复用同一OOF support标签，且outer B20看过完整outer-fit support；这在协议内合法，但不是独立校准集，也没有无泄漏或泛化保证。support margin改善只是训练证据，不能替代outer-held性能门。

K1无inner fold，beta严格为0并逐bit回退D45 unit fallback。K2仍使用同一mean公式，不添加shrink或median；D45 unit components和global 1:1必须在`1e-12`内闭合，否则fail closed。C<2、非有限score/margin/beta、RMS或weight漂移、partition非exact-once、FP16溢出均fail closed。

## 4.资源口径

D48复用D45的B20、`4K+4` LDA inventory和一个query state，不新增fit、optimizer step、query state或sidecar，persistent state与query MAC不变。新增资源包括：

- full/block inner held component scoring：与D46同一精确公式；
- 完整support affine fusion：`2(D+1)(C_old+C_all)`；
- OOF margin residual保守MAC-equivalent上界：K1为0，K>1为`4K(C_old²+C_all²)+8K(C_old+C_all)+16(C_old+C_all)+32`。

当前实际K8/old6/all11的新增margin上界为6416；预计总adaptation为`1,077,334,386`。full/block/fused held logits与margin派生量的adaptation-time peak numeric evidence为26376B；持久化fit-audit改为对实际before/final证据和形式化量化数组执行canonical compact JSON UTF-8序列化后精确计数，真实字节数将在实验结果中报告，不再使用错误的numeric payload上界。它们只属于support训练审计，不是query sidecar，也不计入formal state。最终state仍为2016 trainable parameters、20 epoch/20 optimizer step、8583B、6624 query MAC；host FP64 covariance peak必须保留未实测状态。

## 5.预注册性能门

必须同时满足D42全部协议/lifecycle/source/ground/state/resource/artifact、聚合、三场景、最低before/after/new、joint、forgetting、混淆`26/10/18`和量化`0/0/0`门；D45 seen-new`84.00%`与matched-row H`82.16%`不得退化；全局min-new不得低于D46的`73.33%`；rain after-old/forgetting至少达到D42的`78.33%/10.00pp`；low-elev min-new至少达到D42的50%。

此外，至少一个真实fit的beta必须非全0，且final outer prediction相对D45至少改变1条。若全部预测不变、beta全0、support margin改善但outer门失败，均记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不允许事后换median、缩放beta、clip、迭代或加第二arm。即使全部通过，本探针也必须另行正式化和封闭开发验证，不能直接生成125。

## 6.文件、版本与验证

- 探针：`code/scripts/probe_d48_one_shot_oof_margin_residual.py`。
- D45 helper最小扩展：可选private held-score collector和post-fusion calibration callback；默认关闭，历史D45路径不变。
- 单测：`tests/test_probe_d48_one_shot_oof_margin_residual.py`及D42–D47回归。
- 追溯：`analysis/d48_one_shot_oof_margin_residual_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d48_one_shot_oof_margin_residual_probe_20260718\one_shot_oof_head_margin_residual`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，device=`auto`；runtime=`E:\type10-7\code\snapshots\d41wt`。

当前本地验证：首轮D48＋D45定向`31 passed`、D42–D48全链`124 passed`；修复独立代码初审发现的2项P1和3项P2后为定向`35 passed`、全链`128 passed`；进一步闭合formal int8 coefficient实际数组/fit FP32重编译绑定和真实JSON UTF-8计数后，定向`37 passed`、全链`130 passed`，py_compile通过。第一次`conda activate`落到base Python并因无pytest退出，随后用确认的`ssr-gpu`解释器串行重跑成功，未把包装噪声计为项目失败。设计审计确认协议P0=0并锁定同score单位、one-shot无回流、mean非median和声明边界；最终独立代码复审确认P0=0、P1=0、P2=0，且HEAD与当前D45默认路径在同一K1/K5输入上的coef、intercept与完整audit canonical JSON完全一致。

根目录`E:\type10-7`不是Git仓库；代码、测试、追溯和正式报告进入`github_publish/CVS-RFFI-repo`，只暂存本轮精确文件；根目录保留报告镜像。预注册提交、detached clean worktree、真实105行和完整日志判定待完成。
