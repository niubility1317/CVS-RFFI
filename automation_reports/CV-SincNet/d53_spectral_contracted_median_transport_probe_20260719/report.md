# D53谱收缩median transport开发报告

## 1.状态

- run ID：`d53_spectral_contracted_median_transport_probe_20260719`
- operator：Codex
- 状态：`IMPLEMENTED_AND_TESTED_PRE_RUN`
- 范围：本地receiver20-1、seed713101、K10/new5、3场景×5 folds；不访问N607、不运行125。
- 当前最强合法开发点仍为D46，不promotable。

## 2.目标与唯一公式

D51/D52共同证明median方向能修复old floor，但两个尺度均过大并伤害new。D53不直接按RMS或base norm缩放，而把median位移通过support类均值→D45判别向量的谱收缩映射转换为系数修正：

```text
M_c=mean_r(x_rc); Q_c=coordinate_median_r(x_rc); U=Q-M
M0=M-mean_c(M); W0=W_D45-mean_c(W_D45); tau=||M0||_2^2
G=U M0^T/tau; gamma_c=1-||mean_r(x_rc/||x_rc||)||_2
DeltaW=diag(gamma)G W0; W_D53=W_D45+DeltaW; b_D53=b_D45
```

`||G||_2≤||U||_2/||M0||_2`；无pinv、ridge、rcond、alpha、阈值、clip或扫描。K1/K2在谱检查前精确D45 fallback。

## 3.协议与比较

复用同一`VALIDATED_ONCE`、`p2_min_v1`胶囊；仅support固定received IQ视图，query test-only；禁止clean/source、truth/role/count/quota/global assignment/query optimization/dense query graph。before/final同式。直接比较D45、D46、D51、D52，必须同时评价domain adaptation和new-class registration。

## 4.本地文件与验证

|文件|用途|
|---|---|
|`code/scripts/probe_d53_spectral_contracted_median_transport.py`|探针、closure、资源账|
|`tests/test_probe_d53_spectral_contracted_median_transport.py`|公式、谱界、对称性、fallback测试|
|`analysis/d53_spectral_contracted_median_transport_traceability_20260719.md`|追踪矩阵|
|本报告|执行与性能证据|

D53定向11/11通过；D45–D53联合127/127通过。`py_compile`、`git diff --check`通过。

## 5.成功与停止门

- 相对D45至少1/15预测变化；
- 总体及场景after/new/H、forget、joint、min-after、min-new联合审查，不允许old改善换new伤害；
- 至少保持D46的new84.67%、min-new73.33%，并改善old侧、遗忘或joint；
- query/role/quota/count/global/clean/source保持0/false，量化变化0/0/0；
- 失败即停止，不扫描谱尺度、不clip、不加role/scene门控、不跑第二seed、不formalize、不运行125。

## 6.计划运行与完整报告

实现提交后建立`E:\type10-7\code\snapshots\d53wt`clean worktree；runtime继续只读`d41wt`；输出为`E:\type10-7\automation_reports\CV-SincNet\d53_spectral_contracted_median_transport_probe_20260719\spectral_contracted_median_transport`；本地`ssr-gpu`、`device=auto`、单进程。完成后必须报告7候选、3场景、逐类、15 folds、相对D45/D46/D51/D52、20epoch、混淆、谱行为、量化、资源及artifact SHA，不得只报告缺陷。
