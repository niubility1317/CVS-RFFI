# D60跨块谱稳定性收缩追溯与预注册

## 1.动机与单一机制

D59在SPD流形中点上稳定完成，但与D45处于同一性能台阶：相对D46只把low-elev一个旧类样本换成一个新类样本损失。继续扫描full↔block位置会退化为D44–D50已停止的权重搜索。D60改问：哪些跨z160/FFT96/RF32相关方向能在合法support内部不同physical-rank留一折上稳定复现？

令完整auto-shrinkage协方差为`F`，三块协方差为`B`，白化跨块算子为`R=B^(-1/2)(F−B)B^(-1/2)=V diag(λ)V^T`。对当前support的每个physical-row rank`r`，每类留一条、仅用其余support拟合`F_r/B_r`，并在完整support的固定`B,V`坐标中计算：

`q_rj=v_j^T B^(-1/2)(F_r−B_r)B^(-1/2)v_j`。

第`j`个模态的无参数稳定度为：

`s_j=(mean_r q_rj)^2/mean_r(q_rj^2)`，若分母为0则`s_j=0`。

由Cauchy不等式，`0≤s_j≤1`。最终标准化协方差为`A*=I+V diag(s⊙λ)V^T`，部署协方差`G=B^(1/2)A*B^(1/2)`。它把不稳定模态连续收回block端点，稳定模态保留full强度；没有threshold、rank、geodesic位置或ridge。因`1+s_jλ_j`位于`1`与`1+λ_j`之间，而full协方差SPD，`G`保持SPD。

## 2.协议与边界

- 固定receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer physical-rank held折；实际outer fit K8。
- 复用匹配`VALIDATED_ONCE/p2_min_v1`的同一received-IQ capsule；不重建、不重验数据。
- inner fold只读outer support：每折每类留一条、train/held互斥且所有support row exact-once；inner-held本身不评分，只用于定义支持集补集。
- 不读取outer-held、query、receiver、场景、opaque handle、old/new角色、class quota/count或clean/source样本。
- 所有类别共享同一`G`；不做类别logit缩放/截距/权重，不产生双head query选择。
- K1精确D42单位协方差回退；K2因inner K1无法估计跨块稳定性，精确回退D59 SPD中点；K≥3激活D60。
- 不扫描稳定度公式、阈值、rank、block、ridge、温度、epoch或任何类别/场景参数。

## 3.预注册判门

基准为当前最强D46。必须同时满足：105/105行、query0和全部协议/artifact闭包；量化before/final argmax变化与margin翻转0/0/0；聚合before≥92.22%、after≥81.67%、new≥84.67%、H≥82.33%、forgetting≤10.56pp；joint≥23.33%、min-before≥80%、min-after≥53.33%、min-new≥73.33%且final三项至少一项严格提高；三场景联合不退化；混淆不超过25/8/15；至少1/15 prediction变化。

即使全部通过也只进入另行正式候选验证，不直接运行125。失败时停止D60，不扫描`s_j`指数、阈值或rank。结果报告必须覆盖7候选、3场景、11类、15fold、谱稳定度、混淆、量化、资源和artifact。

## 4.资源与验证

- before/final各1个main LDA＋实际K个inner covariance fit；K8时总fit`18`，必须据实计入MAC。
- D60只持久化一套编译后的int8/FP16 affine state；query extra MAC/state/optimizer step均0。
- 实现：`code/scripts/probe_d60_crossblock_spectral_stability_shrinkage.py`。
- 测试：`tests/test_probe_d60_crossblock_spectral_stability_shrinkage.py`。
- 输出：`automation_reports/CV-SincNet/d60_crossblock_spectral_stability_shrinkage_probe_20260719/crossblock_spectral_stability_shrinkage`。
- 本地`ssr-gpu`串行验证；建立detached clean worktree；本轮不访问N607。
