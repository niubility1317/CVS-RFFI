# D44 full/block RMS固定融合探针报告

## 1.身份与状态

- 实验ID：`d44_full_block_rms_fusion_probe_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`PREREGISTERED_PENDING_LOCAL_15_FOLD_PROBE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、clear/low-elev/rain、5个physical-rank held折。
- query sealed；复用D42/D43同一`p2_min_v1/VALIDATED_ONCE`固定received-IQ capsule、old-only B20、D22 ground int8与D19 class binding。

## 2.单一机制

D43证明full-centered保护新类/rain并实现0量化翻转；3-block-centered同时提高聚合before/after/new/H、旧类floor、joint floor和三类混淆，但硬置零跨块协方差使最低新类从70%降到66.67%，low-elev new和rain old退化。D44只验证两者能否以类对称方式互补，不再改变特征、B20、协方差估计器、support split或量化器。

对每套组件`g∈{full,block}`，在同一fit support上计算所有类logit并做行内类中心化：

`s_g=sqrt(mean_{i,c}((δ_g(x_i,c)−mean_j δ_g(x_i,j))^2))`。

融合score固定为`δ_D44=0.5δ_full/s_full+0.5δ_block/s_block`。`s_g`只读取support feature和两套全registry score，不读取support label、old/new role、匿名handle或场景身份；标签只用于两套原始等先验LDA拟合。两套affine可在编译前线性合成为一套coefficient/intercept，因此query仍是单行单state全registry argmax，不产生双模型选择、quota或global assignment。资源审计把before与final各两次covariance fit计为4次closed-form LDA，不能沿用D42的2次计数；query MAC/state仍只按融合后的单state计算。

固定权重只有`0.5/0.5`，不扫描融合权重、threshold、rank、lr、epoch、shrinkage、类/场景参数。若RMS非finite或≤`1e-12`则fail closed。

## 3.预注册判定

基准仍是D42 SHA256=`4ee51dd3d21ae8751bfaa64eb82d2a5a5371728fc7c1502bdb3af221d349614a`的15条原始全精度int8行；定义、全精度基准和`1e-12`比较规则完全沿用D43第4节，不根据D44结果更改。

D44进入正式实现必须同时满足：

1. lifecycle、ground、source、state、resource、query与复合source lock全部通过；
2. before/final int8-FP32 argmax变化均为0，三类margin符号翻转为0；
3. 聚合before-old、after-old、seen-new、H不低于D42，average forgetting不高于D42；
4. 最低before-old≥0.7666666666666667、最低after-old≥0.5、最低seen-new≥0.7、mean joint floor≥0.23333333333333334；后三者至少一项严格改善；
5. clear/low-elev/rain每个场景的before/after/new/H/joint均不退化，forgetting均不增加；
6. final old→new/new→old/new-new不超过D42的26/10/18；
7. 全部匿名类before/after/new、三类最低值、pairwise、量化、完整trace与资源均保存并报告。

探针复用D43已审计的D41 legacy＋12模块精确闭包、patched candidate lock、support/selection/receipt三方SHA重算和identity/full-K10 guard。即使D44全部门通过，本探针仍不自我晋级；只能据此另行实现正式候选。

## 4.文件与验证计划

- 探针：`code/scripts/probe_d44_full_block_rms_fusion.py`。
- 公共已审计helper：`code/scripts/probe_d43_structured_covariance.py`，仅把arm→structure表抽成可扩展常量，不改变D43三arm算法。
- 单测：`tests/test_probe_d44_full_block_rms_fusion.py`及D43/D42回归。
- 真实输出：`E:\type10-7\automation_reports\CV-SincNet\d44_full_block_rms_fusion_probe_20260718\full_block_rms_equal`。
- 环境：本地`ssr-gpu`，串行执行；只有全部本地门通过才考虑正式实现，当前不访问N607。

本地实现验证已完成：`py_compile`通过；D42–D44定向回归`56 passed`。验证覆盖固定RMS等权融合、公共score不变性、类别置换等变、K1 unit-covariance fallback、四次LDA拟合/MAC口径、D44专属30行artifact字段、资源闭包及篡改失败；D43的额外source closure同时改为拒绝覆盖保留键。pytest退出后的Windows临时目录清理出现既知`WinError 5`噪声，但进程退出码为0，不是测试失败。

当前完成预注册、实现和窄验证，尚未执行真实105行development探针，不构成性能正结果或目标完成。
