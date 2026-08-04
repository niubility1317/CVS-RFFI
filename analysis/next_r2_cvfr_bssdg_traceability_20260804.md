# NEXT-R2 CVFR-BSSDG/r1设计冻结与实现追踪

## 状态

`DESIGN_DRAFT → FEASIBILITY_REVIEW → DESIGN_FROZEN → IMPLEMENTING`

独立Terra/max交叉复核结论为`P0=0、P1=0、DESIGN_FROZEN_GO`。本revision只保留`CVFR-BSSDG/r1`，不叠加RDCE、FABR、CSPAR、SRDH、qKNN或历史D92分支。

## 可行性摘要（20行内）

1. D130中CSPAR K5为`ΔH=-0.556pp、正确数-9`，SRDH为零效应；D137因qKNN精确tie在24/84技术退出。
2. 历史D92完整125可减旧类遗忘但牺牲新类，K1逐值不变；BSSDG必须原生覆盖K1且不按old/new角色分支。
3. CVFR只拟合融合后pre-ReLU160的共享319维轻量残差，不更新checkpoint参数。
4. 同一support received IQ的`0、+π/4、-π/4`只作为数学view，不增加K。
5. 159维Helmert log-scale contrast消除L2归一化的uniform-scale gauge；160维shift与其共同拟合。
6. Jacobian不可辨识时发布合法identity DA并继续四态预测，不把support不足变成实验前阻断门。
7. finite exact-zero统一映射为零向量；BSSDG允许零support/query，只在pooled trace为零时失败。
8. BSSDG使用当前state全部support的class-blind pooled diagonal prior与每类标量Student-t不确定度。
9. `τ0=1、ν0=4、νv=1`固定；不扫描先验强度、trust cap、view角度或量化阈值。
10. K1不alias qKNN；duplicate score tuple或exact top tie继续fail-closed，禁止ID/order/hash/ULP破tie。
11. 最小矩阵为2个source-only receiver×6 held-class LOCO×K1/K5=24个outer key。
12. 每个key产出`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`四态，共96个prediction artifact。
13. REG0只报旧类与floor，`N/H=NA`；REG1才报旧类、新类、H、floor与总正确数。
14. 96态全闭合后才判定；完整负结果关闭候选，不调参、不补跑。
15. 24-key正向只允许进入更完整source-held复核，不进入Target25或125。

## CVFR冻结定义

令`H∈R^(160×159)`为固定Helmert基，满足`HᵀH=I、Hᵀ1=0`；`u=Ha`，参数`δ=(a,v)∈R^319`。当前state canonical support的标签无关global RMS为`m`。共享映射为

`yδ(s)=exp(u)⊙s+m v`，`Tδ(s)=yδ/||yδ||₂`；当`||yδ||₂=0`时定义`Tδ(s)=0`。

只在`δ=0`进行一次FP64 Gauss-Newton线性化，目标由canonical分别对`+π/4`与`-π/4`view的残差，以及全部注册类canonical中心的pairwise Gram保持组成。固定算法约束为`||δ||₂≤√2`；它不是物理安全界，也不得根据性能修改。

数值rank阈值为`max(n_row,319)·eps64·σmax`，condition上限为`1/sqrt(eps32)`。rank不足或condition超限产生`DA_IDENTITY_UNIDENTIFIABLE`；可辨识但解为零产生`DA_IDENTITY_ZERO_SOLUTION`。两者都必须完成预测。只有非有限、错误绑定、协议违规或无法产生预测才是`NO_PERFORMANCE_RESULT`。

## BSSDG-160冻结定义

每个四态用该state全部support重算class-blind prior。设`n=|Z|`、`m0=mean(Z)`、`s_j²=n⁻¹Σ_i(z_ij-m0_j)²`、`sbar²=d⁻¹Σ_j s_j²`，则

`v0_j=(n·s_j²+sbar²)/(n+1)`。

`sbar²=0`或非有限时失败。对类`c`，以`τ0=1、ν0=4`计算收缩均值`m_c`、标准化残差能量`A_c`与标量`ρ_c`：

`τ=τ0+K`，`m_c=(τ0m0+K zbar_c)/τ`，

`A_c=Σ_k||z_ck-zbar_c||²_(V0^-1)+(τ0K/τ)||zbar_c-m0||²_(V0^-1)`，

`ν=ν0+K`，`ρ_c=((ν0+A_c)/ν)·((τ+1)/τ)`。

全类等先验分数为

`g_c(x)=-(d/2)logρ_c-((ν+d)/2)log1p(||x-m_c||²_(V0^-1)/(νρ_c))`。

`s_c、v0_j`使用正normal FP16；`logρ_c`和intercept使用signed FP16，零合法，非零值按绝对值检查normal范围。任何clip、qKNN fallback、class token tie-break或query侧更新均禁止。

## 四态与判定

|状态|support注册集|DA|允许指标|
|---|---|---|---|
|`DA0_REG0`|5个retained class|identity|旧类BA、逐类、floor；`N/H=NA`|
|`DA1_REG0`|5个retained class|CVFR或合法identity|旧类BA、逐类、floor；`N/H=NA`|
|`DA0_REG1`|6个全注册class|identity|旧类、新类、H、floor、总正确数|
|`DA1_REG1`|6个全注册class|CVFR或合法identity|旧类、新类、H、floor、总正确数|

必须报告`DA1_REG0-DA0_REG0`、`DA1_REG1-DA0_REG1`、`DA0_REG1-DA0_REG0`、`DA1_REG1-DA1_REG0`，以及旧类共同指标的差分中的差分。若全部DA1为identity或与DA0预测完全相同，则完整矩阵后关闭为`NO_FUNCTION`。否则K5 pooled的`DA1_REG1-DA0_REG1`必须同时满足`ΔH>0`、总正确数增加、`ΔA_retained≥0、ΔN≥0、ΔF_retained≥0`，才允许进入更完整source-held复核。

## 实现责任边界

- DA作者：`code/cvsrffi/stage2_next_r2_cvfr.py`与`tests/test_stage2_next_r2_cvfr.py`。
- HEAD作者：`code/cvsrffi/stage2_next_r2_bssdg.py`与`tests/test_stage2_next_r2_bssdg.py`。
- 集成作者：后续单独拥有runtime、24-key matrix、真实bridge与runner文件，不修改DA/HEAD核心。
- 独立reviewer：不参与上述核心实现；只做P0/P1复核。
