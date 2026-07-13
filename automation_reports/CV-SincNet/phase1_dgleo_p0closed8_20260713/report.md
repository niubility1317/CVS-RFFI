# phase1_dgleo_p0closed8_20260713

- 协议：Phase1 source-only；ManySig；`rho_label=0.10`；120epoch；final-only；默认测试增强为`leo_weak`三场景。
- 目标：同时保护clean/strict UDU/receiver floor/satellite floor，并直接改善fixed p99、source_episode_overflow、legacy proxy/bridge、tail/overflow accept和radius/inter。
- 声明边界：不声明真实unknown FAR/FPR95、Stage2成功或真实在轨泛化。

## 机制

1. `virtual_detach=true`固定virtual negative，`gate_reference_detach=false`保留拒识gate几何梯度。
2. open梯度拆分为`boundary/source/invariant/u_geometry`四组，先执行分目标份额控制，再执行总open预算和closed冲突保护。
3. pseudo CE使用`confidence∩trusted_core`；U direct使用trusted core；U身份域不变性使用all-valid source-unlabeled。
4. tail绝对阈值继续阻断promotion/export，但不提前停止训练；finite metric reference允许在绝对阈值未达标时建立。
5. `eval_sat_on=all`导出逐receiver satellite指标和seed；scheduler仅在全部子实验`COMPLETE/0`时返回0。

## 矩阵

|GPU|candidate|侧重点|boundary/source/invariant/U|open预算|
|---:|---|---|---|---|
|0|`P0C_C0_BALANCED`|稳定联合|0.35/0.25/0.25/0.15|0.14-0.22|
|1|`P0C_C1_SOURCE_HEAVY`|source episode|0.25/0.35/0.25/0.15|0.14-0.22|
|2|`P0C_C2_INVARIANT_HEAVY`|receiver/day invariant core|0.25/0.25/0.35/0.15|0.14-0.22|
|3|`P0C_C3_BOUNDARY_ALIGNED`|fixed boundary|0.40/0.20/0.25/0.15|0.14-0.22|
|4|`P0C_C4_U_GEOMETRY`|U_s三态几何|0.30/0.20/0.25/0.25|0.14-0.22|
|5|`P0C_C5_SAT_INVARIANT`|clean-sat不变性|0.25/0.20/0.40/0.15|0.14-0.22|
|6|`P0C_C6_INTEGRATED_AGGRESSIVE`|source+boundary激进|0.30/0.30/0.25/0.15|0.18-0.26|
|7|`P0C_C7_DG_PROTECTED`|closed保护|0.30/0.25/0.30/0.15|0.14-0.22|

## 成功标准

- 8/8完成E120；U direct活跃≥80%，U invariance活跃≥95%；四组open梯度均可审计且非零。
- `source_episode_overflow`相对0.973下降至少0.05；fixed p99≤82.38°；legacy proxy/bridge与tail/overflow/ratio至少三项同向改善。
- clean overall/strict/receiver floor及sat strict floor相对上一批J5下降不超过1.5pp。
- 绝对tail、reference→final扩张或readiness不通过时继续fail-closed，不导出正式endpoint。

## 本地验证

- `ssr-gpu`环境下核心/控制/sat/launcher聚焦测试84 passed。
- Phase1 P1协议、post-stage、旧launcher和U_s回归45 passed。
- 新launcher专项3 passed；`py_compile`与`git diff --check`通过。
- 预计运行5-6.5小时，wall limit为10小时。

远端sync、SHA、命令、PID、GPU和终局结果在发布后补充。

## 占用保护

启动前N607存在1个DRIFT baseline及10个Stage2-B/C分片scheduler，GPU3-7各有2个现有任务。新增版本化排队器只在GPU compute与已知现有launcher全部退出并连续3次确认空闲后启动8卡矩阵；最长等待12小时，训练wall limit仍为10小时。排队超时返回75且不创建伪训练结果。
