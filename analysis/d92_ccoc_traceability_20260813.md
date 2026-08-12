# D92 CCOC需求追溯矩阵

状态：`DESIGN_FROZEN / IMPLEMENTATION_PENDING`

设计源：`docs/superpowers/specs/2026-08-13-d92-ccoc-strict-pareto-design.md`

## 证据起点

- 基线：`E0_FULL_ONLY`；当前D92相对E0七个方向改善，唯一反向为new→old约`+0.058333pp`。
- TCRA反证：Hard9只通过3/8方向，8/9 outer预测不变，wall P90=`336.968ms`。
- CSOAS反证：old floor`+10.3704pp`、forgetting`-4.7222pp`，但H`-0.4233pp`、seen-new`-4.6667pp`、new→old`+3.3241pp`。
- 排重：禁止FloorBoost旧类bias、NewGuard多codec扰动、ParetoDistill双fit、TPCE/TCRA原子搜索、CSOAS中心/协方差重估和rank-one Fisher/Mahalanobis变体。

## 需求到实现与证据

|ID|冻结需求|预定实现位置|验证证据|状态|
|---|---|---|---|---|
|CCOC-01|仅K>2注册态激活；K≤2精确D92 FULL alias|`stage2_d92_cross_class_offblock_consensus.py`、D92 probe/slim/query|低Kbyte-exact测试、fit receipt|待实现|
|CCOC-02|复用现有D92`Sigma_g^auto`和类均值，不重估中心/类内尺度|CCOC core|端点与均值identity测试|待实现|
|CCOC-03|raw`S_c`只计算off-block`Q_c/u_c`，不作协方差端点|CCOC core|秩亏K5 fixture、端点审计|待实现|
|CCOC-04|分别以old任务组内全部类和new任务组内全部类计算`rho_old/rho_new`的pairwise Frobenius cosine均值并clip到`[0,1]`|CCOC core|手算fixture、负相关/一致端点|待实现|
|CCOC-05|任一Q零范数/nonfinite即K>2 exact E0 fallback，不丢类、不设epsilon|CCOC core/probe|零Q与nonfinite RED→GREEN|待实现|
|CCOC-06|`Sigma_g*=rho Sigma_g^auto+(1-rho)blockdiag(Sigma_g^auto)`，最终0.5/0.5|CCOC core|公式、trace、SPD测试|待实现|
|CCOC-07|真实Cholesky；禁止伪逆和jitter|CCOC core|非SPD注入拒绝|待实现|
|CCOC-08|row permutation、组内label permutation、task swap对称|CCOC core|state/receipt等变测试|待实现|
|CCOC-09|K>2单FULL fit、单dense solve、无BLOCK/LOO/Fisher/scan|probe/slim/query|actual inventory与篡改拒绝|待实现|
|CCOC-10|query零访问；MAC和永久state与E0精确一致|query evaluation|七项禁用字段、state/MAC闭包|待实现|
|CCOC-11|正常路径一次D42；数值失败exact E0且G0不可用|query codec guard|数值异常与结构异常分流测试|待实现|
|CCOC-12|瞬时工作集上界`334,336B`且实际peak增量≤512KiB|core receipt/G0|公式测试、N607资源收据|待实现|
|CCOC-13|隔离E0/CCOC support-only技术执行；部署state非E0、至少一个rho严格内部，`max_j|Delta M_j|>=max_b(A_b*max four deployed block scales)>0`|G0 validator|真实K10三场景truth-free G0|待运行|
|CCOC-14|G0三场景active/no fallback，wall/ratio/peak硬门全过|G0 report/launcher|N607 artifact与validation JSON|待运行|
|CCOC-15|G0通过后Hard9+K1；八项任一tie/反向即拒绝|独立runner/analyzer|paired rows、逐类/scene/receiver、verdict|待运行|
|CCOC-16|Hard9全过才自动进入新Target125 run|主代理裁决/report|Git状态、不可覆盖run ID、sole runner handoff|待裁决|

## 八项方向

必须严格升高：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy。必须严格降低：average forgetting、new→old、old→new。任何加权总分不得补偿单项反向。

## 自动批准边界

用户已授权同类流程性审批自动通过，包括从已冻结设计进入实现、通过本地门后发布G0、G0通过后发布冻结Hard9，以及Hard9全门通过后创建新Target125 run。公式、协议、数据、矩阵、阈值、权限或破坏性操作的实质变化仍必须由主代理停下说明。
