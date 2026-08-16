# D92 CCOC需求追溯矩阵

状态：`G0_V6_MECHANISM_RESOURCE_PASS / HARD9_K1_IMPLEMENTING / NO_PERFORMANCE_RESULT`

设计源：`docs/superpowers/specs/2026-08-13-d92-ccoc-strict-pareto-design.md`

## 证据起点

- 基线：`E0_FULL_ONLY`；当前D92相对E0七个方向改善，唯一反向为new→old约`+0.058333pp`。
- TCRA反证：Hard9只通过3/8方向，8/9 outer预测不变，wall P90=`336.968ms`。
- CSOAS反证：old floor`+10.3704pp`、forgetting`-4.7222pp`，但H`-0.4233pp`、seen-new`-4.6667pp`、new→old`+3.3241pp`。
- 排重：禁止FloorBoost旧类bias、NewGuard多codec扰动、ParetoDistill双fit、TPCE/TCRA原子搜索、CSOAS中心/协方差重估和rank-one Fisher/Mahalanobis变体。

## 需求到实现与证据

|ID|冻结需求|预定实现位置|验证证据|状态|
|---|---|---|---|---|
|CCOC-01|仅K>2注册态激活；K≤2精确D92 FULL alias|`stage2_d92_cross_class_offblock_consensus.py`、D92 probe/slim/query|低Kbyte-exact测试、fit receipt|本地通过|
|CCOC-02|复用现有D92`Sigma_g^auto`和类均值，不重估中心/类内尺度|CCOC core|端点与均值identity测试|本地通过|
|CCOC-03|raw`S_c`只计算off-block`Q_c/u_c`，不作协方差端点|CCOC core|秩亏K5 fixture、端点审计|本地通过|
|CCOC-04|分别以old任务组内全部类和new任务组内全部类计算`rho_old/rho_new`的pairwise Frobenius cosine均值并clip到`[0,1]`|CCOC core|手算fixture、负相关/一致端点|本地通过|
|CCOC-05|任一Q零范数/nonfinite即K>2 exact E0 fallback，不丢类、不设epsilon|CCOC core/probe|零Q与nonfinite RED→GREEN|本地通过|
|CCOC-06|`Sigma_g*=rho Sigma_g^auto+(1-rho)blockdiag(Sigma_g^auto)`，最终0.5/0.5|CCOC core|公式、trace、SPD测试|本地通过|
|CCOC-07|真实Cholesky；禁止伪逆和jitter|CCOC core|非SPD注入拒绝|本地通过|
|CCOC-08|row permutation、组内label permutation、task swap对称|CCOC core|state/receipt等变测试|本地通过|
|CCOC-09|K>2单FULL fit、单dense solve、无BLOCK/LOO/Fisher/scan|probe/slim/query|actual inventory与篡改拒绝|本地通过|
|CCOC-10|query零访问；MAC和永久state与E0精确一致|query evaluation|七项禁用字段、state/MAC闭包|本地通过|
|CCOC-11|正常路径一次D42；数值失败exact E0且G0不可用|query codec guard|数值异常与结构异常分流测试|本地通过|
|CCOC-12|瞬时工作集上界`334,336B`；注册增量peak目标≤512KiB、硬门≤1MiB|core receipt/G0|公式测试、N607资源收据|v3实测clear=`729,088B`；v4新硬门待验证|
|CCOC-13|隔离E0/CCOC support-only技术执行；部署state非E0、至少一个rho严格内部，`max_j|Delta M_j|>=max_b(A_b*max four deployed block scales)>0`|G0 validator|真实K10三场景truth-free G0|v3除旧peak门外通过；v4待运行|
|CCOC-14|G0三场景active/no fallback，wall/ratio/peak硬门全过|G0 report/launcher|v6 G0 hard pass；candidate wall P90=70.463259ms、ratio P90=1.142053、maxpeak=729088B；target512KiB在clear scene未通过、hard1MiB三scene通过|技术硬门通过；NO_PERFORMANCE_RESULT|
|CCOC-15|G0通过后Hard9+K1；八项任一tie/反向即拒绝|独立runner/analyzer|Hard9+K1 implementing；G0 artifact仅闭合技术门，尚无Hard9 paired rows|不能写性能结论；NO_PERFORMANCE_RESULT|
|CCOC-16|Hard9全过才自动进入新Target125 run|主代理裁决/report|本地不可覆盖发布包、唯一launch、prepare→truth-free smoke→8 shards顺序、sole-runner handoff|release ready no performance|

## 八项方向

必须严格升高：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy。必须严格降低：average forgetting、new→old、old→new。任何加权总分不得补偿单项反向。

## 2026-08-17 Task3本地发布状态

`d92_e0_full_ccoc_hard9k1_20260816_v1`已在truth-order HEAD`07e69f1e4360eacec2b972f465c4c19dd1710fd8`基础上完成本地发布物修复，状态为`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`。归档为49成员、48个源成员，SHA256为`62a8a1f8536e8df6d6cbbdc9314ae86519817d44b3982a7acaf42be8df949cdb`；补入D80/D81 core及D43/D44/D45/D46/D61/D62/D66/D80动态probe，不含G0 runner/core；解包后runner/analyzer `--help`均rc0。launch、报告、交付清单及外部非Git镜像逐字节绑定。注册资源hard上限为1MiB、target为512KiB，query MAC、state、per-query latency及其他冻结门保持不变。未执行SSH、SCP、N607 launch或性能分析；Hard9是否通过及后续Target125仍由主代理依据完整artifact裁决。

## 2026-08-17 v2技术修复发布

v1在远端`prepare`前由启动探针触发`ModuleNotFoundError`，因此保留为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`且不覆盖。新的`d92_e0_full_ccoc_hard9k1_20260817_v2`只将错误模块名更正为实际存在的`cvsrffi.stage2_d92_registration_balanced_covariance`并分配全新的source/output/log/driver/retrieval路径；运行时Git基线为`fe9033be177f52d17b6a391574dd2b755bd40f37`。精确解包探针已验证launcher列出的全部模块可导入；科学机制、9+1矩阵、1MiB注册hard/512KiB target以及严格实时query门保持不变。当前仍是`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`，未执行SSH、SCP、N607启动或性能分析。

## 自动批准边界

用户已授权同类流程性审批自动通过，包括从已冻结设计进入实现、通过本地门后发布G0、G0通过后发布冻结Hard9，以及Hard9全门通过后创建新Target125 run。公式、协议、数据、矩阵、阈值、权限或破坏性操作的实质变化仍必须由主代理停下说明。

## 2026-08-17 v3 SHA-only归档发布

`d92_e0_full_ccoc_hard9k1_20260817_v3`是新的、不可覆盖的CCOC-16发布身份。它只移除解包归档必须包含Git commit/HEAD对象的冗余运行时门；17个method-lock源文件仍按实际字节SHA256逐文件fail-closed验证，48个归档源成员另由source manifest闭合，prepare明确报告`sha256_only`。v3配置相对v2只改变`runtime.output_root`；9+1矩阵、三scene、1MiB注册hard、512KiB target、150ms/1.50 hard以及严格query MAC/state/零访问门均不变。当前状态为`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`；v1/v2证据保持不变，未执行SSH、SCP、N607启动或analyzer。

## 2026-08-17 v4 E0资源基线身份发布

`d92_e0_full_ccoc_hard9k1_20260817_v4`是新的CCOC-16不可覆盖身份。E0 fit-audit改为对当前文件执行存在性、非symlink、outer/scene/E0身份与resource schema/query MAC/state/wall/peak合法性验证，实际observed SHA只写入manifest/receipt追溯，不再与历史SHA比较。`runtime_source_verification_mode=sha256_only`、truth/prediction闭包、9+1矩阵、资源阈值和query门不变。状态为`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`；v1/v2/v3证据不变，未执行SSH、SCP、N607启动或analyzer。

## 2026-08-17 v5内嵌E0资源投影发布

`d92_e0_full_ccoc_hard9k1_20260817_v5`是新的CCOC-16不可覆盖身份。prepare不再要求外部E0 fit-audit存在或打开它，直接验证并使用Git封存的`e0_resource.scenes`；`e0_resource_source_mode=embedded_preregistered_projection`，历史path/SHA只作declared trace metadata。truth/prediction闭包、9+1矩阵、资源阈值和query门不变。状态为`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`；v1/v2/v3/v4证据不变，未执行SSH、SCP、N607启动或analyzer。

## 2026-08-17 v6 query-zero validator发布

`d92_e0_full_ccoc_hard9k1_20260817_v6`是新的CCOC-16不可覆盖身份。fit-audit validator仅要求7个base字段与7个批准`d92_e0d_ccoc_`镜像存在且为false，不再要求generic `d92_e0d_`或raw `d92_ccoc_`别名存在；未批准字段仍由query whitelist拒绝。query代码、truth/prediction闭包、9+1矩阵、资源阈值和实时推理门不变。状态为`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`；v1至v5证据不变，未执行SSH、SCP、N607启动或analyzer。

## 2026-08-17 v7优化core runtime绑定发布

`d92_e0_full_ccoc_hard9k1_20260817_v7`将已验证的core优化提交`1429a496739dfadaf169b83ddf86b3b831f174d5`绑入runtime source lock；core blob为`abb45a15514c8c5758e0cececb930a76d27a29b8`，SHA256为`6f87d4eb041ba8874182a46eb3f2a76dc3f2f075a6692ee217f19bcd2f8ff331`，49成员归档包含该字节。本发布未再改动科学代码、方法、9+1矩阵、阈值、query或truth/prediction闭包。状态为`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`；v1至v6证据不变，未执行SSH、SCP、N607启动或analyzer。
