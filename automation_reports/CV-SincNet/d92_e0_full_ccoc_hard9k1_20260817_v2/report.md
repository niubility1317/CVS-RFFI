# D92 E0 FULL CCOC Hard9+K1 v2实验发布报告

状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`

## 1. 身份与目的

|字段|值|
|---|---|
|run ID|`d92_e0_full_ccoc_hard9k1_20260817_v2`|
|记录时间|2026-08-17|
|操作方|CCOC Hard9 v2发布代理|
|本地工作树|`E:/type10-7/code/snapshots/d92_125wt`|
|运行时Git基线|`fe9033be177f52d17b6a391574dd2b755bd40f37`|
|目标|执行冻结的9个performance outer和1个K1 liveness outer，取得CCOC相对同排E0的完整truth-last证据|
|比较对象|`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`对同排`E0_FULL_ONLY`|

v1运行`d92_e0_full_ccoc_hard9k1_20260816_v1`在`prepare`前因启动探针写成不存在的`cvsrffi.stage2_registration_balanced_covariance`而技术失败，保留为`NO_PERFORMANCE_RESULT`。v2只把该名称改为实际模块`cvsrffi.stage2_d92_registration_balanced_covariance`，并分配新的运行身份与全部远端路径；科学机制、矩阵、阈值、资源门和query门均未改变。

## 2. 本地发布物

|项目|路径或身份|
|---|---|
|配置|`configs/stage2_d92_ccoc_hard9_k1_v2.json`|
|运行时归档|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v2/runtime/d92_ccoc_hard9_k1_source_fe9033be_20260817_v2.tar.gz`|
|归档闭包|48个Git源成员+1份source manifest；不含数据、checkpoint、truth、测试、文档和G0 runner/core|
|归档SHA256|`4f75acdfde68e6879e8fb8199bf8b3869baea8a262da1626731465a79697f988`|
|归档大小|314210B|
|启动脚本|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v2/launch.sh`|
|启动脚本SHA256|`1d7f81486bfb1f6ede505a432429564d43a43318786d65289f85b6ea3eacde2d`|
|外部镜像|`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v2/`|

本地变更仅包括v2运行身份、v2配置、启动模块名、发布报告/清单、同步映射及相应聚焦测试。v1配置和v1本地/外部证据保持不变。

## 3. N607预登记

|字段|值|
|---|---|
|远端project|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_fe9033be_20260817_v2.tar.gz`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_fe9033be_20260817_v2`|
|远端driver|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v2.sh`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_20260817_v2`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_hard9k1_20260817_v2`|
|本地取回根|`E:/type10-7/local_artifacts/d92_e0_full_ccoc_hard9k1_20260817_v2`|
|Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|启动CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs`|
|PID|启动后由sole runner记录|
|GPU|smoke=`GPU0`；shard0–7分别映射`GPU0–7`|

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v2.sh >./d92_e0_full_ccoc_hard9k1_20260817_v2.launch.out 2>./d92_e0_full_ccoc_hard9k1_20260817_v2.launch.err </dev/null &
```

启动顺序严格为`prepare→truth-free smoke→8 shards`，不自动运行analyzer，不允许同run重试。所有v2 archive、source、driver、output、log和本地取回路径必须在各自创建前不存在。

## 4. 冻结矩阵与门槛

- 协议：`p2_min_v1`，复用`VALIDATED_ONCE`数据，不重验数据。
- 场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 矩阵：9个performance outer+`rx_20_1__seed_713106__k_1__new_20` liveness；总计10 jobs、30 scene receipts、8 shards；G0 outer排除。
- performance seeds：`713102`、`713103`、`713104`、`713105`、`713106`，以配置内冻结receiver/K/new组合为准。
- 注册资源hard：逐scene peak≤1048576B、wall≤150ms、candidate/E0 wall ratio≤1.50。
- 注册资源target：peak≤524288B、wall P90≤120ms、ratio P90≤1.25。
- 实时推理：query MAC和永久state bytes必须与同排E0完全相同；query truth/fit/update/selection/role/quota/global reassignment全部为false。
- 性能观察：`H_old_new`、old BA、`c_old_acc`、old floor、seen-new accuracy严格增加；average forgetting、new-to-old、old-to-new严格降低。K1不进入性能裁决。

预注册技术停止仅限协议/安全错误、错误checkout/路径、覆盖风险、零prediction或至少两个不同outer在prediction前产生同一确定性异常fingerprint。不得依据准确率、H、BA、floor或其他性能值停止。

## 5. 预期artifact与完成检查

预期产生`matrix_manifest.json`、truth-free smoke receipt、10份job receipt、30份scene closure、8份shard summary以及完整日志。sole runner只检查进程、PID/CWD/cmdline、GPU、日志增长、prediction/COMMIT/fit/resource/score/summary计数和异常fingerprint，不读取性能。

完成后取回完整source/output/logs及10份manifest绑定truth sidecar；主代理在本地运行冻结analyzer，按同一`outer_key+scene+arm`连接证据并生成唯一裁决。只有`ADVANCE_TO_TARGET125_CANDIDATE`才允许新建Target125发布。

## 6. 本地验证

已验证：旧v1解包导入探针稳定复现缺失模块RED；v2精确归档解包后launcher import list全部可导入；runner/analyzer`--help`可用；matrix/runner聚焦测试、`py_compile`、JSON读取、`bash -n`、tar路径安全及`git diff --check`通过。当前未SSH、未SCP、未启动N607、未运行Hard9性能分析。

## 7. 风险与后续检查

- N607环境与路径存在性必须由sole runner在同步前重新核对；本报告不把本地成功写成远端成功。
- v2只修复已定位的启动探针模块名；若出现新的确定性技术故障，保留全部artifact并按新run ID处理，不覆盖v2。
- 运行结束后优先检查30个scene closure、10个job receipt、8个summary、query零访问、资源hard和truth-last绑定，再由主代理读取性能。
