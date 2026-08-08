# Phase1双读出bundle CPU one-shot发布报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase1_dualreadout_bundle_v2_cpu_oneshot_20260808_v1`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 证据等级：`RUNTIMES_COMPLETE_BUILD_FAILED`
- 时间：2026-08-08
- 实现commit：`f9bc845cc723e3d886275fcdc824e5912118afb9`

## 1.目的与冻结入口

v1–v3已完成两轮发布修复并停止原GPU exporter。新one-shot不是第三轮修补：它是更小的独立入口，一条CPU命令顺序导出B/C两个部署子图，保持原`1e-5`parity门，不改方法、runtime、权重、阈值或校准。CPU只用于消除已观测到的CUDA eager↔TorchScript约`1e-3`数值漂移。

入口：`code/scripts/launch_phase1_dualreadout_runtime_cpu_oneshot_20260808.sh`。`PYTHON`、B/C checkpoint路径与SHA、`OUTPUT_ROOT`全部必填；B/C输出独立，OMP/MKL默认8线程。`bash -n`、原13项focused tests和`git diff --check`通过。

独立定向复审：`P0=0`、`P1=0`、`ALLOW_CPU_ONESHOT=YES`。固定commit的module/exporter/test/one-shot launcher归档SHA256分别为`c177dc87d547bf2f74b11808cec31343805151e80c472744fe8e4e2440d55896`、`8c65c08617f0903681331fb4cac5c8191d22e632a0d8765e809b8dfde31dec56`、`c53ffae3dadbd497ccbac3f1ce06347441c1b1a2781e3c34bbb8a2561b147a60`、`af0714a000d9f2b1af82d3a423aa4284cf1fd304888bd58137e733c5b3ea657c`。

## 2.冻结执行

- CPU one-shot：顺序生成`angular.ts/angular_parity.json`与`robust.ts/robust_parity.json`。
- GPU2：并行重新导出C `z_dom`，不复用v1–v3 partial；`CUDA_VISIBLE_DEVICES`保持UNSET，传物理`cuda:2`。
- 三项成功后，CPU沿用冻结命令串行build→emit 2400→proxy score→held score。
- 输入SHA、TX角色、C `z_dom`参数、bundle成员、content root、source-only fit和评分边界与v3相同。

## 3.N607路径与停止

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_dualreadout_bundle_v2_cpu_oneshot_20260808_v1_f9bc845c`。
- run/log：`/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/phase1_dualreadout_bundle_v2_cpu_oneshot_20260808_v1`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：`<release>/code`；retry=`NO`。

任何CPU parity、checkpoint/代码hash、设备、`z_dom`行绑定、source-only fit、bundle allowlist/content root、evidence字段或执行错误均立即停止且无性能结果。不得放宽tolerance、复用partial、按指标调参或覆盖旧run。

### 3.1实际终态

CPU one-shot和GPU2 `z_dom`均exit=0；B/C parity最大误差均为`0.0`，冻结门仍为`1e-5`，部署图不含GradReverse/adv heads。随后build因把局部作用域`sig_id`误当全局physical ID而失败：2400行中`sig_id`仅913个唯一值，但`(TX,RX,day,sig)`为2400个唯一值；source 1600行也全部唯一。emit和score未启动。

该run不重试、不覆盖。11项日志/control/manifest已回收且哈希一致；runtime、parity和`z_dom`作为已完成技术子产物保留在远端，未下载。后续新build one-shot可在验证固定哈希后引用这些只读产物，但不得把本run重标为完整bundle或性能结果。

## 4.结果表（待回收）

| candidate | category | receiver/TX split | K-shot | seed | known/unknown | coverage/defer | bundle summary | verdict |
|---|---|---|---:|---:|---|---|---|---|
| P1-DUALREADOUT-CPU-ONESHOT | deterministic deployment export | 4/1/1 | N/A | 7281105 | N/A | N/A | runtime/parity完成，bundle未生成 | `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE` |

## 5.科学边界

只回收小artifact；不下载checkpoint、NPZ、runtime、calibration或完整evidence。proxy/held指标仅为`SOURCE_HELD_PROXY_NONDEPLOYMENT_DIAGNOSTIC`，不构成Phase3真实unknown或same-event多节点结论。
