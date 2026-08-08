# Phase1双读出deployment bundle v2修复发布报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase1_dualreadout_bundle_v2_20260808_v2`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 证据等级：`FAILED_TECHNICAL_LANDING_PARTIAL_ZDOM_ONLY`
- 时间：2026-08-08
- 实现commit：`ab3b2f663b830cd6a3facd99d7fa9bb65c75f6cd`

## 1.目标与唯一修复

本run继承v1的全部方法、输入、source-only校准、bundle成员、阈值和评分锁，只修复一个真实执行缺陷：runtime wrapper显式调用`model(...,return_aux=True)`取得`z_id/z_dom/tx_logits`。新增测试复现默认forward只返回logits的真实接口并验证TorchScript trace parity，`ssr-gpu`共13项focused tests通过。

独立定向复审：`P0=0`、`P1=0`、`ALLOW_REPAIR_V2=YES`。固定commit的module/script/test归档SHA256分别为`c177dc87d547bf2f74b11808cec31343805151e80c472744fe8e4e2440d55896`、`5060dddfb1757fbdc42627415314f34a73445f5e22a4b16eafb68d2d010ab244`、`8402ff17d38ad87ebdfbeb30133c3660cfe2b6a83a80b043f9c5578f16a32bcb`。

v1的设备错误不改项目代码。v2并行命令禁止设置`CUDA_VISIBLE_DEVICES`，只传物理`cuda:0`、`cuda:1`、`cuda:2`；不得同时使用可见卡重映射和物理编号。

## 2.冻结输入与矩阵

| 输入 | SHA256 |
|---|---|
| B checkpoint | `f0f89b9251f6ada33778975b08ced3d9d407623b91a16d685d9d3bad9fa2070f` |
| C checkpoint | `9a1be4c739275f3c623f0df3d049f4a8b99b1ac51b21f4a50dca809fb23727e0` |
| B `z_id` NPZ | `31fc239ac7705488d1999b103902a04165ba4f4ccbfb1fe230f89a2a9f507c02` |
| C `z_id` NPZ | `b4e980a5495f2d297d61d461d30c6a510f9eb8bb9c51e31b2f16ebe7e247e4c6` |

GPU0导出B runtime，GPU1导出C runtime，GPU2导出C `z_dom` actual-IQ特征，每卡1个本run进程。三条命令成功后，CPU严格串行执行build、emit、proxy score和held score。方法仍为C负责registered类别、`z_id/z_dom`，B只提供连续JS陌生度；不做对齐、不训练、不调阈值。

## 3.N607不可覆盖路径

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_dualreadout_bundle_v2_20260808_v2_ab3b2f66`。
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dualreadout_bundle_v2_20260808_v2`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dualreadout_bundle_v2_20260808_v2`。
- CWD：`<release>/code`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- retry：`NO`。

## 4.成功、停止与回收

成功要求：两runtime的batch 1/8/64 eager↔TorchScript parity通过；C `z_dom`与B/C `z_id`逐行元数据一致；exact-allowlist bundle和外部content root通过；source-known smoke通过；生成2400条无role/truth的`proxy_unverified`证据；两份source-held非部署诊断完成。

停止规则保持v1不变，只针对hash、路径、设备、parity、行绑定、source-only fit、bundle禁用成员、truth/role泄漏、覆盖或确定执行错误；不得按指标停止、调参或重试。

只回收manifest、source-only receipt、两份parity receipt、smoke/evidence receipt、两份metrics JSON、日志、completion和哈希清单；不下载checkpoint、NPZ、runtime、calibration NPZ或完整evidence。

### 4.1实际终态

设备修复生效：三条子进程均回执`CUDA_VISIBLE_DEVICES=<UNSET>`，C `z_dom`导出exit=0并生成远端NPZ。angular和robust runtime均在`torch.jit.save`阶段出现同一`Could not export Python function call 'GradReverse'`，各exit=1；虽生成临时`.ts`，但没有parity receipt，因此不得进入bundle。CPU build、emit和score均未启动。

本run不重试、不覆盖。9项命令、环境、PID、completion、stdout和manifest小文件已回收且远端/本地哈希一致；partial runtime、NPZ和checkpoint均未下载。Python进程、GPU和SSH均清理完成。第2轮且最后一轮修复只允许导出不含训练期对抗头的部署子图，并使用新的v3 run。

## 5.结果表（待回收）

| candidate | category | receiver/TX split | K-shot | seed | known/unknown | coverage/defer | bundle summary | verdict |
|---|---|---|---:|---:|---|---|---|---|
| P1-DUALREADOUT-BUNDLE-V2-REPAIR | source-calibrated technical bundle | 4/1/1 | N/A | 7281105 | N/A | N/A | 仅`z_dom` partial，无bundle | `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE` |

## 6.科学边界

proxy=`8-20`和held=`6-15`只允许形成`SOURCE_HELD_PROXY_NONDEPLOYMENT_DIAGNOSTIC`。本run不提供Phase3真实unknown、same-event多节点或真实在轨同步结论。
