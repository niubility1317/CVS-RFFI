# D138 D92-Lite-FULL288 Target125实验报告r2

## 状态

- 实验ID：`d138_d92_lite_full288_target125_20260804_r2`
- 登记时间：`2026-08-04`
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 目标：在r1隔离source缺少D92 ground-fit probe依赖的技术故障修复后，完成同一冻结候选的125 outer、375 scene、750 before/after surface及独立truth-side诊断评分。
- 与r1关系：r1在真实smoke的query prediction前因`ModuleNotFoundError: scripts.probe_d92_registration_balanced_covariance`停止；r2为新的不可覆盖run，不续写r1。

## 冻结候选与边界

- 候选：`D92-Lite-FULL288/r1`；method lock：`configs/d138_d92_lite_full288_r1.json`；SHA256：`2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728`。
- 表示：sealed D92 runtime的完整`registered_feature_288`，即z_id160、FFT96和RF32；K1使用support-only full288类质心余弦，K5/K10使用support-only全类共享对角OAS float64仿射头。
- 并列规则：唯一最终float64胜者、full288 support-only类质心、canonical sorted full288 support fingerprint；支持证据完全相同则fail-closed。禁止registry顺序、class ID/hash、query truth/role、quota、global reassignment和跨query状态。
- 协议：`p2_min_v1`、LEO_weak-only、support/query物理ID互斥、query batch=1、query不fit/update/selection；矩阵为125 outer、375 scene、750 surface。
- 输入边界：当前sealed SOMP-H loader明确为`UNVERIFIED_UNDER_CURRENT_PROTOCOL_DIAGNOSTIC_ONLY`，`formal_launch_authority=false`，`formal_metric_claim_allowed=false`；本run完成后只能作为完整诊断结果，不能写成正式晋级/性能声明。

## 本地版本与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；运行时提交：`0270fbd7`（候选实现`a539fc58`、追踪修复`06cf3edb`）。
- 已验证依赖：`code/scripts/probe_d92_registration_balanced_covariance.py`，SHA256=`a04a9185e11ca851a1276004c4e2988cee0f6b61920851fe7a06ca7c740ee601`；r1缺失该文件，r2仅补齐这一运行时依赖。
- 本地`ssr-gpu`：候选核心/adapter/CLI、D92-Lite旧链路和回归测试通过；py_compile、pytest、`git diff --check`通过。

## N607落地与命令

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 不可覆盖run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_full288_target125_20260804_r2`。
- source基底：r1已hash验证的runtime closure；只额外同步上述probe文件，不复制r1的control/shard/partial输出。
- prepared复用：r6的`prepared/target125_plan.json` SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`；`target125_context.json` SHA=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：r2/source；GPU分片：0..7，一卡一分片。
- smoke通过后启动8个固定`predict-shard`，完成后依次merge、validate、build-truth、score；所有输出不可覆盖。

## 健康停止与成功标准

- 不按accuracy、H、BA、floor或中间值停止；只按协议/hash/覆盖故障，或至少两个不同row出现同一确定性prediction前异常停止。
- 成功条件：8/8 shard、125/125 outer、375/375 scene、750/750 surface、prediction manifest闭合、truth catalog和score summary完整；最终保留diagnostic-only标签。
- 预计产物：`smoke_row000_scene000`、`prediction_shard_0..7`、`merged`、`truth`、`score`及启动/监控日志。

## r2启动结果

- r2已完成新run目录创建和probe文件同步；依赖hash与本地一致，py_compile通过。
- 递归import复核在真实ground-fit导入前停止：`probe_d92_registration_balanced_covariance.py`继续依赖`probe_d81_ground_nuisance_cauchy_center.py`，该D81-D80-D66-D62-D61-D46-D45-D44-D43脚本闭包尚未落地。
- r2未产生smoke、prediction、truth或score；按两轮release-engineering规则冻结r3一次性补齐该闭包，不改方法、数据或矩阵。
