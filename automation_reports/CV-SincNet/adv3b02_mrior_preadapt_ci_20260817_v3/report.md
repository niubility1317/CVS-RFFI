# ADV3B02 MRIOR预适应CI对比v3启动报告

- Run ID：`adv3b02_mrior_preadapt_ci_20260817_v3`
- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE`
- 冻结代码提交：`73d8dfae1b2418d21524d49f252ab83b2bd4140b`
- 目标：对ADV3B02按`receiver/seed/new_class_count/K/scene`执行MRIOR-SDA预适应，再以适应模型运行未改动的CSIL/MoPC-HR类增量注册。

## 前序停止与本次修复

v1因跨new-count错误复用旧support在plan生成前停止；v2已正确生成1200/800/2400 plan并通过prepare，但首个smoke job在训练前因runner未把项目`code/`加入`sys.path`而停止，0/6 artifact、无预测、无评分、无性能结果。v3使用全新run root，只修复runner独立CWD导入路径；方法、输入和矩阵不变。

## 冻结输入与矩阵

- 1200个MRIOR job、800个CI cell、2400个scene row；6-job预适应smoke和4-cell注册smoke通过后，才授权8个全量预适应分片。
- MRIOR：200steps、Adam lr=0.0006、estimate=7、target CE=1.0、DV-KL=0.005、mu=0.5。
- v7 plan SHA256=`1c5fb08231bc9d150d625e5360162c3ee287bdf778ac1c16a069ac187b96d65b`。
- source cache SHA256=`dea3bdb01d4f5791d7e92a01dbdcdb7f3d66b26bf134a375264b88eff8c6e4c4`。
- runner SHA256=`ffd93dc8065a7ec0b6c0219885d1eecafe9302afd878746cb915eb14aed077e9`。

## 本地验证

runner focused测试8项通过，其中独立CWD子进程实际执行脚本并成功导入`cvsrffi`；`py_compile`和`git diff --check`通过。N607上Task1、Task3、builder及1200-job修复已完成hash和编译核验。

## N607路径与命令顺序

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- PLAN：`/home/szu2070436088/2510044040/CV-SincNet/protocol_plans/adv3b02_mrior_preadapt_ci_20260817_v3/mrior_preadapt_ci_plan.json`
- RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_mrior_preadapt_ci_20260817_v3`
- 顺序：同步新runner并远端hash/compile→生成external plan→prepare→`preadapt_smoke`6job→`smoke`4cell→PASS receipt→8个`preadapt_shard`。
- GPU：smoke使用GPU0；全量分片使用GPU0至GPU7，每GPU一个本run进程。

## 停机、成功与产物

P0协议/覆盖/hash/checkout/launcher错误，或两个不同job在预测前出现同一normalized fingerprint时，仅停止本run并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；禁止依据性能停止，fresh retry未授权。预期产物为plan及SHA、run-root identity、6个smoke预适应artifact、4个smoke prediction/score/query receipt、smoke receipt、8个PID/日志及最终1200个预适应artifact。技术smoke不构成性能结果。

四状态报告继续使用`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；REG0的new accuracy和`H_old_new`为`N/A`，所有数值必须同排配对。
