# D92 E0 FULL CSOAS Hard9+K1 v2实验报告

状态：`LOCAL_VERIFIED / READY_FOR_N607_HANDOFF`

## 1.目标与冻结范围

|字段|值|
|---|---|
|run ID|`d92_e0_full_csoas_hard9k1_20260812_v2`|
|科学commit|`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`|
|机械实现/P1修复|`ac811820`/`7a434080`|
|v1递归修复|`1fab89eb15d66970f4725e897da585f477a791bd`|
|候选|`E0_FULL_CSOAS`；candidate=`d92_e0_full_csoas`；mode=`csoas_full`|
|矩阵|与G0不重叠的9个最难performance outer+1个K1 liveness；3 scenes；10 jobs；8 shards|
|声明|development-only Hard9；完整artifact和冻结analyzer前无性能结论|
|retry|`false`；v2唯一detached launch；不得resume/覆盖v1|

v1已固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`：prepare完成后，smoke在预测前因runner verifier动态自引用触发`RecursionError`，8 shards未启动。v2唯一变化是冻结原始base verifier引用；方法、矩阵、阈值、数据、scorer和analyzer均不变。

## 2.本地修复与发布门

- TDD RED：新增回归测试在父实现上复现同一`RecursionError`。
- GREEN：runner focused `12 passed`；`py_compile`与`git diff --check`通过。
- 独立复审：`P0=0/P1=0，APPROVE`；确认context内调用冻结原始helper且`finally`恢复base属性。
- selection SHA256=`a851590bc6d502ddbe326a936096d95f5bb382e4cb235b61b0121d98c0b87b5d`。
- 数据继续复用`p2_min_v1/VALIDATED_ONCE`，不重验；query fit/update/selection/truth/role/quota/global reassignment必须全false。

## 3.性能与资源裁决

9个performance outer必须逐row同E0比较，K1只做liveness。八项总体均值全部严格优于E0才允许晋级：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy升高；average forgetting、new→old、old→new降低。任何tie/反向、K>2 fallback/codec retry、稳定性或资源硬门失败均`REJECT_ROUTE`，不跑target125。

大胆目标依次为`+1.0pp,+1.5pp,+1.0pp,+4.0pp,+0.5pp,-1.5pp,-0.5pp,-0.5pp`。资源硬门：wall P90≤150ms、paired wall ratio median≤1.50、peak delta≤512KiB、query MAC/state逐row等同E0；120ms/1.25是目标门。

## 4.交付与远端路径

|交付|size|SHA256|
|---|---:|---|
|`d92_csoas_hard9_runtime_1fab89eb.tar.gz`|6,184,420|`de74fe49d8d24432898e44fddfc3c8a9f2f2444b2d70421e7d69d786c9a25d78`|
|`stage2_d92_csoas_hard10_v1.json`|6,293|`6fcd29dfab77c99745df336f32425dfdc0a0a0a99469c92766a4751fa92e427e`|
|`launch.sh`|3,717|`c8e87a3d75e2d6ac76c50f21bc3fb0826d8ed5967522d605be05740f92fe7bed`|

archive共1466 members，runner SHA256=`623c7138e7e70bde6e4ef49bfd0dcd6f66d1b7203e5ef09e876bc458f1d8c08c`，必需入口齐全，无绝对/`..`/`code/code`路径；launch以LF内容通过`bash -n`。

|用途|远端路径|
|---|---|
|source|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_hard9_source_1fab89eb_20260812_v2`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_hard9k1_20260812_v2`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_hard9k1_20260812_v2`|
|context|`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|

唯一命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_hard9_source_1fab89eb_20260812_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

环境=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；8 shards映射GPU0–7。prepare和真实K5/new20 truth-free smoke通过后才允许shards。

## 5.健康门、artifact与后续

只按wrong hash/CWD、覆盖风险、协议/安全违规、launcher异常或两个distinct outer同一prediction前指纹停止；不得按任何性能值停止。健康完成期望：10 job receipts；正式before/after prediction/COMMIT/fit/resource/execution各20；scores 10；summaries 8；完整取回source/output/logs及10 truth sidecars。

primary只在artifact闭环后运行冻结analyzer。若八项严格Pareto及稳定/资源门全部通过，立即发布完整target125；否则记录失败指标并停止该路线。

## 6.运行与分析结果

待sole runner回填。
