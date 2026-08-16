# D92 E0 FULL CCOC Hard9+K1 v4实验发布报告

状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`

## 1.身份、目标与最小修复

|字段|值|
|---|---|
|证据标签|`CCOC-16`|
|run ID|`d92_e0_full_ccoc_hard9k1_20260817_v4`|
|时间|2026-08-17|
|操作方|CCOC Hard9 v4发布代理|
|本地工作树|`E:/type10-7/code/snapshots/d92_125wt`|
|运行时代码提交|`fbe4e03a`|
|目标|执行冻结的9个performance outer和1个K1 liveness outer，取得CCOC相对同排E0的truth-last证据|
|假设|CCOC同时改善旧类、新类与遗忘指标，注册开销满足放宽后的hard门，query实时推理面与E0一致|
|比较对象|`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`对同排`E0_FULL_ONLY`|

v3在prepare阶段读取当前同排E0 resource fit-audit时，因该合法资源基线文件的当前字节SHA与旧预登记SHA不同而被阻断。E0 fit-audit只提供逐scene wall、peak、query MAC和state资源基线，不包含truth或性能裁决。v4移除旧SHA相等门，改为检查当前文件存在、非symlink、路径outer身份、三scene、E0 arm/candidate、K/registered class、resource schema及所需数值字段；当前observed SHA与实际scene资源写入immutable manifest，并写入prepare/smoke/job receipt用于追溯。truth、prediction/COMMIT/fit/score闭包未放宽。

## 2.本地发布物与验证

|项目|路径或结果|
|---|---|
|代码/config/tests提交|`fbe4e03a`；E0 resource observed-SHA及v4身份|
|配置|`configs/stage2_d92_ccoc_hard9_k1_v4.json`；相对v3仅`runtime.output_root`变化|
|归档|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v4/runtime/d92_ccoc_hard9_k1_source_fbe4e03a_20260817_v4.tar.gz`|
|归档闭包|48个实际源文件+1份source manifest；不含数据、checkpoint、truth、测试、文档和G0 runner/core|
|归档SHA256|`2c8d02122b8092e4f6ebbeac280fcc155fd7e3c678627277f819493231cef807`|
|归档大小|308983B|
|启动脚本SHA256|`9d2d30ed6069c78b3978f6e4721fad40ebd19ac45a6361e18fb9c6914f5807be`|
|外部镜像|`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v4/`|

TDD RED精确复现`E0 resource fit-audit SHA drift`；GREEN证明合法内容漂移通过，arm/scene/resource schema/K/class/query MAC/state/wall/peak篡改仍拒绝。验证范围仅包含matrix/runner受影响测试、精确归档导入与runner/analyzer help、archive-mode prepare、`py_compile`、JSON、`bash -n`、tar安全和source manifest。不增加独立审查或额外SHA/signature门。

## 3.N607预登记

|字段|值|
|---|---|
|远端project|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_fbe4e03a_20260817_v4.tar.gz`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_fbe4e03a_20260817_v4`|
|远端driver|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v4.sh`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_20260817_v4`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_hard9k1_20260817_v4`|
|本地取回根|`E:/type10-7/local_artifacts/d92_e0_full_ccoc_hard9k1_20260817_v4`|
|Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|启动CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs`|
|PID|启动后由sole runner记录|
|GPU|smoke=`GPU0`；shard0–7分别映射`GPU0–7`|

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v4.sh >./d92_e0_full_ccoc_hard9k1_20260817_v4.launch.out 2>./d92_e0_full_ccoc_hard9k1_20260817_v4.launch.err </dev/null &
```

顺序严格为`prepare→truth-free smoke→8 shards`，不自动运行analyzer，不允许同run重试。v4 archive、source、driver、output、log和retrieval路径必须在创建前不存在。

## 4.冻结矩阵、资源、推理与停止边界

- 协议为`p2_min_v1`，复用`VALIDATED_ONCE`数据；三scene、9个performance+1个K1、10 jobs、30 scene receipts、8 shards均不变。
- `candidate_peak_hard_max_bytes=1048576`；`candidate_peak_target_max_bytes=524288`；wall hard≤150ms、ratio≤1.50；wall P90 target≤120ms、ratio P90 target≤1.25。
- 实时推理不放宽：query MAC和永久state逐scene与同排E0精确一致；query truth/fit/update/selection/role/quota/global reassignment全部为false。
- 八项性能方向、幅度、稳定性和唯一裁决不变；K1不进入性能裁决。
- 技术停止只允许协议/安全错误、覆盖风险、零prediction或两个不同outer在prediction前出现同一确定性异常；不得读取性能决定停止。

预期artifact为matrix manifest、truth-free smoke receipt、10份job receipt、30份scene closure、8份shard summary和完整日志。当前仅证明本地发布面可执行，不证明N607落地、Hard9健康或性能结论。v1/v2/v3本地、远端与外部证据保持不变；未执行SSH、SCP、N607启动或analyzer。只有`ADVANCE_TO_TARGET125_CANDIDATE`才允许新建Target125发布。
