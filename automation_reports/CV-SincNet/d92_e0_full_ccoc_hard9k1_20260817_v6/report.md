# D92 E0 FULL CCOC Hard9+K1 v6实验发布报告

状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`

## 1.身份、目标与最小修复

|字段|值|
|---|---|
|证据标签|`CCOC-16`|
|run ID|`d92_e0_full_ccoc_hard9k1_20260817_v6`|
|时间|2026-08-17|
|操作方|CCOC Hard9 v6发布代理|
|本地工作树|`E:/type10-7/code/snapshots/d92_125wt`|
|运行时代码提交|`9c22dc42`|
|目标|执行冻结的9个performance outer和1个K1 liveness outer，取得CCOC相对同排E0的truth-last证据|
|假设|CCOC同时改善旧类、新类与遗忘指标，注册开销满足放宽后的hard门，query实时推理面与E0一致|
|比较对象|`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`对同排`E0_FULL_ONLY`|

v5的真实after fit-audit只包含7个base query-zero字段和7个批准的`d92_e0d_ccoc_`镜像，两组均为false；generic `d92_e0d_`和raw `d92_ccoc_`别名按whitelist应当缺失。v6 validator仅严格要求base与批准镜像存在且为false，不再要求generic/raw别名存在；现有query whitelist仍负责拒绝未批准字段。query代码、truth/prediction闭包、方法、矩阵和资源门未改变。

## 2.本地发布物与验证

|项目|路径或结果|
|---|---|
|代码/config/tests提交|`9c22dc42`；query-zero validator及v6身份|
|配置|`configs/stage2_d92_ccoc_hard9_k1_v6.json`；相对v5仅`runtime.output_root`变化|
|归档|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v6/runtime/d92_ccoc_hard9_k1_source_9c22dc42_20260817_v6.tar.gz`|
|归档闭包|48个实际源文件+1份source manifest；不含数据、checkpoint、truth、测试、文档和G0 runner/core|
|归档SHA256|`f815686e92c4b5e00c79e28795713f22d85edd0accb6584610750fd863e76fd1`|
|归档大小|308770B|
|启动脚本SHA256|`5d936c859f937c2c546e93e775281df289369a5be70b8b0ef2da2dd5a7a7ca40`|
|外部镜像|`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v6/`|

TDD RED精确复现base+批准CCOC镜像全false但generic别名缺失时旧validator返回false；GREEN证明该真实结构通过，任一base/镜像缺失或为True均拒绝。验证范围仅包含matrix/runner受影响测试、精确归档导入与runner/analyzer help、archive-mode prepare、`py_compile`、JSON、`bash -n`、tar安全和source manifest。不增加独立审查或额外SHA/signature门。

## 3.N607预登记

|字段|值|
|---|---|
|远端project|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_9c22dc42_20260817_v6.tar.gz`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_9c22dc42_20260817_v6`|
|远端driver|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v6.sh`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_20260817_v6`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_hard9k1_20260817_v6`|
|本地取回根|`E:/type10-7/local_artifacts/d92_e0_full_ccoc_hard9k1_20260817_v6`|
|Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|启动CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs`|
|PID|启动后由sole runner记录|
|GPU|smoke=`GPU0`；shard0–7分别映射`GPU0–7`|

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v6.sh >./d92_e0_full_ccoc_hard9k1_20260817_v6.launch.out 2>./d92_e0_full_ccoc_hard9k1_20260817_v6.launch.err </dev/null &
```

顺序严格为`prepare→truth-free smoke→8 shards`，不自动运行analyzer，不允许同run重试。v6 archive、source、driver、output、log和retrieval路径必须在创建前不存在。

## 4.冻结矩阵、资源、推理与停止边界

- 协议为`p2_min_v1`，复用`VALIDATED_ONCE`数据；三scene、9个performance+1个K1、10 jobs、30 scene receipts、8 shards均不变。
- `candidate_peak_hard_max_bytes=1048576`；`candidate_peak_target_max_bytes=524288`；wall hard≤150ms、ratio≤1.50；wall P90 target≤120ms、ratio P90 target≤1.25。
- 实时推理不放宽：query MAC和永久state逐scene与同排E0精确一致；query truth/fit/update/selection/role/quota/global reassignment全部为false。
- 八项性能方向、幅度、稳定性和唯一裁决不变；K1不进入性能裁决。
- 技术停止只允许协议/安全错误、覆盖风险、零prediction或两个不同outer在prediction前出现同一确定性异常；不得读取性能决定停止。

预期artifact为matrix manifest、truth-free smoke receipt、10份job receipt、30份scene closure、8份shard summary和完整日志。当前仅证明本地发布面可执行，不证明N607落地、Hard9健康或性能结论。v1/v2/v3/v4/v5本地、远端与外部证据保持不变；未执行SSH、SCP、N607启动或analyzer。只有`ADVANCE_TO_TARGET125_CANDIDATE`才允许新建Target125发布。
