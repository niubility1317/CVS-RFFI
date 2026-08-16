# D92 E0 FULL CCOC Hard9+K1 v7实验发布报告

状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`

## 1.身份、目标与最小修复

|字段|值|
|---|---|
|证据标签|`CCOC-16`|
|run ID|`d92_e0_full_ccoc_hard9k1_20260817_v7`|
|时间|2026-08-17|
|操作方|CCOC Hard9 v7发布代理|
|本地工作树|`E:/type10-7/code/snapshots/d92_125wt`|
|运行时代码提交|`386c659d`|
|目标|执行冻结的9个performance outer和1个K1 liveness outer，取得CCOC相对同排E0的truth-last证据|
|假设|CCOC同时改善旧类、新类与遗忘指标，注册开销满足放宽后的hard门，query实时推理面与E0一致|
|比较对象|`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`对同排`E0_FULL_ONLY`|

v7将已完成的CCOC core峰值优化提交`1429a496739dfadaf169b83ddf86b3b831f174d5`纳入正式runtime source lock。method lock精确绑定core Git blob `abb45a15514c8c5758e0cececb930a76d27a29b8`和实际字节SHA256 `6f87d4eb041ba8874182a46eb3f2a76dc3f2f075a6692ee217f19bcd2f8ff331`；49成员归档包含该core字节。本发布不再改动科学代码、方法、9+1矩阵、阈值、query门或truth/prediction闭包。

## 2.本地发布物与验证

|项目|路径或结果|
|---|---|
|代码/config/tests提交|`386c659d`；v7 runtime source lock身份|
|配置|`configs/stage2_d92_ccoc_hard9_k1_v7.json`；相对v6仅`runtime.output_root`与scientific/core source lock必要变化|
|归档|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v7/runtime/d92_ccoc_hard9_k1_source_386c659d_20260817_v7.tar.gz`|
|归档闭包|48个实际源文件+1份source manifest；不含数据、checkpoint、truth、测试、文档和G0 runner/core|
|归档SHA256|`17abc626c49d669f12aceaa9e2cdf3deaa766eef1f5d06b700d66db59d500a3d`|
|归档大小|310047B|
|启动脚本SHA256|`a2d4292d534aa4484df1f14c672b787a58e09b01360604242517ae4a09069426`|
|外部镜像|`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v7/`|

core现有测试已随`1429a496`通过；v7只运行release单测、精确归档导入与runner/analyzer help、archive-mode prepare、`py_compile`、JSON、`bash -n`、tar安全、source manifest与外部镜像一致性。不增加review或额外gate。

## 3.N607预登记

|字段|值|
|---|---|
|远端project|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_386c659d_20260817_v7.tar.gz`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_386c659d_20260817_v7`|
|远端driver|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v7.sh`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_20260817_v7`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_hard9k1_20260817_v7`|
|本地取回根|`E:/type10-7/local_artifacts/d92_e0_full_ccoc_hard9k1_20260817_v7`|
|Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|启动CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs`|
|PID|启动后由sole runner记录|
|GPU|smoke=`GPU0`；shard0–7分别映射`GPU0–7`|

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v7.sh >./d92_e0_full_ccoc_hard9k1_20260817_v7.launch.out 2>./d92_e0_full_ccoc_hard9k1_20260817_v7.launch.err </dev/null &
```

顺序严格为`prepare→truth-free smoke→8 shards`，不自动运行analyzer，不允许同run重试。v7 archive、source、driver、output、log和retrieval路径必须在创建前不存在。

## 4.冻结矩阵、资源、推理与停止边界

- 协议为`p2_min_v1`，复用`VALIDATED_ONCE`数据；三scene、9个performance+1个K1、10 jobs、30 scene receipts、8 shards均不变。
- `candidate_peak_hard_max_bytes=1048576`；`candidate_peak_target_max_bytes=524288`；wall hard≤150ms、ratio≤1.50；wall P90 target≤120ms、ratio P90 target≤1.25。
- 实时推理不放宽：query MAC和永久state逐scene与同排E0精确一致；query truth/fit/update/selection/role/quota/global reassignment全部为false。
- 八项性能方向、幅度、稳定性和唯一裁决不变；K1不进入性能裁决。
- 技术停止只允许协议/安全错误、覆盖风险、零prediction或两个不同outer在prediction前出现同一确定性异常；不得读取性能决定停止。

预期artifact为matrix manifest、truth-free smoke receipt、10份job receipt、30份scene closure、8份shard summary和完整日志。当前仅证明本地发布面可执行，不证明N607落地、Hard9健康或性能结论。v1至v6本地、远端与外部证据保持不变；未执行SSH、SCP、N607启动或analyzer。只有`ADVANCE_TO_TARGET125_CANDIDATE`才允许新建Target125发布。
