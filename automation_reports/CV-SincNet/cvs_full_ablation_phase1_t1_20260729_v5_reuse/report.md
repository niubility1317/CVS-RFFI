# Phase1 T1全消融v5复用发布报告

## 基本信息

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|时间|2026-07-29|
|操作方|Codex主代理；N607唯一runner：`/root/phase1_t1_n607_runner`|
|目标|完成Phase1 T1六臂×五配对种子的30行矩阵，同时复用v3已完整产物，避免重复训练和重复数据审计|
|当前状态|`LOCAL_VERIFIED / WAITING_INDEPENDENT_REVIEW`|

## 假设与比较

比较`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`，每臂使用种子`7281101–7281105`，训练比例`0.07/0.63/0.30`，每个新训练行200epochs。旧行与新行可来自不同发布批次；报告中保留`direct_reuse/reexport_only/new_train`来源标签，不把复用行声明为本次重新训练。

## 复用与缺口

|类别|数量|范围|
|---|---:|---|
|直接复用|10|`P1-SUP`五种子、`P1-A0`五种子|
|只补导出|1|`P1-B0__train_seed_7281101`，复用E200 source-val checkpoint|
|新训练|19|`P1-FULL`五行、`P1-C0`五行、`P1-D0`五行、`P1-B0`种子7281102–7281105|
|安全续训|0|中断行不续训|

## 启动前输出闭环

- 每个新训练行必须同时具有非空且可加载的checkpoint和prototype PT、可解析且非空的prototype JSON、resource summary、独立`frozen_phase1_heldout_eval.json`、terminal和completion receipt。
- prototype路径必须严格位于本行输出目录，禁止跨行引用。
- terminal、receipt和真实子进程退出码必须同时为0。
- PID文件使用独占创建；run root、log root和`launch.out`均不可覆盖。
- `runner_summary.json`必须按30行汇总并分列`direct_reuse/reexport_only/new_train`。
- 按用户指示不重新读取整份WiSig文件做SHA256审计，只确认文件存在并沿用已登记的数据标识。

## 本地实现与验证

|文件|用途|
|---|---|
|`code/SSDG/train_ssdg.py`|将独立held-out结果路径和摘要绑定到terminal/completion|
|`code/scripts/run_full_ablation_phase1_t1.py`|复用调度、补导出调度、产物可加载/非空/本行路径验证、19个新训练并发|
|`code/scripts/reexport_phase1_prototypes.py`|从v3的B0 E200 checkpoint仅补prototype导出|
|`code/scripts/seal_full_ablation_phase1_plan.py`|将补导出脚本和复用配置纳入release|
|`configs/full_ablation_20260728/phase1_t1_reuse_v5.json`|冻结10+1+19复用矩阵|
|`tests/test_run_full_ablation_phase1_t1.py`|复用、补导出、held-out和损坏产物故障注入|

验证结果：`55 passed,1 skipped`（含真实B0 checkpoint导出）；sealer/runner/factory定向测试`51 passed`；`py_compile`和`git diff --check`通过。

## N607发布参数

|字段|值|
|---|---|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|工作目录|待seal后填写|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|launch log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v5_reuse.launch.out`|
|GPU调度|8张GPU，每张最多2个进程，共16槽|
|预期执行|19个完整训练+1个checkpoint-only导出；10行直接复用|
|成功条件|30/30有效闭合，19个新训练成功、1个补导出成功、10个复用行验证通过、无P0、无重复异常指纹系统停机|
|系统停机|P0协议/输出覆盖风险，或至少两个不同新执行行在生成prototype前出现同一确定性异常指纹|

## 风险与完成后检查

- 复用行和新训练行不是同一发布批次，最终统计必须保留来源列。
- B0补导出必须使用当前修复后的endpoint exporter，原v3失败terminal不能当作完整训练receipt；使用独立reexport receipt闭合。
- 完成后检查30行同一行指标、最佳epoch/checkpoint、held-out指标、prototype、资源、异常指纹和GPU释放，再推进设计报告下一批实验。
