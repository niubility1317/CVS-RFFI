# AJLR 独立分析阶段日志顺序修复

日期：2026-10-01。状态：`ANALYSIS_R02_RUNNING_VERIFIED`。完整训练 run `20261001-phase2-d92-anchor-joint-support-m2-r01` 已完成四 row、160 parent，全部训练进程退出；当前还没有性能汇总。本文只记录分析层修复，不改变方法或训练产物。

首次分析 release `d92_anchor_joint_analysis_20261001_r01` 使用已推送 commit `889fa1332ae91f0104a805348fee87f91db39709`。独立进程证据确认 PID386856 已退出，没有 summary、analysis_execution 或输出目录。原始错误为 `Stage stream mismatch`；日志保留在该 release 的 `analysis.log`，本地证据为本 run 的 `evidence/analysis_failure_r01.json` 与 `analysis_readback_1790795703741209900.json`。本次分析动作判定为 **FAILED，失败层为汇总器阶段流重建**。

只读实际 `fit_stages.jsonl` 的 event/state/scope/fold/trial/class_count 字段，已独立确认真实顺序为 `BASE B0/C0 → prep B → fit B → prep C → fit C`。证据 `analysis_stage_order_root_cause.json` 不含 query、源域或外层分数。入口代码按每阶段准备后立即拟合，汇总器却先添加所有 preparation，再添加全部 candidate stage。训练 runtime 与首次分析之间的这三份核心/入口/汇总代码没有变更，因此问题不是训练版本漂移。原仓库 summary fixture 只消费重建日志进行资源汇总，没有验证实际 callback 流，未覆盖此边界。

修复位于 [summarize_d92_anchor_joint_probe.py](../tools/summarize_d92_anchor_joint_probe.py)：逐对验证并追加 preparation 与对应 fit，额外核对 preparation_ref。完整内容与严格顺序检查 `next(stages)==compact_event(...)` 保留。新增 [实际 callback 回归检查](../tests/test_summarize_d92_anchor_joint_probe.py) 从真实合成入口即时保存日志，比较全部字段、坐标与顺序，并明确每条新增类路径的六个事件。测试文件 6 项通过，耗时 65.44 s，证据 `.codex_tmp/pytest_utf8_1790796368297854100`。没有修改核心数学、入口、预算、数据、旧产物、分数或评价公式，也没有重复无关核心检查。

下一步预登记分析 release 为 `d92_anchor_joint_analysis_20261001_r02`，root 是唯一 launch owner。先提交并推送上述修复，再用该版本复核同一四 row 的完整合法 support 产物；r01 目录、tar、错误日志和退出进程证据保留。原输出路径尚不存在，新输出继续禁止覆盖。实测结果以完整独立 summary 和 analysis_execution 为准，不以返回码、修复检查或数学证明代替性能证据。

ANALYSIS_R02_RUNNING/VERIFIED：使用阶段顺序修复的独立分析已单次启动，PID394399/argv/cwd独立读回匹配，analysis commit fc5cd282f93e2b7cc1a2fcd686b4217fb4abfe4a。本地等待handle99201有效，完整summary尚未产生。四row/160训练已完成；r01失败证据保留，不重复训练或任何analysis启动，目标ACTIVE。
