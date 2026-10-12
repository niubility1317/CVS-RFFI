# 独立 P0/P1 发布审查

结论：**PASS；剩余 P0/P1 为 0。** 审查日期：2026-10-12。审查者：`/root/stage2_release_review`。

范围为 `20261012-phase1-nocfo-matched-engineering-manysig-m24-r01` 的实际训练、来源、预测、控制与收尾路径。审查者未参与本包实现；仅写本文件，未启动实验、操作远端、使用 GPU，未读取真实目标评分或预测文件。此结论确认发布正确性，不代表性能结果或远端执行已完成。

## 设计与执行

- 当前登记的 24 个 config 与 `design.config` 逐项相等：8 个 arm × 3 个 seed，随机原生学生，20 epoch／4440 更新，共 106560 更新。学生初始化可以在禁止 `torch.load` 时构建；无父权重、EMA、星地增强、Fishr 或辅助头。训练和预测均应用同一无显式 CFO 路径，参考接口 slot 27 恒零，内部 CFO 补偿保留。
- `teacher.py` 严格验证新纯 CE 无 CFO foundation 的 run、seed、完整 config、执行提交 `4aeef661c33cb140f5c750ab4718b863739fed65`、固定 final20／4440、合法源数据角色及空祖先。学生只用另外两个 seed 的固定概率均值。新缓存 schema 和 exact proof 拒绝旧教师；U 真值字段被拒绝。缓存只前向完整 U56700，教师参数、缓冲、RNG 与梯度不变。
- L/V 原始 IQ 与物理 ID 依既有源契约核对，U 与 L/V 不重叠；V 仅健康评估，不选模。前 444 步纯 L，随后 3996 步 U；所有配置完整执行固定预算。L_REPLAY 与 hidden mix 的额外 L CE 使用相同 ramp、权重和 clean dropout 重放；hidden mix 的特征与标签共用 lambda，供体来自不同 TX 的 L，单层交替。SAM 复用同样本、噪声和 dropout，完整目标两次反向，临时参数在 finally 中精确恢复后仅执行一次 AdamW 更新。
- 每行 source freeze 后立即在同一行执行链进行 clean168000 预测，再由独立 scorer 读取 truth。同 seed 新 CE prediction 只作固定评分对照，不初始化学生；其来源及预测完整性再次核对。24 行闭合、3 seed 配对和全部负结果保留。
- 发布使用不可变提交归档、新 run/release 与独占输出；UNKNOWN 不自动重发。共享容量统计 CUDA 所有者和启动前预约，最多每卡 3 个总进程且保留 12 GB 空闲；唯一 dispatcher 和每行 owner，失败保留产物且不干预其他健康任务。实际 configure 得到 64 个存在的归档路径，含新包与 foundation；新包直接跨包导入均有运输覆盖，远端 preflight/dispatch 入口绑定正确。

## 验证证据

已读回本 run 的 `evidence/nocfo_engineering_cuda.json` 与 `evidence/nocfo_engineering_local.json`，均为 PASS：8 臂原生 CUDA 目标、实际 L_REPLAY CE 等值、SAM 手工 AdamW 对照及异常恢复、source 入口，以及 24 配置预算和 8 臂 strict own-checkpoint roundtrip。审查期间未重复运行 GPU 检查。

独立执行临时合成收尾验证，调用真实 `final_report.finish`、collector 和报告构建：24 行、336 条主评分记录、96 条日期记录、42 条去重 CE 对照记录、11 个配对比较均闭合；最终报告、源报告、证据 JSON、原 experiment 状态和 events 正常生成。该测试只模拟已闭合源分析结果，源日志扫描函数被 mock；没有把它当作真实 106560 步训练验证，也未读取真实 IQ、truth 或目标结果。实际源日志扫描代码已审查其完整 4440 步／20 epoch、LR、计数、SAM 与 replay 语义核验。

本次未发现需要修复的 P0/P1；无额外审批或性能门槛。
