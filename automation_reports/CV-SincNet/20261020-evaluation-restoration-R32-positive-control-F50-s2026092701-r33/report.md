# R33：R32 配对增强 U-Net 对照的条件正式补测

状态：LOCAL_VERIFIED，等待 R32 完整结束；R33 尚未发布。补测条件已在 R32 源域最终预测和新正式结果前固定。方法 commit `4d8a51c522ad7c1e40bdbe109a52eba5e0972ba9` 已推送并独立核实；27 项聚焦检查、CLI help、一次独立 P0/P1 通过。实际最终权重尚在训练，不宣称已完成 checkpoint smoke 或正式测试。


本规则在 R32 尚未完成源域预测、尚未访问正式测试前固定。R33 不训练任何网络，只补测源域有正收益的 `pair_aug_control`。R32 的两行训练、原资格判断及原正式测试计划保持不变。R33 是独立 run，不能覆盖 R32。

## 源域条件

等 R32 全部完成且所属控制器、训练和预测进程退出后，读取已完成的同一源域 1,800 个物理样本、同一冻结 F50（seed 2026092701）及默认头的七视图结果。对照固定为源域选出的 R17 `unroll_unshared`，不依据任何正式测试结果选择父权重。

只有以下两项同时成立，R33 才访问正式测试：

1. 六个受损场景等权平均准确率相对 R17 严格提升。实际以六张混淆矩阵对角线正确数的差之和大于 0 判断，避免浮点平局误判。
2. clean 准确率相对原始 clean IQ 经同一 F50 的准确率，下降不超过 0.1 个百分点。

这沿用用户“只测源域有正收益 U-Net”的补测范围。很小的平均收益也可能满足条件，不宣称统计显著，不改写 R32 或 R17 的原完整资格结论。源域不满足时，保留负结果，正式测试为 0。

## 固定模型和测试

仅加载 R32 原 `pair_aug_control` 完整 E20 / 4,940 更新 checkpoint，沿用 R32 严格 schema、完整 R02/R07 来源和原物理数据契约检查。模型为原三步不共享 U-Net，CG 2/2/2。恢复训练未读 TX 标签、F 权重或 F 输出；R33 不进行任何优化、重训分类头或目标自适应。

资格满足后，启动器首步用两个合法 L_s IQ、实际完整 checkpoint、原 F50/default head 做无 query smoke，通过即继续。复用原 R05 的 168,000 个物理样本和接收 IQ，固定 clean、high/mid/low suburban、high/mid/low urban 共七视图。一条 IQ 进入恢复器，一条恢复 IQ 进入原 F，由原 F 统一派生 FFT、PA、时间和统计分支。

七视图 IQ、特征、默认头 logits 及物理 ID 全部固定且有效后，由单独 CPU 进程连接 truth。评分包含准确率、Macro-F1、修复/损伤、IQ/FFT 误差及 TX/RX/day 分组。性能验收沿用 low urban ≥70%、其他五个受损场景距原 raw clean ≤5 个百分点，恢复 clean 损伤另报。相同冻结 F 的识别收益仍不能单独证明每项恢复细节属于输入 TX。

## 执行和产物

执行模块为 `experiments.pamf_response_control_formal.runner`。独立不可覆盖 run：`20261020-evaluation-restoration-R32-positive-control-F50-s2026092701-r33`。精确 commit、命令、环境/CWD、路径和逐行矩阵登记在对应 `experiment.json`。

N607 普通账户；沿用 GPU 池 0/1/5/7，最多一个本任务 GPU worker，并使用既有 `receiver_residual_capacity4.lock`、每卡全部 owner 少于两个进程和至少 12 GB 空闲条件。必须先等 R32 所属进程退出，不抢占、停止或重启其他任务。

预期产物为 `launch.json`、`resolved_config.json`、`source_decision.json`、`complete.json`；资格满足时另有 `native_map_prior/final.pt`、`smoke.json`、`worker.json`、七视图 `target_confirmation/native_map_prior/prediction`、独立评分 `summary.json`、`goal_assessment.json` 和 `target_complete.json`。

父权重/权限/输入错误、输出冲突、非有限值、不可运行或无法合法预测属于技术失败；保留产物、停止本次新增启动，不自动重试，不影响健康任务。已有源域和正式数据句柄复用，不重新验证数据或增加哈希链。本基准已经暴露，结果是条件补测，不是新的盲测；正式结果不反馈到 R32 或后续候选排序。


精确命令、环境、路径和逐行矩阵见[登记](experiment.json)。


## 用户范围更新：由 R36 替代

2026-10-20 用户新范围：后续网络每版均做正式测试，源域无收益也不能取消。R33 尚未发布，状态 SUPERSEDED_USER_SCOPE；由 R36 固定测试 R32 两版。保留本报告原条件规则、实现与登记作为历史，不启动 R33，不改健康 R32。
