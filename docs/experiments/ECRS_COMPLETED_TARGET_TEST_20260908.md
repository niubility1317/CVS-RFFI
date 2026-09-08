# 已完成ECRS候选的冻结target测试

run_id：`phase1_ecrs_completed_target_test_20260908_r1`。状态：`LOCAL_VERIFIED`，发布后追加运行/结果证据。
用户在询问完整实验数据后，明确要求“跑完的进行测试”。本次仅测试已完成的B0、B3c、B4，不训练、不适配、不重选checkpoint。

## 固定输入与声明

以[冻结清单](ECRS_COMPLETED_TARGET_TEST_20260908.json)为实际输入：B0 source最佳E199，B3c/B4 source最佳E195，均为各自best.pth。三行已验证训练E200、exit0、SOURCE_SCREEN_COMPLETE和完整source四场景结果。B2-V1、B2、B3a、B3b仍在训练，不在当前测试清单。

这是Phase1已见TX闭集target检验：ManySig equalized=1，TX0—5，RX0/2/5/7/9/10/11，day0—3，每场景168000个物理样本。固定clean及三个LEO_WEAK场景；raw为预先声明的决策路径，response/fused仅作分路径诊断，不事后切换最优头。该测试不声明new/unknown、Stage2适配或真实在轨性能。

单一target索引内physical_sample_id为sample:k，三行/四场景复用完全相同索引；不与source同名sample:k连接。预处理固定center crop到256、RMS normalize、不减均值。每个LEO场景的GPU generator种子为2027+scenario_index×1009，三行一致，batch=256、shuffle=False；这是本次统一target评估流，不声称与历史named-loader子集的逐样本LEO扰动完全一致。

## 执行与安全边界

独立脚本`code/scripts/eval_ecrs_completed_target_20260908.py`先验证全部source选择与checkpoint epoch，严格重载真实模型并无query smoke，然后才加载target。预测dataset返回dummy label并仅保留ID/RX/day；`evaluate_revision_paths(source_metrics=False)`不读取TX标签、不向模型传标签、不输出truth，保持eval/no_grad，不更新BN/EMA/参数。四场景×三行全部预测完整关闭后，独立scorer子进程才读取truth文件按ID连接，核对无重复、全覆盖并产生summary。

输出：项目根`runs/phase1_ecrs_completed_target_test_20260908_r1`，包含frozen_manifest、target_truth、三行四场景无truth JSONL、prediction_resources、独立target_test_summary。logs根同run_id保存stdout/PID。输出目录独占创建，保留partial，不覆盖已有结果、不自动重跑。预测与truth文件分开；不存在任何target结果→训练/选模调用路径。

Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。CWD：本次独立release根。使用GPU2做单进程顺序推理，训练/适配任务不停止或迁移；本任务没有optimizer，不增加每卡训练实验数。当前显存可容纳推理，仍需启动读回验证实际PID/GPU/日志。不得把共享负载下耗时当独占GPU基准。

```text
CUDA_VISIBLE_DEVICES=2 <python> -u <release>/code/scripts/eval_ecrs_completed_target_20260908.py predict --manifest <release>/docs/experiments/ECRS_COMPLETED_TARGET_TEST_20260908.json --output <project>/runs/phase1_ecrs_completed_target_test_20260908_r1 --device cuda:0
```

本地3项检查通过：truth-blind wrapper、独立scorer覆盖/泄漏拒绝、B0严格重载与原train.py双骨干输出一致。新入口独立P0/P1审查仅检查直接正确性，不扩展未知类测试权限。发布按本地Git/push读回、一次归档SHA比较、一次远端编译、实际smoke与进程读回执行。

技术失败只记录本次评估产物，不影响训练。完成必须有3×4×168000=2016000条唯一预测、三行四场景独立评分及checkpoint身份。测试结果只检验已冻结模型，不用于后续候选重排、调参或选择性重跑。
