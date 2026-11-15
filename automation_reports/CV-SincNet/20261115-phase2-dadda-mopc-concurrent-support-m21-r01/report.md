# DADDA→MoPC 同一 B 的并发支持诊断

对 affine 接续 run 的 `0086-hybrid-dadda_sda`，读取原保存 B-K1、原 seed 和相同 clean K1/N20 支持，运行仓库原公式和原优化更新。每次进程重复三次，共 21 个诊断任务；与正式失败时多个 worker 并行接近，但不读取之前 query，不能重建原任务完整执行时序。仅用于观察是否再发生真正非有限，不产生预测、不接 truth。原输出、claim 和失败隔离保持。

代码提交 `24219ae4472c2b69291a53893e911bd576d1a44e`，在 N607 的 `releases/20261115-phase2-clean-health-audit-cpu4-r01-release` 运行 `python -m experiments.pamf_phase2_practical.clean_mopc_concurrent_diagnostic --runtime preparations/20261115-phase2-mopc-clean-affine-resume-m21-r01/runtime/0.json --job preparations/20261115-phase2-mopc-clean-affine-resume-m21-r01/jobs/0086-hybrid-dadda_sda.json --output runs/20261115-phase2-dadda-mopc-concurrent-support-m21-r01 --k 1 --n 20`（实际命令的输入输出使用项目绝对路径）。物理 GPU 5、6、7，每卡本诊断最多 7 个 worker，调度前空余至少 6000 MiB，预留 3000 MiB，其他 owner 不受干预。出现非有限或系统性异常即停止新派发、排空已启动诊断并留存记录。输出是独立 owner、state、完成摘要与支持诊断日志，不访问 query/truth。

此前该原始 B 的单进程原公式及两个裁剪诊断各完成 80 更新；均未复现，不证明根因已解决。该并发实验仅是受控定位，同样不能证明原正式实验已修复。

结果：运行在物理 GPU5、7 的 10 个已启动进程均在首次注册发生真实非有限；余下 11 项按停止规则未派发，所有进程已排空。其中 `diagnostic-03` 在 C 第 3 阶段第 8 epoch、iteration 8 出现非有限损失和梯度，与正式失败位置一致。该结果说明关闭逐层诊断观测后可复现，不能单凭并发归因或宣称根因已证实。GPU6 因其他 owner 占用自动跳过。无 query、truth、预测或旧产物变更。
