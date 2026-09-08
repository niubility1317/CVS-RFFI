# Tweak配置可移植性复现实验V2修复发布

- run_id：`tweak_config2_portability_20260908_v2`
- 当前状态：`COMPLETED_ARTIFACTS_VERIFIED_SINGLE_SEED_DESCRIPTIVE`
- 上一运行：`tweak_config2_portability_20260908_v1`在N607的NumPy2.2.5/PyTorch2.1数组接口处确定性失败；保留其日志和空输出根，不复用。
- 修复与代码提交：`c266304406078bc4c1391a46b2a9b20396b6c0d8`以`torch.frombuffer(memoryview(...))`替代不兼容的`torch.from_numpy`路径，并新增先失败后通过的回归测试。
- 本地验证：23项Tweak聚焦测试及两个模块编译检查通过。
- 发布核验：N607源代码归档SHA-256=`c9dbc12722ca1cdb93639b819f3af8852ff198e988c1ce670955b60eb6d1f353`，与本地一致；在N607真实Config2官方数据上，10个记录读取、`[64,2,128]→[64,12]`前向及有限值检查均通过。
- 数据与矩阵：沿用已验证的Config1—4、各设备ID1—10共80文件，仅执行Config2训练的图13b单域校准4×4闭集矩阵和图14四配置联合校准4行闭集结果；不运行消融或硬件可移植性实验。
- 固定方法：`[2,128]`原始IQ、12维embedding、batch-hard triplet、margin=0.1、SGD momentum=0.9、batch=64、五学习率网格、100epoch、75/25切分、每类`N=11,718`校准帧、`M=10`帧聚合。
- N607代码目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/tweak_config2_portability_20260908_v2/source`
- N607输出目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/tweak_config2_portability_20260908_v2/official_config2_full`
- N607日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/tweak_config2_portability_20260908_v2/train.log`
- 启动规则：用户已明确允许每卡增加实验；启动前仍记录实际GPU、PID、命令、日志增长和输出根。仅本run的确定性技术故障可触发修复，低性能不停止。
- 启动记录：于GPU0启动，PID=`3791806`，完整命令未传入smoke限制。启动后真实进程、命令行、GPU绑定和唯一输出根均已读回一致。
- 完成核验：N607日志以`{"event":"complete","status":"PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS"}`结束；绑定PID`3791806`已退出。`best_checkpoint.pt`为8,903,716字节（07:06:59 CST），`results.json`为82,096字节（07:08:00 CST），均在唯一V2输出根内。独立读回`results.json`成功，包含训练、checkpoint-no-query smoke、图13b的16个单域校准结果和图14的4个多配置联合校准结果。
- 训练闭合：五个学习率均完成100epoch；依据预登记的最低完整epoch平均batch-hard训练损失选择`learning_rate=0.01`、`best_epoch=100`、`best_mean_training_loss=0.10000042192869861`。这只说明该单seed、未公开默认值的实现内选择，不能作为跨seed稳健性或方法晋级证据。
- 图13b单域校准闭集准确率（行=校准配置，列=测试配置；每格39,060个M=10聚合决策）：

| 校准\\测试 | Config1 | Config2 | Config3 | Config4 |
|---|---:|---:|---:|---:|
| Config1 | 33.228% | 6.966% | 12.983% | 6.528% |
| Config2 | 11.828% | 11.677% | 11.813% | 10.228% |
| Config3 | 12.486% | 9.462% | 18.451% | 7.857% |
| Config4 | 0.584% | 10.189% | 6.700% | 24.191% |

- 图14四配置联合校准闭集准确率（每行39,060个M=10聚合决策）：Config1为29.662%，Config2为8.697%，Config3为15.543%，Config4为12.448%。它们是一次完整复现的描述性观测，不与论文图中数值作无条件等同，也不构成外部泛化、统计显著性或默认方法选择结论。
