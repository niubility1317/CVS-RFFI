# Tweak配置可移植性复现实验V2修复发布

- run_id：`tweak_config2_portability_20260908_v2`
- 当前状态：`REMOTE_VERIFIED_READY_TO_LAUNCH`
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
