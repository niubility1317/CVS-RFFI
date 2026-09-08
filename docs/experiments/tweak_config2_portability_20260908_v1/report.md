+# Tweak配置可移植性复现实验预登记

- run_id：`tweak_config2_portability_20260908_v1`
- 当前状态：`REMOTE_READY_GPU_PENDING`
- 唯一launch owner：Codex主Agent
- 代码提交：`b3483331a9862e87270203f66d0caa0d043be152`
- 代码分支：`codex/two-paper-rffi-reproduction-20260827`
- 本地工作目录：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\two-paper-rffi-reproduction-20260827`
- Python环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`
- 数据源：OSU公开LoRa数据集的`Diff_Configurations_Setup/Config1...Config4`。本次封装并已上传40个`.dat`与40个`.sigmf-meta`（Config1—4、各设备ID1—10），每个`.dat`为20,000,000个`cf32`复IQ样本。
- 数据边界：仅使用Config1—4、设备ID1—10；跨接收机数据尚未下载完成，因此不执行硬件可移植性实验。
- 非消融主实验：
  1. 论文图13b对应的Config2训练、4个单域校准×4个测试配置闭集准确率矩阵；
  2. 论文图14对应的同一Config2模型、4配置联合校准、4个测试配置闭集准确率。
- 固定方法：`[2,128]`原始IQ、12维embedding、batch-hard triplet、margin=0.1、SGD momentum=0.9、batch=64、学习率网格`[1e-2,1e-3,1e-4,1e-5,1e-6]`、100epoch、75/25切分、每类`N=11,718`校准帧、`M=10`帧聚合。未公开默认值记录于`strict_method.json`。
- N607代码发布目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/tweak_config2_portability_20260908_v1/source`
- N607数据目录：`/home/szu2070436088/2510044040/CV-SincNet/datasets/tweak_official_lora_configurations_20260908`
- N607输出目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/tweak_config2_portability_20260908_v1`
- N607日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/tweak_config2_portability_20260908_v1/train.log`
- 远端发布核验：代码归档SHA-256=`3f9df83655449025032a94d1c6f8155a7492bf8f58165dd5847cc76ebee3ec50`；数据归档SHA-256=`967589f0257c168ba6a4095f50718bb0a7628845b464b68894e5668d77380da9`。二者均已独立读回一致；源代码和80个本次所需数据文件已解包，CUDA环境编译检查通过。
- GPU：在代码与数据落地后，仅选择已有训练进程少于2个且显存满足预检的GPU；最近一次预检中GPU0已有3个计算进程，GPU1—7各有2个，故当前8张GPU均无合规空位，尚未启动。
- 计划命令：`python -m paper_reproduction.gaskin_tweak_2023.official_lora_experiment --data-root <N607数据目录>/Diff_Configurations_Setup --output-dir <N607输出目录> --device cuda:<预检GPU>`
- 预期产物：`best_checkpoint.pt`、`results.json`、逐epoch训练日志、单域4×4闭集准确率和多配置联合校准的4行闭集准确率。
- 系统技术停止规则：仅当本run绑定的进程出现确定性启动/数据读取/训练异常、输出根冲突、checkpoint或`results.json`无法闭合时，停止该run自己的进程树并保留partial artifacts；低性能不停止。
- 已完成验证：本地官方`cf32`读取、64×`[2,128]`输入、12维前向、batch-hard反传、22项Tweak聚焦测试和两份模块编译检查均通过；N607的CUDA可用、代码模块编译通过。该远端环境未安装`pytest`，因此未在远端重复执行pytest套件。
