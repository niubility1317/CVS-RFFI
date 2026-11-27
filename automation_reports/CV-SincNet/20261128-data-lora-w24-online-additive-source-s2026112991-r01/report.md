# LoRa 同一源波形的独立加性噪声观测

现有 Practical 源库对每个物理包固定了一次传播和噪声实现。为单独检验训练时反复遇到同一噪声的影响，本次构建允许逐批重抽加性噪声的源库。旧库的 25,200 个合法 U_s 物理包、接收机、场景分配、传播、接收机作用及每包相位轨迹均按原种子重放。数据只用于恢复训练。 [原 Practical 配对登记](../20261127-phase1-lora-w24-native-practical-supervision-s2026112991-r01/report.md)

Full post-sync/no-EQ 路线的模拟器可返回加性噪声作用的直通和镜像算子。缓存同一物理包的加噪前波形 `s`，以及吸收原噪声标准差的两个系数 `a`、`b`。每个训练样本可重抽独立单位复高斯噪声 `z`，构造 `s+a*z+b*z.conj()`。按原完整含滤波延迟的块重新计算 AGC，再裁回原生 8,192 点。受扰输入与同条 `s` 监督使用同一个 AGC。算子与模拟器噪声只在源侧组成训练配对。

首步用五台源接收机、六种 Practical 视图各一包，共 30 个 U_s 样本。按旧噪声流从算子重建新增噪声并核对相对误差，在原生裁剪处和旧配对库逐样点比对。此外重抽两份独立噪声，报告复相关幅度，验证支持范围和 AGC 一致性。初始化和复现失败时停止并保留输出。之后只构建该源 U_s 库，记录 `source-smoke.json`、`operators.npy`、`agc-limits.npy`、完成标记与日志。W24、TX、源 V 和目标读取均为零。

N607 使用普通项目账号、原源上下文 Python 和模拟器环境，在独占发布目录执行。CPU 四个低优先级重放进程，检查本次实际磁盘空间和源 U 角色。命令为 `python -m experiments.pamf_lora_main.w24_online_additive_source publish --repo . --context <原 Practical 源上下文> --ssh-config <N607 配置> --output <独占本地发布目录>`。执行代码以本次登记提交为准。输出 `runs/20261128-data-lora-w24-online-additive-source-s2026112991-r01`。训练和正式测试队列各自独立运行。该源库不产生识别分数。 [生成入口](../../../experiments/pamf_lora_main/w24_online_additive_source.py)
