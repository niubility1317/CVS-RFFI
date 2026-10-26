# 三份额外源 U 成对 H/G 缓存

PLANNED。按 [固定计划](../../../docs/goals/PAMF_U_REALIZATION_SSL_20261026.md) 先构建数据支持，不训练或评估 F。m3 是三个缓存任务，不是三个学生种子；原联合目标未达成。

完整原 F0 U56700、原 VALIDATED_ONCE 物理顺序及同一冻结原 G-LU。只读原 seed01 E30 complete.json 配置元数据来定位 source cache/contract，不载入 F 权重或输出，不读 L、V、target 或 U TX 真值。U batch 标签全部 −1，tx_label_visible=False，metadata 不含 y。G 由现有 load_G 校验同契约，执行 InferenceEngine(reuse=True,graph=True,native=False)，FP32/TF32off。

固定 fresh1/2/3 namespace 为 PAMF/realization_diversity/U/fresh1/2/3；三景 lowSuburban、midUrban、lowUrban。保留 source_channel 的 residual/post_sync/noEQ/reflect/SNR30、fs25e6/fc2462e6、channelSeed20261016421、hardwareSeed20261016423 及 receiver/day hardware，仅改变信道随机实现。无额外白噪声、插值、虚拟 RX 或 G 后归一化。

每份成对保存 channel_U.npy/restored_U.npy，FP32 [3,56700,2,256]，物理键 eq_i/rx_i/day_i/sig_i/base_index 及 IDs；新 IQ float 字节共2090188800。旧 fixed H/G 缓存直接复用且不覆盖。三份均保留、不通过 F 评分或挑随机种子。

首先共享真实 G hidden-U2 eager/graph smoke（4逻辑G包），PASS立即三个缓存任务。各 worker 先独立 warm U2（共6逻辑包），随后 snapshot 全参数及全部 registered buffers，生成510300逻辑G包后核实冻结；H/G 输入、输出在 graph 复用处 clone。无 optimizer/G/F update、F forward、U truth/V/target。

N607普通账户、既有 CVS-RFFI Python/Torch2.1，各CPU2线程，无安装。CWD PROJECT/releases/20261026-diagnostic-PAMF-URealizations-cache-release-r01；命令 PYTHON -u -m experiments.pamf_channel_pl.u_realization_cache dispatch --config该release/runtime.json。不可覆盖 PROJECT/runs/20261026-diagnostic-PAMF-URealizations-cache-m3-r01。实际 code commit 记于发布 runtime/submit/完成记录，不另加 SHA 链。

GPU0..7按全局CUDA+pending<2、free≥10000MiB、最多3个本run worker准入，保护其他owner。一份archive一次传输比较、remote compile、独立PID/CWD/argv/GPU/log-growth检查。首技术/协议/非有限/coverage/state/smoke错误停止新派发，已启动健康任务继续；保留输出，不自动重提或干预他人。

3项聚焦负例/调度检查已通过；唯一一次独立P0/P1检查PASS，未发现P0/P1。仅代码、配置、登记和公共聚合入Git，IQ/逐包索引/runtime/日志留私有。本run不执行F源门槛或目标测试；缓存完成后按另行登记的四条匹配训练计划继续。
