# 纯 F 冻结 E20：训练命名流与新信道拟合诊断

状态：PLANNED，尚未运行，没有新性能结果。运行 20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01。这是零更新源诊断，不是新的训练候选，不改变场景风险实验的固定晋级规则，也不把本诊断作为额外发布门槛。

固定问题：普通 As 的源 urban 误差，来自训练期观测未拟合，还是来自新信道及新物理样本泛化？四个 checkpoint 均为同 seed 2026092701 的纯 F 自身 FINAL20：从 20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01 复用 B0/B1，从 20261019-phase1-F-sceneRisk-e20-s2026092701-m2-r01 读取 BALANCED/RISK。严格校验自身 scratch20、原数据角色和默认头，不使用目标成绩选择权重。B0 没有看过这些 As，仅作参照。

固定抽样：从 B1 完整 L 消费序列的 E20，即第4219至4440步，按真实 step/slot 扫描。每 TX×RX×day cell 取最先出现的两个不同 physicalID，最多180，缺项如实保留、不补抽。只有核实属于合法 L6300 后才按原生 base index 读取 float32 raw。源 RX1/3/4/6/8、day1/2/3、split392005 不变；U TX、目标 IQ 和目标 truth 均不访问。

按原始 epoch/step/physicalID 命名流、原 Config/receiver 重建训练期 As。检查完整128包与独立单包的新建对象回放相等，以及 BALANCED/RISK 已记录的 scene_index 对齐。历史没有保留 As IQ，因此只称“训练期固定命名流确定性重建”，不声称独立历史字节核验。随后用两个新命名流为同一 L 生成三个原训练场景的新观测；比较时匹配同一 physicalID 和原场景。各模型的 V 只复用既有 V1800、两 draw、三个场景的 logits，不重复推理。

四模型均冻结原默认头及骨干，0 optimizer/head/EMA 更新，前后 state 完全相等。逐模型报告 raw、重建 As、同 L 新 As、V 新 As 的 n、accuracy、默认头 CE、场景/RX 及修正/新增错误；明确标准 CE 与训练 smoothing0.01 的口径。最多1440个 L 神经前向包/模型，四模型最多5760包，首步合法源单包 smoke 另计。n=1 模型 seed，不能证明可重复收益或物理上限。

代码：analysis/cvs_pure_F_observed_fit_20261019.py；代码/config commit 在实际提交后记入登记。普通 N607 既有 FP32/TF32off 环境、CPU2；CWD PROJECT/releases/cvs_F_sceneRisk_E20_20261019_r01（新诊断脚本位于独立发布目录）；GPU5 单进程，含其他 owner 每GPU最多2实验进程、至少12000MB空闲。唯一 owner codex/root/F-observedChannel20，不干预健康任务。

命令：CVS-RFFI/python -u analysis/cvs_pure_F_observed_fit_20261019.py --project PROJECT --output PROJECT/runs/20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01。首次创建不可覆盖输出根；保存 config、selected_L、source predictions、各模型及聚合分析、complete/failure 和日志。只在所有自身 FINAL20 及完整源日志可用后执行。角色/来源/非有限值/状态改变/流或接口/输出错误即技术停止并保留现场，不自动重启；无源阈值救援或新晋级规则。

发布前完成窄范围协议负测、真实 checkpoint 无 query 源 smoke、一次 P0/P1 审查、一次归档传输比较及远端编译；启动后以实际 PID/CWD/argv/GPU/日志验证。相同代码不重复审查，不增加签名或文件哈希门槛。

本地验证：py_compile PASS；E20按序cell上限与step/slot保留、非法role ID/metadata/base_index在IQ读取前拒绝、hard CE与smoothing0.01及配对计数均PASS。一次独立P0/P1审查PASS，覆盖source-only构建、自身FINAL20、L成员校验、训练流/新draw配对、冻结状态与GPU共享容量锁。尚未进行远端模型读出。
