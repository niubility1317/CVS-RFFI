# 现有 EMA 的源 V 校准：预登记

目的：区分旧 batch-neighbor 的低覆盖与 clean/弱 EMA 软标签本身的可信度。原 18 个 E30 模型已经完成，本分析不训练新模型、不更新参数、不拟合分类头，不改变任何训练行、原正式七场景 IQ 或评分。

候选/矩阵：固定 C4 的三模型 seed 2026092701/02/03，各加载其已核实 scratch final E30 checkpoint 内的自身 EMA；不按目标成绩选择 seed 或 checkpoint。仅打开合法源 V 全部 27000 条及其已有源标签，U 保持未打开，query/truth 不读取。数据复用既有源契约，Phase1，不引入 Phase2 句柄或新数据验证。

每模型记录 clean EMA 默认头准确率、置信度均值、clean/同物理样本弱视图预测一致率，以及预先固定置信阈值 0.92/0.95/0.97 的精度、覆盖率、错误软质量和 RX×TX 分层。分别统计 clean confidence 和同样本弱一致条件；不把置信度均值、L 高置信覆盖率或源 V 精度冒充 U/target 精度。分析不选择下一轮参数或模型。

弱视图仅为已有源 V IQ 加每 IQ 坐标标准差 0.01 的固定高斯扰动，seed=2026101501；原输入归一化保持，不额外输出归一化。三模型共用同一源 V 顺序及相同弱扰动。它不是正式 practical 信道修改。没有困难 urban 学生置信门控。

实际前向沿原 teacher 完整调用；首个真实 V batch 与部署 identity-only 默认头输出进行等值 smoke，随后继续分析。严格加载 EMA 来源，所有参数冻结、eval/no_grad，结束按 tensor equality 核查参数与 buffer 未变。每模型 54256 个网络前向样本，合计 162768；新增训练更新、epoch、拟合头更新均为 0。实际墙钟后报。

代码/config 由本分支实际提交标识，运行保存 release_commit.txt；没有额外设计、文件或报告 hash。命令为 `python -u -m experiments.cvs_urban_sigtime.ema_source`，环境为 N607 既有 CVS-RFFI Python，CWD 为 PROJECT/releases/cvs_sigtime_urban_paired_20261014_r02_ema_vcal。CPU 两线程，CUDA_VISIBLE_DEVICES 为空，不占训练 GPU，不修改其他 owner。

唯一输出根为原 run/source_ema_vcal/，不覆盖现有 source_mechanism、训练、prediction 或评分。唯一 owner 仍为 codex/root/urban-sigtime-20261014。技术停止：来源/源角色不匹配、打开 U/query、默认头前向差异、非有限、模型状态变化、输出已存在或无法运行；保留产物，不自动重启。低精度或低覆盖保留为负结果。

发布只做一次归档传输比较、远端 compile/preflight，以及启动后 PID/CWD/argv/CPU/log 或进度增长检查。新分析的完成状态必须来自 source_ema_vcal/results.json，不能借用原训练 complete.json。

期望产物：progress.json、results.json、源校准报告与源分层图；原始结果留在实验存储和本地 artifacts，不进入 Git。已执行五项聚焦检查：置信度不等于精度、同样本条件计数、空集合精度为缺失、隐藏 U 哨兵拒绝、非有限拒绝及独立发布路径。真实 checkpoint smoke 在 launcher 内执行。

下一轮新增模型数量仍待用户答复，本分析不消耗或扩大 E30 模型预算，不把 60% 目标记为完成。
