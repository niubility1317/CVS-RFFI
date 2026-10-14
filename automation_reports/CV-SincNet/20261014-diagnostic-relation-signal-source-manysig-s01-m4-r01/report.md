# 跨 RX 关系与 CVS 增量响应的源身份学习信号

本诊断固定读取原关系矩阵首种子 ZERO、ALL_REL、CROSS_REL、CROSS_RESP 的 final50 student 与各自 ownEMA。四条件由原方案预先指定，不按目标分数挑选。目的是判断非零特征梯度是否沿着降低同一源样本身份误差的方向，而不把“梯度存在”当作有效学习信号。它不改变已发布关系训练矩阵，也不训练、替换或重训分类头。

## 固定矩阵与测量

每模型使用合法 L_s 中 180 个参照包（5 RX×6 TX×3 day×2）和与其不重叠的 720 个源探针（每单元 8 个）。相同物理包重复用于四个冻结模型，共 2880 次模型观察；不是 2880 个独立物理样本。所有探针均属于已见训练 L，不能代表未见 RX 或真实在线 U 的标签正确率。

教师为 clean/eval，参照银行及探针仅前向。学生分别为 CLEAN_EVAL 和 STRONG_NATIVE（原 noise=.015、train Dropout），每视图 6 个固定 120 包批次。每模型 2340 次 IQ 包前向，四模型共 9360 次；不更新参数、buffer、EMA 或优化器。源 V 缓存被立即丢弃，没有 V 前向/拟合，U IQ/TX 与 target 均不访问。

主测量是同一探针批次上实际默认头身份 CE（label_smoothing=.01）与 ALL_R、CROSS_R、CROSS_E、CROSS_R_E 的完整身份特征参数梯度范数、点积、余弦。参数集合为可训练 id_backbone，排除默认 class W 与两种融合增益。辅助目标沿用当前关系 objective，真实 joint160 的数值与部署输出完全一致，辅助 W/gain 梯度必须为 None。记录 t1/t2/t3、PA 卷积和 reference projection 的实际梯度。

点积正说明该辅助方向的局部最速下降一阶上降低当前同包身份 CE。它不直接预测有动量、裁剪及 weight decay 的 AdamW 实际更新，更不是泛化改善的证明。两批不同 L/U 数据的梯度余弦不能替代本测量。

补充测量为逐包 joint160 的部分梯度；明确固定 PA/ref 增量，仅用于解释，完整参数梯度为主。教师 PA/ref 签名响应另与合法 L 的同 TX 参照质量及默认头真实类间隔增量对照。L 标签只进入这些诊断和 CE，不进入关系损失，不提供 U 伪类投票或新筛选机制。

## 来源、运行与止损

- 数据：既有 VALIDATED_ONCE ManySig 契约，L6300/U56700 隐藏 TX/V27000，RX1/3/4/6/8、day1/2/3、6 TX。既有 capsule/split/物理角色完全复用。
- 父 run：`20261014-phase1-evidence-relation-pair-oldgate-e50-clean-manysig-m12-r01`，四个 scratch33+17/final50，无继承、无目标训练或选模。加载前执行既有 source_proof，严格加载 student 与 ownEMA，并核实 default head、11100 次更新、原 args 与 source freeze。
- run：`20261014-diagnostic-relation-signal-source-manysig-s01-m4-r01`；唯一 owner：`codex/root/relation-signal-source-20261014`。
- 代码身份：release 内 `release_commit.txt` 为实际 Git commit；不要求额外代码/report SHA。
- 环境：N607 普通账号，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CPU only、2 intraop/BLAS threads、CUDA_VISIBLE_DEVICES 空。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_relation_signal_source_20261014_r01`。
- 命令：`python -u -m experiments.cvs_relation_signal_diagnostic.audit`。
- 不可变输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-diagnostic-relation-signal-source-manysig-s01-m4-r01`。
- 止损：仅本 run 的来源不一致、输出碰撞、非有限值、joint 默认头不等价、参数/buffer/.grad 改变或保存回读不一致时停止本诊断，保留产物，无自动重试，不影响健康 GPU 任务。
- 预期产物：preflight、resolved_config、source_signals.pt、逐 row analysis/completion、总 analysis/completion、progress 和 stdout。

数学正/负/零方向、隐藏 TX 和形状负例，本地真实模型接线、W/gain stopgrad 和 native tensor roundtrip 已通过。fresh reference gain 初始为 0；合成接线 fixture 单独设置 .1 验证该路径，真实冻结模型的 gain 不修改。唯一 P0/P1 审查及真实 checkpoint no-query smoke 由发布流程完成。源结果不提供当前训练的参数、排序或终止反馈。

当前为预登记，无真实诊断结果。84% 三种子泛化目标仍未达到。

唯一 P0/P1 审查发现旧 multiscale validator 会拒绝新 relation parent config；已改用 relation.make_args 比较 checkpoint 参数，并保留 row 身份。该问题的一次定向复审 PASS，其余直接 P0/P1 未发现。下一步为发布首步真实 checkpoint no-query smoke。

## 已完成源结果

状态 **ANALYZED / VERIFIED**。四 frozen student/ownEMA 均完成，720 个固定 seen-L 探针×4 模型、9360 次 IQ 前向；参数/buffer/.grad/EMA/headfit/optimizer 更新均为 0，U IQ/TX 与 target 未访问。保存张量由独立 CPU 进程完整复算，四模型物理 IDs 与每个 TX/RX/day 的 8 包计数一致；单 archive transport 比较、远端 compile、真实 checkpoint no-query smoke 和独立 post-state 已核实。

| 冻结模型 | clean R+E 参数余弦 | strong R+E 参数余弦 | strong 辅助/CE 范数比 | strong 正向批次 | strong joint 部分梯度反向比例 |
|---|---:|---:|---:|---:|---:|
| ZERO-s2026092701 | 0.3588 | 0.0597 | 1.4505 | 5/6 | 68.06% |
| ALL_REL-s2026092701 | 0.2772 | 0.0590 | 0.5708 | 4/6 | 70.28% |
| CROSS_REL-s2026092701 | 0.1867 | 0.1061 | 0.5022 | 3/6 | 69.72% |
| CROSS_RESP-s2026092701 | 0.2081 | 0.0833 | 0.5395 | 3/6 | 70.28% |

参数余弦采用六个批次梯度串接后的点积除范数，逐批均值及所有 norm/dot 另存 result_evidence。STRONG_NATIVE 三种关系训练模型的同包 R+E 正向批次仅 3 或 4 个；总余弦仍为正，不能写作全部参数梯度反向。逐包 joint 部分梯度固定 PA/ref 增量，它约 70% 反向也不能替代完整参数更新结论。

教师 clean 默认头在 seen-L 探针的准确率为 99.44% 至 99.58%。关系训练模型的跨 RX 同 TX 参照质量约 79.3% 至 79.5%，高于 ZERO 的 70.6%；这种源关系集中并未证明域泛化更好。PA 增量使真实类间隔降低的包比例为 17.08% 至 24.03%，ref 为 25.14% 至 26.11%。因此教师的签名关系响应不自动等于有益身份贡献；仍需区分补充分支证据和复制当前分支偏好。

源证据支持的下一研究问题是：如何提供与身份学习方向更一致的跨 RX 证据信号。它不支持直接增大 R/E 权重，不触发当前健康矩阵调参或重跑；也未证明信号冲突是目标下降的唯一原因。seen-L 小代理、单 seed、终轮冻结模型、原强视图 Dropout、label_smoothing=.01 以及默认头坐标都会影响该解释，真实 U 正确率及未见 RX 仍未知。

当前诊断只完成机制测量，不是 84% 泛化目标完成，也不确认本关系机制有效。
