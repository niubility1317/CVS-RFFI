# 浅中层特征：校准时点与联合梯度上限

状态：实现、本地 CPU/CUDA 验证与独立 P0/P1 审查均通过，尚未发布。目标为每个完整方案的 clean 三种子均值达到 84%，逐种子报告；结果不回流改变固定矩阵、参数、选模或重跑。

本轮是源日志支持的优化：前批从头训练中层方案的初始固定系数在后续 999 个辅助步骤上出现梯度尺度漂移。证据见前批 `source_calibration_drift.md`。这一现象不等于已解释测试变化，也不证明限幅一定提高泛化。校准时点和运行时约束必须分别比较。

## 固定矩阵

|作用层与路径|DELAY：延后校准|CAP：联合限幅|DELAYCAP：二者结合|
|---|---|---|---|
|浅层 `time_down`|DELAY_M_SHALLOW|CAP_M_SHALLOW|DELAYCAP_M_SHALLOW|
|中层 `t2.pool`|DELAY_M_MIDDLE|CAP_M_MIDDLE|DELAYCAP_M_MIDDLE|
|浅层＋中层|DELAY_M_BOTH|CAP_M_BOTH|DELAYCAP_M_BOTH|
|浅中层＋主路径 MixStyle|DELAY_JOINT|CAP_JOINT|DELAYCAP_JOINT|

共 12 方案×3 种子＝36 行。种子为 2026092701、2026092702、2026092703。主方案预先指定为 DELAYCAP_JOINT；全部方案逐行自动测试，不按测试排名筛选种子或 checkpoint。

学生全部随机初始化，保留旧门控、参考响应及原生分类头；cos 指 cosine 学习率。20 epoch×222＝4,440 次身份更新。AdamW 峰值学习率 0.0002、末轮 0.000001、weight decay 0.0001；L 批量 128、label smoothing 0。前 444 步只训练 L CE，此后 888 步 ramp。无 EMA、无星地增强，无额外最终 embedding 损失。

固定同预算对照复用 `20261012-phase1-scratch-boundary-style-manysig-m27-r01` 的 CE、SSL、M_SHALLOW、M_MIDDLE、M_BOTH、JOINT，同种子配对；包括所有种子，不重跑相同配置。此批的最优层或最优种子不作为选择条件。原 E20 的 P0 另作固定评分参考。对照结果仅由收集/报告模块读取，训练、缓存和校准不读取评分。若对照未完成则标 N/A，不能补造或改用其他权重。

## 损失与作用位置

L_id 为已有的主 L CE＋ramp×U soft KL；JOINT 再加 ramp×额外风格 L CE。U 的目标分布为另外两个原生 E20 教师概率均值，全部样本参与 KL，温度 1；教师冻结且只访问合法 U。随机学生并不意味着整个方法无需预训练：每行复用两教师共 40 模型轮，连同学生共 60 模型轮。预测仅执行学生。

Fishr 类项沿用主身份 margin 对浅中层通道虚拟仿射坐标的敏感度。margin＝正确 logit−其余 logits 的 logsumexp；敏感度逐样本 RMS 归一化，epsilon＝1e-6。两批独立 L60 在同一天按 6 TX×5 RX×2 样本分层，计算有限总体修正的 RX 方差中心化内积。保留正负值和真实二阶梯度，每 4 个身份更新启用一次，共 999 次。它是 Fishr 类敏感度方差匹配，不能称为原版参数 Fisher；没有学习型辅助头或额外身份分类器。

两层损失按固定系数和 ramp 加权后除以 2，未启用层贡献为零，得到联合 R。单层与双层的这个固定归一化与前批完全一致。

MixStyle 修改浅中层特征的通道均值和标准差，供体为合法 L 中同 TX、同 day、跨 RX、不同 PID；不使用 U 的真实身份。保留四候选、整批 eval CE 最大者选择、与 clean 相同 dropout 随机状态的梯度重放。候选选择仅访问当前合法 L。风格 margin 诊断使用正确 logit−最大错误 logit，与敏感度的平滑 margin 区分。非 JOINT 行没有风格生成或额外风格前向。

## 校准与联合上限

CAP 沿用第 0 次更新的一次校准；DELAY 与 DELAYCAP 在第 444 次 optimizer.step 完成后校准，首次辅助步骤为 448。使用同一组 12 对 L60、1,440 样本读出、24 前向，不更新参数、缓冲区或优化器，恢复随机状态和模式。每层系数为 min(10000,0.2×平均 CE 梯度范数／(平均自身敏感度项梯度范数＋epsilon))，固定到训练结束。一次校准的 0.2 是参考，不是全程上限。

CAP/DELAYCAP 额外对实际联合 R 求导。将所有可训练身份参数互斥、完整划分为 `.time_fuse.`、`.t2.` 和其余参数三组，包含频率参数。对每组 G 计算 g_id＝∇L_id、g_R＝∇R，使用单一标量：

`s = stopgrad(min(1, min_{G: norm(g_R)>0}(0.2*norm(g_id)/(norm(g_R)+1e-12))))`

实际目标为 L_id＋sR。全部辅助梯度为零时 s＝1；某组主梯度为零而辅助梯度非零时 s＝0。复数梯度范数使用绝对值平方。统一 s 能约束联合项，避免逐层独立裁剪后相加破坏上限。记录 raw/capped 范数和界，对每组直接核验 `s*norm(g_R) ≤ 0.2*norm(g_id)`。

这是瞬时梯度范数界，不保证 AdamW 位移、梯度方向或泛化。s detach 后是自适应更新规则，不声称等于某个固定总体目标的完整梯度。每个限幅辅助步新增两次 autograd 求导，保留实际耗时；DELAY 无此开销。限幅也可能抑制有效正则，三组消融用于检验这一风险。

## 数据、日志和测试

L 6,300、隐藏 TX 的 U 56,700、V 27,000；沿用既有物理 ID 与角色契约。V 仅 E5/10/15/20 eval/no_grad 健康诊断，不参与校准、更新、选模。学生不加载 checkpoint；两份教师完整来源仍经现有 source-only 原 E20 检查。只使用相同已验证 IQ，不重复验证数据集。

初始 resolved 单独保留，校准 pending/ready 明确区分；唯一 CALIBRATION_READY 记录实际身份更新数，最终 resolved、校准文件、辅助状态和资源一致。详细文本、4,440 条逐步记录、20 轮紧凑 JSONL/CSV、各层原始统计、联合上限和权重移动全部保存。

固定 final20 后每行立即预测 clean 的 168,000 条 query，再由独立 scorer 连接 truth，累计 6,048,000 条决策；报告每种子、均值/SD、RX/TX/day、Macro-F1、NLL、配对变化和资源。目标基准已用于历史探索，属于固定方案开发评估，不能宣称全新盲测。

八张 GPU 按共享容量锁分配，每卡最多 3 个总 GPU 进程（包含现有任务）；每行唯一 launch owner，训练和测试沿同一 worker 串联。旧健康任务保持原 release，不覆盖任何日志、权重或输出。技术失败保留产物，不因低性能停止，也不自动选择性重跑。
