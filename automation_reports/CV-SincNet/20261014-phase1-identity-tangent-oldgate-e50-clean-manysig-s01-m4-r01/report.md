# 身份保护接收机切向平坦化：训练机制首种子

- run_id：`20261014-phase1-identity-tangent-oldgate-e50-clean-manysig-s01-m4-r01`
- group_id：`phase1-identity-protected-receiver-tangent`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：LOCAL_VERIFIED。数学负例和真实原生 CPU/CUDA 接线通过；唯一独立 P0/P1 审查 PASS。尚无训练或识别结果。

## 目的与对照

用户本轮明确主线为训练机制、损失函数及创新性。只新增一个假设：对同 TX/day 的源 RX 统计方向先做身份保护，再约束其在浅、中层对实际部署特征的局部影响，能否比无保护方向产生更稳定的身份表征。

源依据来自已有实际旧门控 final50 权重的浅中层诊断：另一分支报告同 TX/day 的 RX 统计差占中心平方和 t1 32.66%、t2 28.21%，固定源干预实际进入 joint160。这里复用该 owner 已完成报告，不称本轮重新读取了它的原始样本。比例不是因果接收机分离证据，只支持检验这类统计方向。原时序容量/覆盖控制已闭合，故本轮不再修改卷积结构或加载目标分数较高的权重。

比较过三个方向：直接条件特征对齐容易压缩身份信息，且项目已有条件交互实验；Fishr/分域梯度约束已有实现及另一 owner 的真实梯度方差诊断，不重复派发；本轮选择身份保护接收机切向平坦化，以可证伪的方向保护控制为唯一新机制。

切向不变性本身已有 [Tangent Prop](https://proceedings.neurips.cc/paper/1991/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html) 思想，[Fishr](https://arxiv.org/abs/2109.02934) 对齐的是跨域损失梯度方差，[MixStyle](https://arxiv.org/abs/2104.02008) 混合实例统计量。本轮项目内的新组合是：用同身份实测 RX 差估计方向、显式排除批次 TX 统计方向、以等能量控制检验保护作用、直接惩罚真实 joint160 的对称方向响应。未完成系统文献查新，不能声称全球首次，也没有有效性结论。

## 机制与损失

每次只取合法 L_s 的同一天 6 TX×5 RX×2 个互不重复物理样本，共 60 包。在实际 t1 或 t2 的输出计算每通道均值与标准差，组成统计向量 `s=[mu/RMS_batch(sigma), log(sigma)]`。所有方向和子空间估计都 detach，不参与梯度。

同 TX 在不同 RX 的统计单元均值差定义 `v_raw`；RX 按固定 1/3/4/6/8 循环选择，公式对所有 TX 一样。TX 均值先对 RX 平均，再对 TX 中心化，通过相对阈值 1e-6 的 SVD 得到最多 5 维身份均值子空间 P。保护方向为 `v_safe=(I-P)v_raw`。

RAW_MATCHED 将原方向缩到与 v_safe 每包相同的 L2 长度；两者再共同按最大坐标限幅为 1，保持投影正交性和能量相同。不存在每 TX 专属超参数。统计的 ±0.1 变化通过可微逐通道仿射返回真实特征图，保持清洁前向和推理接口。

主损失保持原联合 CE、旧伪标签 CE/熵。新增一个特征损失：

`R = mean(||normalize(z_plus)-normalize(z_minus)||²)/(4*0.1²)`，其中 z 均为实际 `id_feat_joint160`，不是新 embedding 或类别 logits。

额外梯度只作用于 f_theta。分类权重 W 不进入 R，也不进入梯度预算分母；base_norm/pa_norm/gain 属于 f_theta，保留。系数为 `ramp * min(1000, 0.2*||grad_theta L_CE||/(||grad_theta R||+1e-12))`，零 R 系数为 0，系数 detach。因而辅助梯度最多为当前主 L CE 特征梯度的 20%，不改变 W 的原联合训练规则。

前 1110 次主更新（5 epoch）不开启。之后每 4 次主更新一次，1110 次更新线性 ramp；t1/t2 按辅助调用交替。每次原生模型 eval 执行清洁统计前向、正方向、负方向 3 次，再恢复全部模块原模式，避免 dropout 不同或 normalization 状态更新混入方向响应。新增前向保存/恢复 CPU/CUDA 随机流，U/teacher 不做干预。EMA 仍按原流程吸收学生的实际参数更新。

保护只覆盖本批次线性 TX 均值统计子空间，不能保证非线性身份信息全部保留，也不能声称接收机因果可辨识。随训练更新的方向可能退化；逐调用保存身份秩、保护剩余能量、原/保护方向的身份重合量、损失、系数和真实梯度比例。CE、源 V 及完整负结果共同限制结论。

## 固定首种子矩阵与成本

四行：BASE_REF 复用原历史默认头预测；ZERO、RAW_MATCHED、IDSAFE 各从零 50 epoch、33 label＋17 pseudo、11100 主更新。模型种子固定 2026092701；后两预定义种子尚未在本 run 启动。

三个新模型结构、全部初始参数/buffers、默认 cosine6×160/scale30、主数据与随机流规则相同；新增参数 0。ZERO 同样采样/执行全部 3 个辅助前向，但其方向、R 和辅助梯度严格为 0，用于隔离额外计算和 source 观察。RAW_MATCHED 隔离身份保护步骤，不用更小扰动冒充方向优势。

每模型新增 2498 次辅助调用，t1/t2 各 1249 次；每次 180 包次，合计 449640 个额外 L 前向包次。三个新骨干合计 150 model-epochs、33300 主 optimizer/EMA 更新、1348920 额外 L 前向包次。历史基线 50 epoch/11100 更新单列。额外梯度求导与耗时实测，不能称等计算预算或加速。没有新头、换头、RX/day 探针拟合或 Phase2 成本。

## 数据、seed与模型来源

源物理角色固定 L_s6300、U_s56700 隐藏 TX、V27000；源 RX1/3/4/6/8、实际 native day1/2/3、6 TX、原划分/标签映射不变，训练池标注率 10%。直接从本次原生 L dataset 采样，逐物理 cell70 条校验；不使用旧 embedding/cache 或 V 构造训练方向。NoExplicitCFO 和内部逐包补偿保留。

历史 batch_neighbor＋EMA＋普通 cosine 配方完整保持：AdamW lr2e-4、wd1e-4，原 cosine 至1e-6，label_smoothing0.01、原 L128/U256、EMA0.999、原 lambda_u0.16/lambda_ent0.01、原强弱门控。无新星地训练。固定终轮第50，不用源或目标挑 checkpoint。

全部训练后按原逐行管线自动 clean 默认头预测、fresh CPU 独立 truth-last 评分。每行168000包、18条 CM；本矩阵672000决策，其中历史168000复用、新增504000。不能把首种子超过84%称为三种子目标完成，源码强制 `goal_84=false`。

首次源健康标准预先固定：各新 arm 末轮 L CE≤0.5，源 V 总体较 BASE 降低≤1个百分点，最差源 RX 降低≤2个百分点。全矩阵完成后先只读源证据写 source_extension_decision，再读取目标指标汇总；若健康 PASS，才可按既定完整三种子目标登记同参数/同预算的后两种子新 run。目标负分也不能改方向/强度、放松原门控、选父权重、重训首种子或选择性重跑。

## 执行与存储

Git 工作树：E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011；分支 codex/report-physical-ssl-clean-delivery-20261011。代码身份以发布归档的 release_commit.txt 为准，发布前 commit/push 并独立核对 remote OID。

远端 CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_identity_tangent50_20261014_r01；解释器 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。命令 `python -u -m experiments.cvs_identity_tangent50.dispatch`。

唯一 owner codex/root/identity-tangent-20261014；普通 N607 账户。输出根为 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-identity-tangent-oldgate-e50-clean-manysig-s01-m4-r01，不可覆盖；逐行日志为该根 logs/。每 GPU 计入其他 owner 总实验进程≤3，空闲显存≥12000 MB；保护已发布 MixStyle 等健康运行。

错误角色/来源、输出碰撞、非有限值、启动/预测/评分技术失败停止本批后续派发并保留输出，不自动覆盖重跑，不干预无关任务。低目标分数不是技术停止理由。一次预检、归档传输 SHA 比较、远端 compile 和真实同契约 checkpoint no-query smoke，通过后立即继续，启动后核对 PID/CWD/argv/GPU/log 增长。

本地 CPU 与实际 CUDA 模型：三个新 arm 全初始 state 一致；R 对 classW 梯度为 None；辅助约束参与真实 f_theta 数值更新，原 CE 正常更新 W；W 修改/删除不改变 z，诊断 params/buffers 不变。CUDA 实际预算比例0.2；合成 IQ 的正 R/参数更新只证明接线，不是识别结果。唯一独立 P0/P1 审查 PASS。没有重复审查或额外审批。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
