# 载波增强随机开关的相关性 × 普通 practical：E30 四组预登记

状态：IMPLEMENTED_LOCAL_VERIFIED，未发布、未开始训练。固定共享 RX 响应及归一化解释已失败并关闭。本行不继续拟合 RX，也不重做历史 IQ VAT／频移范围扫描；检验既有增强的批次组织，明确不是新 IQ Ag 家族或 Ag×As 效果的替代证明。

## 唯一假设与实际依据

当前真实 DataAugmentation._apply_general_channel 对相位与 CFO 使用 torch.rand(()) 的整批 Bernoulli 开关，参数本身按包采样。F＋跨 RX 同 TX 对比在 time_down／t2.pool 由当前 L 批次均值和标准差归一化，因此开关相关性可能影响对比目标。假设：只改为独立逐包开关，在不改变逐包观测分布的条件下，能改善跨 RX clean 和普通 practical 适应。没有性能保证。

最近真实调用来自已冻结 F runtime 的 DataAugmentation 与 SSDG.train_ssdg；normal augmentor 仅在两处互斥 L 路径调用，U 不使用该实例。上游真实 E13 与既有 E30 配置均记录 scale=0.2506003339349741、p_cfo=0.08771011687724094、cfo_max=0.00010024013357398965 cycles/sample、phase_noise_sigma_max=0.0015036020036098448；这些是已配置值，不把 base_config 原始概率当运行值。相位开关实际概率由实例记录，范围保持原 0 至 2π。新 helper 直接读取已配置实例属性，不设置新幅度／概率。

RFFI 的小扰动不自动保持身份：[Lu 等，Erasing Radio Frequency Fingerprints via Active Adversarial Perturbation，2024，arXiv:2406.07349](https://arxiv.org/abs/2406.07349) 的原始摘要明确区分通信容忍度与指纹可识别性；因此本行不以小范数为新扰动辩护。历史 IQ／特征 VAT 已有完整实验，本行也不重复。这里复用已有算子及其逐包边缘分布，只改变包之间的选择相关性，识别价值必须经本行实际训练检验。

## 四组、共同父模型与固定预算

| 行 | L 相位／CFO 开关 | 第二监督视图 |
| --- | --- | --- |
| NOAS_BATCH | 原整批共享 | 同一份已生成 ordinary IQ |
| NOAS_PACKET | 独立逐包 | 同一份已生成 ordinary IQ |
| AS_BATCH | 原整批共享 | 原始 raw IQ 经完整 practical |
| AS_PACKET | 独立逐包 | 原始 raw IQ 经完整 practical |

四组均从同一个原始 fullsource REPEAT 固定 source-final20 student＋own EMA 独立分叉：`PROJECT/runs/20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02/REPEAT-s2026092701/source/final_ssdg.pth`。该父模型从零 E20、未训练 As，无继承未知或不同数据权重；发布时复用现有来源证明，核对真实 source role IDs，加载前检查来源，并做一次真实 CP 无 query smoke。它不是按目标成绩挑选；基准此前已暴露，不声称新盲测。

每组只追加 10 epoch／2220 更新至累计 E30／6660 更新，重置 AdamW／原 cosine10，严格恢复共同父 EMA；旧门控、U 规则和 source SupCon 的累计 E21..30 权重保持。四组不是采用各自 NoAs／As20 父模型，As 的实际训练曝光仅发生在本次 10 轮。旧 As20＋10 结果不能代替本行 AS_BATCH。共同父训练只计一次 20 model-epoch／4440 更新，新训练合计 40 model-epoch／8880 更新，唯一训练成本合计 60 model-epoch／13320 更新；不把四次父复用冒充四次实际训练。

完整 L6300／U56700 TX 真值隐藏／原 V27000、全部 RX[1,3,4,6,8]／day1/2/3／原六 TX／split392005／256 点保持。不留出训练 RX，不使用 L5580／Ldev720。旧普通增强作为四组共有成分，不把 NoAs 说成完全无增强；不新增 PA／DAC／量化／剪裁、窗口切片或 IQ 通道。保留 NoExplicitCFO 和必要内部逐包补偿。

固定 CVS/F、三路 SIG_TIME＋统一 Sinc、joint160、原默认 cosine6×160/scale30。骨干和默认头正常联合训练，不冻结后重训头、不另训身份／RX 诊断头，不增加部署分支或参数。

## 随机流、损失和视图成本

BATCH 模式与原 native 输出按相同 RNG 逐位一致。PACKET 同样消费原整批开关和条件参数随机数，新增包级 mask 与原开关未启用时所需参数使用独立生成器，保留其他 Torch／Python／NumPy 随机流；不按 TX 标签、样本难度或模型置信度采样。只装到 native.make_augmentor 创建的唯一 L 实例，不装到 U、teacher 或 practical generator。载波变化顺序保持原 general_channel 内部位置，后续 multipath／noise／原 PA 的依赖照原代码执行，不连续重复施加完整 practical。

每步同 128 个物理 L，四组均 4 次 L 前向：.5 RD(ordinary)＋.5 RD(同 ordinary 或 practical(raw))，再加原 weight0.1 SupCon 和原 U 项。两次 RDrop 前向使用同一份已生成 IQ；As 的所有 128 个合法有标签样本参与 CE，无 urban 高置信度门槛。SupCon 仍只用原 ordinary RDrop 对的两层特征，不增加损失、额外 VJP、教师或神经前向，原始监督视图权重 .5、总 L 规模 1 固定。记录实际 NN、teacher、U、Sinc 更新和墙钟，不把独立随机数／CPU 日志成本省略。

## 科学放弃条件与自动评价

所有末轮固定冻结；固定原 V1800、全部 5 RX／90 单元，clean 一次、每个 practical 两个命名独立 draw，原 V27000 仍作常规验证。原 V1800 是同一 V 角色的固定源读出，不创建新 V_cal/V_select，不拟合参数，不用 target 选择幅度、父模型或候选。

固定源资格必须同时满足：NOAS_PACKET clean 相对 NOAS_BATCH 至少 +0.25 个百分点；AS_PACKET clean 不低于 AS_BATCH 和 NOAS_BATCH；每个 draw 的 low urban 相对 AS_BATCH 至少 +0.5 个百分点，mid urban 不下降；每个 suburban 场景／draw 相对 AS_BATCH 不低于 −0.25 个百分点。任一失败关闭固定两个载波开关、概率／范围、相关性组织、父／seed／E30 配方，不改组件、概率、幅度、父或 seed 救跑。通过仅为源资格，不是新 IQ Ag 或联合目标完成。

suburban 接近 clean 在运行前按每个场景距自身 clean≤2 个百分点记录为探索目标，同时检查 clean 净收益，不能靠降低 clean 缩小差距。完整正式比较仍要报告各场景绝对准确率、差距、配对修复／新增错误及交互项；本行交互是“训练组织×As”，不能写成新 IQ Ag×As。

每行训练冻结后立即自动做源读出、clean＋全部六种固定 practical（168000／view）预测，再独立 CPU truth-last 评分，不等待其他行训练，也不等待新测试许可。目标 IQ、SNR、信道、样本和默认头不变，不删除困难样本。源跨行决策先冻结，再由主分析读取正式数值；各行分数不回流其他健康训练、选模、参数或选择性重跑。全部四行均固定评价，无源失败后的选择性丢行。

## 最小验证与发布

本地 CPU：8 个完整原增强／BATCH／PACKET 对照（含旧 PA 等），BATCH 逐位等价，外部 RNG 结束状态一致；2048×128 的两种开关采样显示边缘概率相同、批次相关性不同；256 个合成特征 mask 枚举确认既有批次对比目标可改变（非识别收益）。4 个适配器负例＋19 个预算／父来源／真值隔离／源判据负例通过；坏的一个 low draw 不能被均值掩盖。0真实 IQ／CP／NN／更新／target。本地没有 CUDA，未宣称通过 CUDA 检查；实际 GPU 计算在发布后的所属进程验证。唯一直接 P0/P1 审查PASS，未发现P0/P1；未读取真实输入/权重/远程或重复测试。

早期合成脚本种子超出 NumPy32bit 范围，发生在 IQ 生成和模型使用前，改为固定合法种子后通过；NumPy2 与本地 Torch2.1 的桥接警告未引发算法失败，未改环境。适配器随后减少 GPU→CPU 同步，相关检查已重跑通过；不重复不变审查。实际成本按各次真实执行保留，不把失败前的检查算成训练。

CWD `PROJECT/releases/cvs_F_carrierGate_E30_20261018_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_carrier_batch30/runner.py`；唯一输出 `PROJECT/runs/20261018-phase1-F-crossRX-carrierGate-e20plus10-s2026092701-m4-r01`；owner `codex/root/F-carrierGate-E30`。本地唯一发布入口 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_carrier_batch30.publish`。实际代码／配置由一个 Git commit 标识，普通账户一次 preflight、单 archive 一次本远 SHA、remote compile，所需 checks 和一次真实共同父 CP／2 合成包无 query smoke 后继续。无设计 SHA、成员 hash、seal、额外 review 或审批。

每 GPU 最多 3 个总实验进程、计入其他 owner，由现有 capacity lock 调度。所属系统性技术／角色／非有限失败才停止新调度，保留产物、健康行完成，不自动重试、不热改／停止旧健康任务，不以性能早停或擅自延长预算。预期每行完整 source checkpoint／初始化／真实 args、2220 步／10 轮日志、carrier_steps、L/U 物理曝光、Sinc 更新、23400 源读出 NN／21600 源传播、七视图固定预测／独立评分，以及全矩阵源决策和同 row 正式报告。

整体联合目标 ACTIVE；本行尚无训练或正式性能结果。
