# CVS跨通路复相关：匹配源训练与条件clean测试

状态RUNNING，8份训练已独立核实存活并输出epoch。两种架构实现完成：跨通路整体能量归一化Gram与逐投影通道coherence；8份从零训练，每种4个model seed，E200。识别性能与Phase1论文目标仍未达成。

## 结构与研究依据

依据前轮完整源训练与冻结读出归因，研究两路独立池化未显式保留的time×behavior关系。原Shallow骨干及两路原320维统计全部保留；每条末层复数通路32→8投影，固定lag0/4/8、共同behavior位置8至63，对应time位置t-lag，每个lag恰好56对。计算未中心化复交叉矩，差别仅为整分支RMS或逐投影通道RMS分母。三矩阵实虚部384→16→320为无偏置线性低秩出口，末层零初始化，加到behavior统计。原time统计、频域分支、160维最终表征和分类器保留。两种模型均233275参数，新增12288。

详见[设计推导与边界](../../../docs/CVS_CROSSPATH_RELATION_DESIGN_20261003.md)及[模型实现](../../../experiments/cvs_crosspath_relation_identity/model.py)。低秩出口不保证避免共同适应；投影特征的增益不变性不等于真实信道/RX不变性。三个lag不是物理同步操作。单一结构矩阵不能排除新增容量解释，不能直接声明充分论文创新。

## 冻结训练与输入边界

新模型scratch-only，无checkpoint祖先、预训练、resume、teacher、EMA或蒸馏。旧Shallow及Anchor仅为固定源统计控制，不加载其权重进行训练。复用已核实源物理划分：L6300、U56700不使用、V27000，6TX、RX1/3/4/6/8、day1/2/3，equalized=1、center256、unitRMS，split392005。四seed为2026092701至2026092704，仅改变模型/loader RNG。

保持原单一TX交叉熵、batch128不丢尾批、E200×50步、AdamW lr0.0002、wd0.0001、余弦最低lr0.000001、FP32/TF32关闭。不加增强、额外loss、采样/加权、阶段训练或目标适配。V只评估；不更新参数、归一化或其他状态。详细step/epoch/文本及紧凑CSV/JSONL保留实际loss、LR、梯度、关系范数/数值下界触发、源V/RX、耗时和资源。末批诊断范围为28包，不冒充全源或因果物理诊断。

## 选择与测试收尾

固定16源记录：8新候选+4Shallow+4Anchor；比较四seed E200平均的0.5×V准确率+0.5×最差源RX准确率。性能完全并列后比较成本，不按更轻优先接受较低性能；不按最佳epoch选择。旧控制只读取既有合规源产物，目标指标不参与设计或选择。

新结构胜出后，条件run `20261003-phase1-cvs-crosspath-relation-clean-manysig-m48-r01`执行4份新clean预测并复用44份固定控制，168000个物理query、6TX、7RX；预测固定后独立truth-last评分。旧控制胜出则核实复用其既有冻结测试，不对落选候选新增query。只测clean，不追加LEO、support或SFT。具体逐行路径、六类seed与输入输出见[experiment.json](experiment.json)，训练配置见[configs](../../../experiments/cvs_crosspath_relation_identity/configs/experiment_spec.json)。

## 本地验证与发布

模型15项针对性测试通过，涵盖配对初始化/旧state与RNG、独立复数公式、lag方向/共同窗口、范数界、精确增益性质适用范围、零/低能量梯度、全模型CE连通性、逐包独立性和诊断状态保持。8组public synthetic smoke验证所有4seed×2结构的初始函数相等及3步原CE更新；不加载正式数据或历史checkpoint。源管线及完整分析102项聚焦测试通过，合计117项模型/协议/分析测试。一次独立P0/P1审查PASS，无发现；检查了实际8配置、8控制、16源记录选择、发布832文件中的59个导入依赖以及四段远端模板。该审查范围是模型与源发布，条件clean实现需在源冻结后核对。

唯一launch owner为`codex/root/cvs-crosspath-relation-identity-20261003`。从固定Git提交发布，不可覆盖release/archive/run/log；SCP前核实路径、身份、磁盘与GPU。远端先执行无正式数据的模型smoke，各训练row加载数据前执行自己的真实scratch checkpoint smoke，再按每GPU最多两个训练任务启动。低性能不触发停止、重启或选择性重跑；技术异常按原预登记规则保留产物处理。

## 启动读回

实际发布commit为`e136f7868e5ff2569d2a61578ea909a4b1d3daa0`，dispatcher PID2289161。全部8个worker的PID/CWD/argv、父进程、CUDA可见设备和nvidia-smi物理GPU映射一致，GPU0至7各一份训练；两次日志读回均增长。全部实际配置匹配233275参数、12288关系参数、原单CE、FP32和原源数据角色。启动验证时各row处于E4至E9，源分数尚未冻结，不能据早期结果选模。

[启动独立核验](evidence/launch_validation.json) · [完整实际配置与进程](evidence/launch_readback.json) · [SCP前路径与资源](evidence/pretransfer_readback.json)。条件clean入口正在准备，当前零目标访问。

## 条件测试实现就绪

[条件clean入口](../../../experiments/cvs_crosspath_relation_clean/prepare.py)已实现并通过147项新测试、102项既有readout兼容测试。独立P0/P1审查PASS，另执行14项有界合成回归，核对了发布范围内100个本地导入依赖及四段模板。上述均为本地合成/模型加载验证，不是实际目标成绩。新scope严格核对源release commit、完整16源记录、来源/物理角色/FP32/参数与checkpoint payload；全部48行预测固定后才连接truth。

正式clean configs仍不存在，未启动测试或读取真实目标数据；训练release保持原版本。最新已核实8组训练处于E90至E95/200，见[进度证据](evidence/source_progress.json)。完成E200及合法源冻结后直接按预登记条件收尾，不再索要已有范围的测试许可。

[条件测试本地验证](evidence/conditional_clean_local_validation.json)。
