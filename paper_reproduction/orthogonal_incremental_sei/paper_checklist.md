# 正交空间约束FSCIL-SEI论文复现清单

本目录只承载《正交空间约束的特定辐射源小样本类增量识别方法》的闭集FSCIL忠实复现，不写CVSStage2-B/C、satellite/LEO部署、unknown FAR或open-set拒识结论。

|论文项|实现位置|状态|
|---|---|---|
|伪目标数量边界`|C|<=N<=d+1`与规则单纯形几何|`pseudo_targets.py`、`train.py`|已实现闭式规则单纯形构造；当`N<=d+1`时直接使用理论最优几何，避免随机迭代劣化伪目标内积边界；超出可行边界才保留迭代优化路径|
|伪目标扰动`\tilde t_i=t_i+epsilon_i`|`pseudo_targets.py`|已实现，默认严格加性扰动；归一化需显式开启|
|固定映射函数`h:C->{1,...,N}`|`pseudo_targets.py`|已实现；默认保持输入class/session顺序，显式`sort_labels=True`才按数值排序|
|六层Conv1D-BN-MaxPool特征提取器|`model.py`|已实现可运行骨架，通道数为实现选择|
|余弦相似度分类器|`model.py`|已实现|
|基类交叉熵损失`Lce`|`losses.py`|已实现|
|自监督对比损失`Ls`|`losses.py`|已实现论文等价结构；样本锚点正集包含自身和同类样本，分母保持论文负集口径|
|类中心分离损失`Lc`|`losses.py`|已实现|
|增量阶段冻结特征提取器、类均值初始化新权重|`train.py`|已覆盖正式runner流程；基类旧权重由训练后特征均值生成，新类权重由5-shot支持集特征均值初始化，增量阶段显式冻结encoder并只优化新类权重|
|边际竞争校准`Lh`与原型对齐`La`|`losses.py`|已实现；配置使用论文符号`lambda_a`|
|`A_bar/H_bar/F_bar`指标|`metrics.py`|已实现；`A_bar/H_bar`默认按论文表格口径跳过A0并对A1...AT取均值，`F_bar`默认按论文增量任务分母，不做非负截断；可显式设`average_denominator=total_sessions`复现旧口径|
|WiFi增量5-shot测试协议|`train.py`|已修正：基类仍按80%/20%划分；增量类按每类K个support样本训练，其余样本全部作为query|
|表1早停轮数|`train.py`|已接入`early_stop_patience`和`early_stop_min_delta`，base和每个increment session分别记录`early_stop`状态|

## 子agent逐项对照结论

|论文内容|当前工作对应|审计结论|
|---|---|---|
|摘要贡献：正交伪目标与扰动伪目标|`pseudo_targets.py`、对应单元测试|机制已覆盖；正式训练收益未验证|
|摘要贡献：`Lce/Ls/Lc`协同优化|`losses.py`、base loss梯度测试|公式机制已覆盖；仍需真实batch训练曲线|
|摘要贡献：增量阶段分类器权重校准|`incremental_calibration_loss`、dry-run增量梯度检查|校准损失已覆盖；多session增量训练未完成|
|算法1基础阶段训练|`run_formal_wisig`|已支持真实WiFi/ManyTx闭集FSCIL训练、早停和训练后base class-mean权重；严格论文官方WiFi数据源仍需另行提供|
|算法2小样本增量训练|`run_formal_wisig`|已支持K-shot支持集、剩余样本query、冻结encoder、class-mean初始化新权重和多session校准；严格同论文receiver/TX顺序仍需查证|
|公式(4)正交空间目标|`pseudo_target_orthogonal_loss`、`optimize_pseudo_targets`、`train._build_pseudo_targets`|已覆盖闭式构造、几何保持测试、可选迭代优化和训练入口配置|
|公式(18)-(29)基础阶段损失|`base_training_loss`|已覆盖；`Ls`已按论文正负样本集合和逐正样本平均修正|
|公式(30)-(36)增量校准|`incremental_calibration_loss`|已覆盖；已补shape/device检查|
|公式(37)-(41)评价指标|`metrics.py`|已覆盖；已补空old/new集合、长度检查、`A_bar/H_bar`论文表格口径、遗忘率差值口径和`F_bar`分母口径|
|表1仿真参数|配置文件与清单登记|已使用论文符号`tau_s`、`tau_c`、`q`、`lambda_a`登记，并执行早停轮数；数值仍属implementation choice，需作者代码或网格确认|
|表2-表5 ADS-B/WiFi正式结果|无正式输出|未完成；需要ADS-B100类和WiFi/WiSig130类loader与训练运行|
|图4/图7遗忘率曲线|无正式输出|未完成；需要正式session accuracy matrix与遗忘率曲线数据|
|图5/图6类别数量敏感性|无正式输出|未完成；需要初始类数量和增量类数量网格运行|
|表6消融实验|无正式输出|未完成；需要逐项关闭`Ls/Lc/校准`等模块运行|
|表7训练与推理时间|无正式输出|未完成；需要目标GPU计时|
|图8超参数敏感性|配置登记|未完成；需要`N/top-k/tau_fuse/lambda_a`敏感性实验|
|图9混淆矩阵|指标函数可支撑|未完成；需要保存正式混淆矩阵|
|论文结论|`findings.md`与缺口登记|当前不能称完整复现，只能称机制级骨架和synthetic dry-run|

## 仍需真实数据确认

- ADS-B公开数据文件路径、类别排序与100类筛选规则。
- WiFi/WiSig同一接收机样本筛选后的130类清单。
- 论文表3实际class order、receiver ID和seed/多seed平均策略。
- 论文增量校准公式(31)-(36)的完整排版细节；当前PDF文本抽取在公式区混排严重，仍需原PDF截图或作者代码核验困难样本风险定义。
- 论文未给出的六层CNN通道数、卷积核、stride、`tau_s`、`tau_c`、边际`q`具体数值等细节。
- 正式表2-表7和图4-图9需要在数据和GPU运行可用后生成，本目录当前只提供论文机制复现骨架和dry-run验证。

根目录`paper_reproduction/paper_original_matrix.md`和`paper_reproduction/repro_gap.md`记录了跨论文复现矩阵与缺口登记；本子目录不单独复制这两个文件，避免发布面把机制骨架误读为完整论文结果。
