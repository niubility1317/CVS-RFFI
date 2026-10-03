# CVS纯神经网络读出优化

2026-10-03。沿用用户的纯网络结构优化要求，保持原训练流程与单一交叉熵。本文件只保存源域设计依据和冻结方案；测试结果另存独立报告，不回流修改本方案。

## 源域依据与假设

当前源规则胜出的结构为`neural_residual_shallow`。前轮adaptive、浅层、深层模型的四seed平均训练CE分别为0.001142、0.000711、0.000587，源V CE分别为0.083226、0.086423、0.089719。增加深度改善训练拟合，但没有稳定改善源验证。浅层综合分数较adaptive提高0.0412个百分点，配对标准差0.1577；深层降低0.0139个百分点。新增卷积分支有实际输出和梯度，不能归因于完全失活。证据限定为[源最终指标](../automation_reports/CV-SincNet/20261002-phase1-cvs-neural-residual-identity-manysig-m8-r01/evidence/source_final.csv)、[完整源曲线](../automation_reports/CV-SincNet/20261002-phase1-cvs-neural-residual-identity-manysig-m8-r01/evidence/source_curves.csv)与[源分支输出](../automation_reports/CV-SincNet/20261002-phase1-cvs-neural-residual-identity-manysig-m8-r01/evidence/neural_outputs.csv)。

现固定读出将每路32×64复特征压缩为320维，保留各通道的logpower均值/标准差、lag1/2相关及4段位置均值。这个算子对每个通道独立常相位旋转也不变，强于共同相位不变；它也没有可学习的位置权重。上游复卷积能够提前编码部分跨通道关系，所以不能断言网络完全没有这类信息。待检验的假设是：**保留原统计读出，同时让网络学习额外的局部位置权重与通道组合，可以改善当前源验证瓶颈。**

旧attentive在更早实值骨干上替换单头均值池化；旧crossphase用固定通道锚点替换位置统计。本轮保留全部旧统计和现有复数骨干，作为新的匹配结构实验，不把不同旧骨干结果当作本轮消融。

## 唯一改变量

每个模型从零构建当前浅层骨干，不加载历史checkpoint。time与behavior各增加独立读出残差：

`复特征 → logpower32 → Conv1d(32,16,k3,pad1) → GELU → Conv1d(16,4,k1) → 本包时间softmax → 4头加权32维值 → 展平128 → 无bias Linear(128,320) → 加原320维统计`

|候选|额外通道混合|新增参数|总参数|
|---|---|---:|---:|
|readout_attention|无|85160|306147|
|readout_complex_attention|logpower前可学习复数32→32、k1|89256|310243|

出口线性层置零，复混合初始化为单位阵；新增构造隔离RNG，因此初始函数及基座参数与同seed浅层控制精确相同。score使用正常随机初始化。第一步内部零梯度是零出口的预期性质，测试要求第二步有实际梯度。没有附加损失、额外dropout、参数组学习率、预训练、分阶段训练或冻结骨干。共同相位性质不等于任意接收机不变性；attention读取完整本包，不能声称整网严格在线因果。

## 固定矩阵与判定

两候选各4个modelseed：2026092701至2026092704。L6300、U56700不使用、V27000；RX1/3/4/6/8、day1/2/3、splitseed392005及原物理划分不变。每模型200轮×50步，batch128、不丢尾批；单一CE，原AdamW学习率0.0002、weight_decay0.0001和cosine终值0.000001，全FP32且关闭TF32。原160维embedding、frequency路径与cosine30分类头不变。

8条新源记录与4条既有浅层控制、4条当前源赢家Anchor的源元数据组成16行排名。固定E200，最大化四seed均值`0.5×V准确率+0.5×最差源RX准确率`；性能完全相同后才比较成本。不能按最佳epoch选模，也不读取历史目标成绩重排模型。源V包含已见源RX，本规则本身不证明未见接收机泛化。性能优先，参数和实测成本作为次要比较。

若新候选胜出，只冻结并测试它的4个模型；与44份原冻结预测组成48行clean矩阵，每模型同168000个物理query、6个注册类和7个目标RX。所有预测固定后才独立truth-last评分；复算384个混淆矩阵、96组汇总和88组配对，并核对旧352条评分完全不变。若任一固定控制胜出，复用其既有完成证据，两个新候选均不访问query。只测clean，不追加LEO、SFT、support适应或新增类。

目标是取得可验证的识别性能改善，但没有证据时保留负结果，不把低性能当技术故障停止，也不选择性重跑。历史测试已暴露；审计中旧源文档末尾亦曾意外返回追加测试段，未作为本轮设计依据。这里的推断只来自明确源产物和代码；最终结果不能称为首次盲测。

## 验证与产物

聚焦验证包括实际参数/contract、旧state与RNG一致、初始logits一致、两步CE梯度传递、读出实际调用与归一化、逐包独立性、共同相位数值性质、配置及输入权限负测。依现行流程进行一次独立P0/P1审查。本地Git版本固定后发布，健康任务不热改或重启。

保留详细step/epoch文本、JSONL和紧凑CSV，记录实际CE、权重、LR、梯度、注意力熵/有效位置数、读出输出变化、源V、时间和显存。资源记录参数、训练/推理耗时、峰值CUDA内存、常驻状态；未测CPU峰值、星载传输为N/A。Conv/Linear MAC不包含softmax、FFT、归一化等全部运算，不当作总FLOPs。

源run：`20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01`。条件clean run：`20261003-phase1-cvs-neural-readout-clean-manysig-m48-r01`。主Agent是唯一launch owner。

## 2026-10-03用户目标更新

当前实现已通过本地验证但未发布或启动。用户进一步明确Phase1独立成文与信道鲁棒结构创新；后续先按[信道鲁棒研究设计](CVS_PHASE1_CHANNEL_ROBUST_ARCHITECTURE_20261003.md)确定结构假设，保留本候选作为可选读出消融。上文条件测试计划仍为未执行预登记，不自动启动。

## 2026-10-03恢复为机制控制实验

当前源赢家已更新为Anchor；本轮在发布前加入其4份源记录，不改变两个候选、训练算法或逐row配置。依据仍为原源曲线和读出算子，不使用目标分数。原本地候选尚未发布已由N607实时路径缺失证据核实。这个实验用于判断读出信息瓶颈，不能替代信道鲁棒架构的论文贡献。

对复混合矩阵M，功率读出含`|(Mz)_k|²=Σ_i|M_ki|²|z_i|²+2Re Σ_{i<j}M_ki conj(M_kj)z_i conj(z_j)`，允许相对相位影响读出，同时共同相位抵消。原逐通道功率/自相关读出没有这个跨通道交叉项；上游卷积仍可能表达它。因此该恒等式仅定位局部结构差异，不是整体不可表达性或新的一般理论。

[可执行公共反例](../experiments/cvs_neural_readout_identity/information.py)及[数值证据](../automation_reports/CV-SincNet/20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01/evidence/readout_information_counterexample.json)表明两类合法输入在原读出上碰撞、在一个合法复混合读出下可区分；没有拟合真实数据或修改正式初始化。是否学到有用区别仍由正式训练决定。
