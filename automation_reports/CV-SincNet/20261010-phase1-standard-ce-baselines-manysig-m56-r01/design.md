# 常用网络纯CE固定对比

状态：LOCAL_VERIFIED；尚无新训练或测试性能结果。

用户要求常用网络、包含较大容量的1D和2D复数CNN、只使用CE，并在训练完成后自动测试clean。本实验固定14种网络×4个模型种子，共56行，每行从零训练，不以参数量筛选或扩大通道来凑规模。

## 架构依据与实际参数

|配置|结构依据及IQ适配|实际参数|
|---|---|---:|
|ResNet18-1D|标准BasicBlock，[2,2,2,2]，64/128/256/512；卷积/池化改1D|3,847,430|
|ResNet34-1D|标准BasicBlock，[3,4,6,3]，64/128/256/512；改1D|7,221,766|
|ResNet50-1D|标准Bottleneck，[3,4,6,3]，扩展系数4；改1D|15,966,982|
|VGG16-BN-1D|13卷积层，标准D配置，64/128/256/512/512；1D池化7，两层4096分类层|36,406,022|
|DenseNet121-1D|[6,12,24,16]，growth32，瓶颈4倍，transition压缩0.5；改1D|5,526,086|
|ConvNeXt-Tiny-1D|[3,3,9,3]，96/192/384/768，MLP4倍，layer scale1e-6；改1D，stochastic depth关闭|26,768,550|
|Transformer encoder|6层，d512，8头，FFN2048，ReLU/postnorm，dropout0.1；4点IQ为token|18,921,990|
|BiLSTM|3层，每向hidden512，dropout0.1；4点IQ为token，取最后双向hidden|14,743,558|
|复数ResNet18-1D|标准ResNet18拓扑/通道转复数Conv，split BN/CReLU，packed1024→6分类头|7,693,958|
|复数ResNet18-2D|同上，平方卷积核，IQ确定性排列为1复通道16×16|22,346,630|
|ResNet18-2D|标准ResNet18深度/宽度，首层2通道，分类头6类|11,176,454|
|ResNet50-2D|标准ResNet50深度/宽度，首层2通道，分类头6类|23,517,190|
|原CVS|既有lite_d/no_dac身份骨干，cosine scale30无标签margin|382,146|
|残差融合CVS|既有冻结结构residual_fusion，cosine scale30|164,225|

卷积骨干依据[torchvision ResNet源码](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)、[VGG源码](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)、[DenseNet源码](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)及[ConvNeXt源码](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)。标准层数与宽度保留；1D网络并非ImageNet原始2D模型，其参数量不能直接引用ImageNet表。

序列网络使用[PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)和[LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)。这两类架构没有唯一的分类器规格；上述深度/维度在target访问前固定。没有丢弃IQ采样，也没有生成额外训练视图。

复数网络采用标准ResNet转复数层的公开路线，见[torchcvnn IJCNN2025论文](https://liam-huynguyen.fr/documents/IJCNN-2025.pdf)。复数卷积采用标准实虚耦合乘法，归一化和激活选用[complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)中公开支持的naive实虚分量BN及CReLU思路。本实现明确是IQ适配版：独立分量最大池化及实虚打包线性分类头，不能声称与torchcvnn论文的协方差BN配置原样相同。曾拟定的三层CVCNN宽256草稿不进入56行矩阵，不用于正式结果。

2D输入为同一2×256 IQ按时间顺序reshape到2×16×16，无插值、STFT或合成图像。这个排列改变邻域关系，属于明确披露的架构输入适配。网络原有Dropout保留。最终对比包含分类头和输入适配差异，不能将差异全部归于卷积类型。

## 固定训练和测试

- 数据：沿用既有ManySig物理契约；6个TX，source RX1/3/4/6/8、day1/2/3；L_s6300、U_s56700、V27000。只对L_s反向传播，U_s不使用；V只评估，target不参与任何状态更新。
- 六类seed角色逐行见experiment.json。model seed为2026092701至2026092704，也控制初始化、Dropout和独立loader RNG；split seed392005固定；其他不适用seed为null。
- 所有行从零训练E200，每轮50次更新，batch128/drop_last=False，AdamW2e-4、wd1e-4、cosine最低1e-6，FP32关闭TF32，无梯度裁剪。
- 唯一目标为身份CE。无输入/信道增强、额外损失、PL、EMA、蒸馏、预训练或自适应训练日程。每行固定用自己的E200，不跨网络选模，不依据目标结果重排或重跑。
- 每行E200完成后立即冻结，优先进入clean预测队列；168000相同物理包、7个目标RX、6类统一竞争。预测固定后由独立进程接truth，输出Accuracy/Macro-F1及RX/TX分层。
- 最终给出实际seed数、均值/样本标准差，以及与两种CVS的同seed配对差；未完成seed明确缺失。参数、梯度使用参数、训练/推理耗时、显存及状态成本单独报告。
- 本轮用户指定自动clean测试，故只安排clean。该目标benchmark此前已暴露，结果是固定预算比较，不能冒充新的盲测，也不能外推为每个模型调到最佳后的性能上限。

## 已验证与运行交接

模型批128合成CE更新已通过；此前12项结构及两项新复数ResNet均通过有限输出/梯度检查。真实checkpoint严格加载与无query测试在远端launcher第一步对最终14项执行。4项协议负测通过：训练拒绝target/继承/额外损失、矩阵重复/越界输出拒绝、checkpoint冻结和来源校验、预测不完整时truth保持关闭。独立P0/P1审查及复数ResNet新增维度定点审查未发现阻断问题。

唯一launch owner：codex/root/standard-ce-baselines-20261010。新release/run/log不可覆盖；每GPU至多两个worker，健康任务保持原配置；仅技术故障使所属row失败，保留产物且不自动重跑。

远端路径：/home/szu2070436088/2510044040/CV-SincNet/runs/20261010-phase1-standard-ce-baselines-manysig-m56-r01。

下一步：提交/push及OID读回，发布不可变release，核实dispatcher和各row进程、实际参数及log增长。训练、clean预测、独立评分、同row报告及汇总由已实现队列自动衔接。
