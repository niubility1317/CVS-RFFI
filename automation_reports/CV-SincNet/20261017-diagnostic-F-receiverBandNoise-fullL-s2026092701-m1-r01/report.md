# 等期望功率的接收带限观测噪声：完整 L6300 源检验

状态 **IMPLEMENTED／LOCAL_CHECKED；未启动真实源检验或训练**。基座仍是 F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、默认cos30联合头、NoExplicitCFO及必要逐包补偿。E30预算保持；本行0新modelE/更新，只用已核实完整源REPEAT source-final20做冻结默认头检验。全部5源RX1/3/4/6/8和L6300使用，不留训练RX，也不以L5580/Ldev720替代。

## 唯一假设与依据

假设：在当前EQ1输入中，新增观察噪声的谱支撑应受接收处理带宽约束；与相同SNR和总期望功率的白噪声相比，正确的粗频带先验可能支持更合适的A_g。此处尚未证明跨RX泛化或任何星地收益。

[WiSig 官方均衡实现](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_channel.m)先转20MHz，在64点频域中保留±1至±26号子载波，进行MMSE式均衡，再回25MHz并保留频偏。由26×20MHz/64得到8.125MHz的粗频带边界。本候选只借用谱支撑；未知每包均衡器、CFO和窗口效应未恢复，不能声称是原噪声的精确协方差或真实RX替换。当前本地实际args为wisig_equalized=1，loader先取float32再逐包RMS（center=false），这也使EQ后取整不能直接解释为ADC更换。量化/截幅/DAC候选未启用。

本候选与旧三条路线区别明确：不对已接收IQ做DCT/FIR滤波；不从已失败的周期残差关联估计有色噪声；不以更高SNR救旧白AWGN。本行沿用旧20至36dB的完整强度范围、保留全部Gaussian尾部，只改变独立新增噪声的相关结构。原固定白AWGN训练负结果继续保留，源身份保持通过不能预告新训练成功。

## 算子与最小比较

令C[i,j]=sinc(2×8.125MHz/25MHz×(i−j))，长度256。C是矩形连续谱的有限Toeplitz协方差，Cii=1；用其唯一PSD平方根着色proper复Gaussian噪声。仅数值负特征值（量级约机器精度）截为0，并验证重建误差；不截断或重抽任何噪声样本。白／带限使用相同底层Gaussian向量、同一个逐包SNR与raw功率，因此期望总噪声功率相同；实际有限包噪声功率分别记录，不能偷做逐包能量匹配来降低噪声。没有改raw、crop、padding、RMS、PA/TXIQ/量化/CFO或任何practical调用。SNR范围有界，Gaussian振幅本身不严格有界，全部尾部保留。

固定模型seed2026092701／原split392005；raw一次，白/带限各两次命名独立draw，所有5×6300源判决完整保留。31500源NN＋真实同契约CP的2个合成smoke；1CP、1源cache、0教师/头拟合/梯度/训练/卫星视图。已有carrier的V特征会反序列化后立即不用；U仅原不透明role-ID排除，不读U IQ/TX真值；V无统计或更新，target完全不访问。Gaussian生成只按不透明物理ID/draw命名，不依赖标签或模型预测。

每draw带限相对raw：accuracy和macroF1下降均不超过0.5pp、原正确集合保持率至少99%、各RX/TX/day accuracy下降不超过1pp；每RX的源IQ粗频带内功率中位数至少90%；实际扰动中位数至少1%、最大不超过15%。白噪声是等功率观察控制，不用其结果更改范围。所有样本/全部draw都计入，任一固定条件失败关闭本固定协方差/强度/seed/频带候选，不调带宽、SNR、窗口、阈值或RX/尾部救跑。通过也只表示源身份与粗谱先验可用，仍需独立、同预算/数据/初始化/普通头/U规则的完整E30四因子识别实验；当前launcher不会自动训练。

## 预登记与执行边界

实际命令、不可变run/release、CPU2/CUDA隐藏、来源及stop rule见 [experiment.json](experiment.json)。本行使用原fullsource REPEAT最终20轮checkpoint，加载前核对scratch/固定final/无祖先/无目标选模和完整6300/56700/27000物理角色；不使用旧留RX折或按target挑出的候选父模型。原immutable fullsource runtime严格固定，本行入口不调用target evaluator或loader。一次代码/config提交后，单份归档、一次普通N607预检、一次archive传输比较和远端编译；唯一P0/P1只查直接影响下一真实运行的缺陷。源smoke通过立即继续，无额外审批或seal。

独立合成检查已实际PASS：10输入反例、16白/带限对，raw不变、命名RNG与批顺序独立、全局RNG不变、私有draw刷新；协方差最大误差2.442491e−15，总期望功率相同；有限256窗带外期望功率比例0.299142%，白噪声34.765625%。0真实IQ/CP/NN/更新。这是算子数学证据，不是身份或泛化结果。

最近实际完成：真实预处理核对、不同算子实现、合成检查和源预登记。被否定的是EQ后直接取整可当作ADC替换的解释；尚未证明带限观察噪声有效。当前唯一候选如上，最小实验和放弃条件已固定。完整目标仍ACTIVE。

唯一独立P0/P1静态检查PASS，构建接口、来源、31502计数和发布路径未见直接缺陷；审查者本地Python缺NumPy，未完成其自行数学复算，不能将该项写作其测试通过。本工作者已用已验证科学Python实际完成上列数学检查。无真实IQ/CP/目标访问。
