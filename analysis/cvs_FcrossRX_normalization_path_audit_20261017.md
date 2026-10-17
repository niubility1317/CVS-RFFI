# F＋跨 RX：归一化调用链核对与下一候选

核对起点 commit：`b1856829c0c4cdac077f2a3b62f33c4b555b218c`。本轮只读取源代码、既有登记和固定正式视图的配置元数据，并在独立进程打印原构建环境的模块路径。真实 IQ 读取0、checkpoint读取0、网络前向0、优化器/EMA更新0；没有新实验启动、数据重建或测试修改。

## 结论

没有发现当前源增强与固定正式视图在外部 RMS 顺序上的错配。排除“通道后少补一次 RMS 因而导致差距”的实现解释，不重开已失败的 ordinary 输出 RMS 恢复或常数增益配方。

|环节|当前源训练/源读出|固定正式输入链|
|---|---|---|
|clean记录|EQ1、中心取256、normalize=True、center=False|原builder使用同类WiSigCompactDataset；out_len256、crop_mode=center、normalize=True，旧类center默认False|
|数据RMS|sqrt(mean(I²+Q²)+1e−12)，不是两通道均值、不是去DC|旧发布类的公式及默认值相同|
|星地前|手动normalize_input，单位复数RMS|batch接口normalize_input_rms默认True，调用相同公式|
|接收处理|完整native residual/post_sync，EQoff G=1，包含原生AGC|固定builder使用residual/post_sync/noeq原生接收处理|
|星地后|输出只堆叠/转FP32，不额外做包RMS|batch输出和视图builder只堆叠/转FP32；正式predict转tensor后直接前向|
|模型入口|既有pad_crop，无统一IQ RMS；常数增益的非平凡函数作用已有真实默认头检查|冻结预测直接送入原model，不新增后置RMS|

原生AGC属于接收机过程，不能和外部包RMS混为一谈。单位RMS也不等于总SNR；不能用包MSE或幅度统计直接推断噪声被修复。模型对常数增益敏感不证明这种增强有泛化价值。

## 保存配置确有的差异

逐字段比较全部六个源actual_config与正式received_views/manifest的config，只发现两项差异：源显式fc_hz=2462000000，正式为2450000000；源input_processing_state='unknown_ground_recording_processing'，正式为'WiSig_equalized1_center256_unit_rms'。源端真实预处理已由代码核实，不能将该文字字段当成未归一化的运行值。

相差12 MHz，约为正式carrier的0.49%。本轮未测量此差异的识别影响，不把它解释成P20失败原因，不做carrier扫参，不改变固定正式IQ或信道，也不以此重开P20。P20四组内部配置一致、源physics配对和独立评分仍有效；它不是与正式配置逐字段相同的目标成绩。

本核对只覆盖归一化顺序和保存配置，不证明两个发布版本所有数值实现逐字相同。已有VALIDATED_ONCE输入继续复用，没有增加数据重验或哈希链。

## 真实路径与证据

当前PairedChannel实际来自 `PROJECT/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01/experiments/cvs_crossrx_satmix20/channel.py`。它加载的leo_practical来自 `PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/`；源dataset_wisig来自同release的 `code/dataset_wisig.py`。

固定正式六视图的构建commit为`7669a6de973959290089a04293cc542db26e5fe1`，真实导入打印为 `PROJECT/releases/cvs_residual_sixscene_20261002_r01/code/leo_practical/channel.py` 和同目录batch.py。batch的normalize_input_rms默认True。正式预测代码只读既有npy并转FP32 tensor，不重新生成或归一化收到的IQ。

正式clean源于已保存的 `20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/clean.npy`。原builder代码位于 `PROJECT/releases/comparison_final_20260927_r01/tools/comparison_final_eval.py`，commit登记`614d79e5da86925e600444a2629ad89b7010fbd3`；只读代码核实其WiSigCompactDataset调用及当时dataset类的center=False默认值。没有读取clean.npy、index.npz或truth。

代码证据存于 [本地源代码/配置核对产物](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/normalization_config_metadata_audit.json)，以及同目录 normalization_actual_channel_source.json、normalization_formal_builder_code.json、normalization_formal_dataset_code.json、normalization_formal_import_path.json。原始JSON未提交Git。

先检索并读取的既有负结果：`20261016-phase1-F-crossRX-constantgain-sourceRX1-e20-s2026092701-m2-r01`；`20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r02`。前者模型有增益响应却未通过E20；后者两冻结模型四抽样全部变差。两者保持关闭。

## 下一唯一待检验问题：有界全通相位响应

登记检索 allpass、全通、群延迟均无匹配条目；这只说明未找到同名登记，不是方法新颖性证明。新问题是小幅频率相关相位响应是否能提供独立A_g效力，不是常数增益、逐时点phase乘法或已关闭的后置RMS。

全通滤波可具有平坦幅度响应和随频率变化的相位，官方文档给出这种构造及群延迟关系。[MathWorks Allpass Filter](https://www.mathworks.com/help/dsp/ref/allpassfilter.html)、[Group Delay and Phase Delay](https://www.mathworks.com/help/signal/ug/delay.html)。这些是算子背景，不是WiSig真实RX相位测量或RFFI收益证据。

候选从稳定一阶全通H_a(z)=(a+z^-1)/(1+a z^-1)出发。研究的是去掉已知纯数值延迟后的频率相关残差响应，必须单独核对边界、对齐、有限256点窗的功率变化，不能把无限稳态的单位幅度性质直接套到截断IQ；也不能用固定RX频域相位不可辨识的旧结果，声称已估计真实硬件。

最小下一步先预登记并执行纯合成G0，0真实IQ/CP/NN：检查a=0是否严格identity、理论/实际频响、既定小扰动上界、已知完整context与短窗边界的误差，并用纯延迟/线性相位控制排除“只是换了一种时间移位”。人工边界主导、有限窗不满足界限或不能区分纯延迟时放弃该固定构造，不调源数据上的幅度或窗口救跑。

G0通过也只证明工程构造可用。之后才可独立登记匹配来源的冻结F默认头源身份保护检查；失败不训练，通过仍需scratch E20的匹配raw/功率/延迟控制证明跨RX clean与practical保护。首轮U规则、NoExplicitCFO、joint160、默认cos头和网络结构不变。不提前宣称A_g有效，不直接堆到P20或其他已失败方案。本候选尚未实现、预登记或运行，无性能结果。

当前已完成的是调用链疑点排除。有效A_g、稳定A_s、四格交互、跨seed和正式目标结论仍缺，联合目标保持active。
