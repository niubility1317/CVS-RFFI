# F跨RX源特征：分离样本的类内精度几何诊断

状态：LOCAL_VERIFIED，正式缓存分析尚未启动。聚焦数学/权限负例检查PASS，唯一独立P0/P1审查PASS。当前弱TX改善目标未实现；不把本诊断当作识别性能结果。

只读取已完成的20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01三BASE源V缓存，每模型27000×160，原网络来源commit31467086c3ced30237f3ac86e91eae68645425d7、固定final50、源角色和物理划分不变。缓存元数据须确认V、stateunchanged、0headfit/optimizer/prototype decisions/target/truth/U_TX读取。当前不加载任何网络、checkpoint或IQ，不拟合分类头、不调用默认头、不预测类别。原F＋跨RX同TX、默认头联合、无星地增强/仅clean约定保持；本步明确仅源诊断，不做目标测试或训练。

## 固定计算

按固定partitionseed2026102009，将每个TX×RX×day的300个缓存包随机分为A/B/C各100；每模型三部分各9000。三模型使用相同标签/RX/day顺序和划分。A只估计协方差；B/C只提供各类、RX及RX/day均值和类内方差。组内缓存行不重叠；三个modelseed不是三批独立物理数据，分组也不代表独立采集环境。

所有特征逐包L2单位化。全部15个TX对均分析，不只保存弱14-7/20-19。每个TX对固定比较：RAW=I；TX=A中减各TX均值后的pooled covariance；TX_RX_DAY=A中减每TX×RX/day均值后的pooled covariance。后者仅解释RX/day成分，不作为备选晋级方法。协方差采用残差自由度分母N−groups，随后Sreg=.9S+.1trace(S)/d I。逆谱权重采用64步标量二分，使clip(lambda/eigenvalue(Sreg),.25,4)均值1；零协方差退为I。无超参数扫描、类别专属头或持久化训练状态。

仅B/C求跨RX类差内积（不同RX的deltaB dot P deltaC等权均值），除B/C pooled类内trace(P S)得SNR；各弱TX源RX均值相对其全局均值的B/C交叉散度，再除跨RX类差信号，得到RXspread/signal。保留有符号估计，不把负交叉散度包装成真方差。协方差/精度矩阵只在诊断内存中使用，不导入训练、推理或导出；仅保存标量谱摘要和各类对结果，不生成概率、argmax、分类准确率或任何读出参数。

## 读缓存前固定的判断

主假设只比较TX精度与RAW：弱pair1/3每个原模型seed的跨RX信号均为正，SNR至少RAW1.10倍，而且两弱类各自RXspread/signal不超过RAW.95倍（RAW该散度须正）。三模型全部通过才支持下一步单独预登记的L-only特征损失研究；不能据本诊断直接换头、宣称提升或启动额外训练。TX_RX_DAY结果全报告，不选择最高源分数作为替代路线。无目标访问或目标反馈。

历史E200白化锚点曾另外拟合读出，本实验不复用该网络或读出。高维类内多模态是传统Fisher分析的已知局限，参见[LFDA原论文](https://www.jmlr.org/beta/papers/v8/sugiyama07b.html)；本步不复现该论文分类器，也不把统计度量称为Fisher最优表征。源RX均已被原BASE训练见过，读出改善/未知RX准确率/84%与星地性能均不在本诊断声明范围。

## 执行与收尾

native Windows现有Python3.10/NumPy，单CPU进程、OMP/MKL/OpenBLAS线程2，无GPU context；保留所有其他工作。不可覆盖输出LOCAL_RUN/20261020-diagnostic-FcrossRX-source-withinTX-precision-m3-local-r01，CWD为IMMUTABLE_LOCAL_RELEASE/cvs_F_source_precision_local_20261020_r01。代码/config以实际Git commit唯一标识；一个归档复制后比较一次，发布目录compile一次。没有checkpoint或新模型使用，真实checkpoint无查询smoke不适用于该缓存统计阶段；launcher首步执行已验证数学检查，PASS立即分析，不设token或额外批准。

命令：LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_source_precision.run --base LOCAL_RUN/20261020-diagnostic-FcrossRX-source-withinTX-precision-m3-local-r01 --source-cache LOCAL_RUN/20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01 --commit RELEASE_COMMIT

元数据/角色/来源/类别RXday/count/order/输出冲突、非有限或无效散度、数学检查失败时停止本CPU运行并保留已写row，不自动重试。0新网络训练、0IQ读取、0U真值/target/分类头更新。逐row summary、完整result、completion/checks/execution和stdout保留；Git只交付代码/config/聚合统计/报告，不含源特征缓存、矩阵、权重、逐包产物或私有进程状态。完成后独立核实实际PID/CWD/cmdline/日志及产物，不将退出码单独视为成功。

检查已覆盖谱权重范围/均值、scale与正交旋转不变性、isotropic/zero covariance、分组平衡与不重叠、B/C改动不影响A精度、条件组平移、source决策正负及4个非法协方差/标签负例。唯一独立审查PASS，无确定P0/P1。没有新增泛化/性能结论。
