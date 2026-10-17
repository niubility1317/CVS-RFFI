# NoEQ／MMSE24 接收处理混合：同父模型累计 E30

状态：IMPLEMENTED／LOCAL_CHECKED；新训练尚未启动。上一轮完整源 MMSE24 输入干预通过固定条件，但这不等于训练或正式测试收益。本轮唯一假设是：在已有完整 practical 中增加接收处理多样性，可以改善同一个 F 在仍保持 NoEQ 的正式测试上的识别，并保留 clean。

## 唯一训练行与匹配参照

新行 MMSE_MIX-s2026092701：对应普通星地 CONCAT source-final20 及自身 EMA 追加10轮，累计 E30／6660更新，新增2220更新。与既有 NOMINAL_AS E30 共用同一父模型、初始化／EMA恢复规则、空 AdamW＋原基准学习率重置 cosine10、原 E13 普通增强冻结日程和累计 E21至30 的 SupCon 权重0.1。不是从零E30或恢复原optimizer轨迹，不继承源诊断或PDP变体权重。

模型仍 F＋跨RX同TX，三SIG_TIME、统一Sinc CENTER_25K／CENTER_UNITS、joint160、默认cos6×160 scale30与骨干联合更新。NoExplicitCFO和必要逐包补偿保留，0新参数／部署分支／头／损失／新U规则；RDrop是既有随机子网络正则，不称为IQ A_g。

全部5源RX1/3/4/6/8、day1/2/3、L6300/U56700/V27000、split392005保持。不存在留出训练RX或内部L5580/L_dev720。U TX真值隐藏，原旧门控／自身EMA／batch_neighbor半监督不变，不增加U星地或teacher视图。V仅原健康观测，不反向更新。

## 实际 IQ 与损失

每个L先保留原普通增强IQ的两次RDrop前向；原始L独立生成一次完整practical IQ，再复用它做两次RDrop。总监督仍0.5 RD(ordinary)＋0.5 RD(As)＋0.1 crossRX SupCon，按物理L和每对视图原规则归一化；原监督份额与总损失尺度不变。全部有标签As参加监督，不看困难学生置信度。

唯一改变：私有、与TX标签无关的逐物理样本／累计step抽样，以0.5／0.5先验选择NoEQ或现有MMSE24。传播scene选择、H、遮挡／几何／阴影／散射、原噪声N0、RX/day session硬件及标定、CFO／相位／IQ补偿和质量依赖沿用。MMSE既有FIR长度129、增益上限20dB、H加误差NMSE−25dB代理及质量scale不变；目标延迟24点在源诊断前由NoEQ共同数值延迟固定，64点不作为候选。选择MMSE但unlocked时仍按原规则跳过EQ；G同时作用信号和噪声，AGC按处理后的实际输出重新计算。不增SNR，不裁剪／删除困难包，不连续叠加两次完整星地信道。

实际接收处理发生在新的L-only IQ adapter。native未消费的practical_EQ参数仍为false，不能据该参数误判实际关闭；actual_channel、每step选择／生效数和各RX/day/scene/state/lock覆盖记录实际执行。共同PA/DAC/class-signature/CFO等旧普通增强仍按真实parent args原样记录，不擅自更改或称其已验证身份保持。

## 预算、自动测试和放弃条件

新增每模型10轮／2220更新，1136640个L NN前向、284160个完整As观测；原U／EMA／验证、真实checkpoint smoke和测试分别计数。MMSE额外CPU处理、实际EQ生效数、信道墙钟及总墙钟单列，不声称与NoEQ浮点计算成本完全相同。父20轮／4440更新单列；既有NoEQ E30控制不重训或重复预测。

固定final10／累计30冻结后自动完成原168000个物理样本×clean和六种practical的全部prediction/logp，再exec为独立CPU scorer连接truth。测试received IQ、SNR、信道、困难样本和分类头均不改。精确命令、环境／CWD、独占输出root、GPU容量选择和工件见[experiment.json](experiment.json)。每GPU跨owner总进程≤3、空闲≥12000MiB，实际GPU启动后读回；不停止或热改任何健康任务。

预先固定的否证比较是MMSE_MIX−同父NoEQ E30：low urban至少+0.5个百分点，clean和mid urban不下降，每个suburban不下降超过0.25个百分点。联合目标另要求相对同4L B00的clean净正、low>60%、每suburban与自身clean差距探索目标≤2个百分点，不能靠降低clean缩差。任何正式结果不回流混合概率、NMSE、SNR、延迟、loss、父模型、seed或重跑；不自动扩种子。本行只检验 A_s 的一个接收处理变化，不冒充新的A_g＋A_s四因子完成。

## 已执行验证与当前边界

本地33个协议／预算／来源／数据反例通过；实际训练adapter30个合成样本，15个逐值重放（6个NoEQ、9个MMSE24）与原完整core最大误差0。输入不变，批顺序重排一致，Torch／NumPy全局RNG不变；真实F结构900722参数、joint160和默认6类头的2个合成前向通过。新增检查0真实IQ／checkpoint／训练更新。旧Torch与NumPy2的ABI警告保留，已验证路径使用值拷贝／frombuffer，检查exit0且PASS，不修改环境。

发布前仅对新训练接线做一次有界P0/P1检查，不重复审查已通过的源候选或原物理core。单份archive一次传输SHA比较、普通N607预检和remote compile，launcher真实同契约checkpoint no-query smoke PASS立即继续。源角色／CP继承、非有限、预算、输出冲突或系统性技术失败保留本run产物并停止新派发，不自动重发；观察超时先核实原进程和产物。

当前未训练、未预测或评分，不能把源MMSE24输入收益提前当作本模型成绩。

新训练接线的唯一独立P0/P1检查PASS：18文件语法和登记一致、2220更新／284160 As预算复算、私有抽样／单residual／累计偏移／receiver_calls／同父ownEMA／七视图后独立评分均核对；审查者未读取真实IQ／CP／target或执行远端操作。
