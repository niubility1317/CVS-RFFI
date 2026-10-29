# G10 开放集表征学习与匹配对照

上一轮 RX/day 投影保住了源类均值的线性空间，但重新单位化仍改变几何，且主未知拒识低于匹配控制。下一候选直接更新表征，检验源辅助未知成员监督能否降低错误已知置信，同时保留未知记录间差异。所有目标及联合指标保持，不从目标 Q 调参。

固定三组，均独立从原 Phase1 完整六类 F0 E50/11100 严格初始化：anchor 使用源 L 的两视图 CE 加原 F0 同输入 logits 锚定；oe 在相同已知骨架上加辅助源未知的 uniform-six KL；oe_vic 再加 raw fused160 的方差、协方差与两视图一致性。唯一主法预先固定 oe_vic/P1/energy，另外两组是机制对照。anchor 含 KD，不称纯 CE；新配方是 Phase3 增量候选，不改名为原完整 Tri-SIG-TIME SSL。

自监督目标借鉴 [VICReg 原论文](https://openreview.net/pdf?id=iWpcWZ8phD) 和[作者实现](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py)的 invariance/variance/covariance 思想。这里在原生 raw fused160 上直接施加25/25/1三项、整体权重0.1，无 projector、LARS 或分布式 gather，因此是项目适配而非完整复现。raw fused160 没有最终单位化；std目标1数学可行，但不意味着一定改善未知身份。源 heldout 的跨 RX 最近邻仅是保真诊断，不能代替独立未来注册。

新增模型消费 IQ 全部先G10，沿当前固定MIX01场景3001/信道4001/receiver2027。训练只消费冻结 DURA 的 P1 计算结果；两视图是同一物理 P1 的原值与全局复相位/微小增益变换，不再叠信道、裁切、补噪或二次RMS。L/OE增强各有独立随机流，L批次/增强在三组严格匹配，OE批次/增强在后两组匹配。

L仍为原五源RX、三日的6300条真实标签；原U_s不打开。OE为既有21辅助设备6134条原索引，只有“初始六类以外”的成员监督，真实TX/K不进入训练；95个空cell及不均衡domain不复制补齐。它与历史OE4重叠537物理记录，不声称历史独立。V27000和novel3150均不优化、早停或拟合持久判决参数；novel的三TX曾有开发暴露，不称新TX盲测。正式/开发目标U/Q访问均0。

每组固定2220次更新，累计13320步；这是优化步预算，不是10个自然数据epoch。L128、OE128，fresh AdamW CENTER_UNITS、lr1e-4常数、继承parent weight_decay、一次physical clip5。所有F参数及默认六类头可微更新，但用eval模式关dropout/BN更新；不用旧native六前向hook、As、CRIC、RDrop、U/EMA/PL。旧clean阶段E51/E60及其优化器、几何和判决状态不作为祖先。完整trace报告实际loss、梯度、head/Sinc变化、采样和前向曝光。

缓存阶段原F0+G先通过合成星地P0/P1 no-query smoke，再生成42584条计算记录（L6300+OE6134+V27000+novel3150）的固定P1；原F0只为L缓存6300条teacher logits、为V缓存两path54000条closed baseline。训练阶段每个新native构造器再用2条合成G10 received做F梯度smoke；该检查不再运行G，不冒充第二次P1路径验证。所有smoke恢复state/RNG，PASS即继续，无许可token。

每组冻结终点后，L重新拟合原八分支geometry与五分支MD/RMD，不使用失败的nuisance投影。V只在same-path原F0 baseline约束下校准42组的七分数阈值；新encoder闭集损伤过大时报告无可行阈值/科学false，不能换成自身闭集隐藏损伤。然后才读取novel公共IQ，P0/P1全部预测完成后由独立source scorer连接私有truth。宏A_K、C_K、R_U、H_OS、AUROC/OSCR及完整分组全部报告。

只有预指定主法通过源总体、5RX、6scene拒识90%及42known保护，且三组完整，才取得后续冻结小目标矩阵的source资格；源成功仍不代表跨RX或总Phase3完成。无标签适应、发现、匿名注册和未来80%等完整目标保持。本阶段未执行的A_N/H_OW为N/A；不以source邻居命中率代替。

两个run分别登记：本机CPU数据 `20261029-phase3-star-source-exposure-g10-r01`；N607缓存/三组训练/逐组自动预测评分 `20261029-phase3-star-representation-s2026092701-r01`。每卡最多新增2本任务GPU进程，矩阵最多3卡各1worker；启动查实际余量，不修改外部进程。代码、命令、环境、CWD、输出、技术停止和预期产物在最小登记固定。技术故障保留该row，其它健康row继续；科学失败完成评分、不调参、不扩seed、不访问target。
