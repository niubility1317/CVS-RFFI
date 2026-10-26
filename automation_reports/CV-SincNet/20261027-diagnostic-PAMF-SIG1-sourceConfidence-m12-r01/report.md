# 冻结E50源域置信度与解析logit梯度

状态：PLANNED，尚未执行。研究问题：0.95置信PL在合法源域的高覆盖/高精度，是否仍使主要错误留在拒绝池；对高置信伪标签加LS目标，解析logit梯度是否经常与硬CE相反。此诊断不证明实际参数更新方向或泛化因果。

只使用已完成的四方法SIG1-PL、SIG1-Sup、RXMean-U、RXMean-LU各三seed2026092701/02/03的固定E50默认头。只读输入原sourceV27000 clean IQ；每行先sourceL2实际checkpoint smoke，PASS直接继续。原6个PL/Sup source目录没有保存逐包源概率（已只读inventory核实），因此本次补充源域读出，0训练、0正式对照重测。为避免设备混用造成置信度细微差异，四方法全部统一CPU FP32、2线程，约324024个F包，0GPU，不改变当前GPU矩阵。

原完整PAMF训练权重、单SIG_TIME、原默认头全部严格加载。部署reader是完整训练模型的识别分支，不要求其参数数目等于863730；检查实际head6x160、SIG1池18976、一视图和有限输出。每行完整state_dict在source推理后逐Tensor保持，eval/inference_mode且requires_grad False，无optimizer/EMA/prototype/新头拟合。Sup还要求所有祖先的U/EMA原生标志关闭。

V标签只在固定预测后用于source统计，不传入识别器、不更新持久状态；U_s不打开、TX真值保持隐藏，target完全不打开。没有DURA或星地恢复输出，也不新生成H。尚未sourcefrozen的Mean03只等待其source complete.json，不用别的epoch或seed替代。sourceV是已有单一验证角色，不拆V_cal/V_select。

固定tau0.95、label_smoothing0.01、15个等宽ECE区间；保存默认头logits，按float64稳定softmax做解析统计。训练中真正接受率仍以各run日志为准，不能把CPU源代理接受率称为实际U真标签精度。指标包括覆盖、接受精度、拒绝池占全部错误比例、NLL/Brier/ECE，以及以自身argmax为伪标签的硬CE和LS.01的logit梯度点积/范数。负点积只表示该局部logit目标不同向，既不等价实际网络/AdamW梯度，也不自动说明LS有害。源域高置信不保证未见RX的高置信正确性。

全部四方法三seed均完成后才报告配对汇总，不选择最佳seed或据此修改正在运行的Mean/MeanMax矩阵。完整三seed正式clean/六景及原目标仍由原神经矩阵负责，本source-only诊断不代替其验收。精确路径/命令/角色/模型来源/停止规则见experiment.json，原目标active。
