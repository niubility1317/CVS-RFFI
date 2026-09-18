# DAOT＋FastTrust-RC4 practical四组实验

状态：LOCAL_VERIFIED，等待本次发布。

用户授权四个独立实验：full关闭均衡、full正则化ZF、residual关闭均衡、full MMSE。逐行配置以experiment.json及其config_ref为准。

ManySig equalized=1保持不变；开/关均衡指新叠加practical信道的接收均衡。

practical源码固定a8746f38；默认post_sync、星载RX、quality_adaptation、IQ/phase补偿和AGC保留；不是实测LEO。

频率fs=25MHz fc=2.462GHz，来源WiSig原文IV-B：均衡后重采样回25Msps；native loader不重采样。https://arxiv.org/html/2112.15363

ZF regularization=1e-6，ZF/MMSE max_gain=20dB，FIR129 delay64；unlocked跳过均衡，execution摘要区分configured/applied。

residual保留默认关闭均衡，≤1ms快照；full波形SNR与residual白输入期望质量估计差异保留，不宣称严格等价。

课程E1-40 p.30 high，E41-90 p.60 mid/low_urban，E91-200 p.80三场景；卫星CE E80 lambda.68，DAOT E21 clean+high。RC4 hard-sat路径仍关闭。

每组测试clean+本组practical三场景；E200附加既有LEO_WEAK参考。目标先预测再独立scorer，不参与选择/重跑。

无继承任何旧checkpoint；四个row各自独立初始化，使用相同model seed，不是4seed统计。

practical为CPU NumPy参考路径；保持信道实现，不替换近似GPU核。

## 验证

4组scratch checkpoint严格重建、三场景单2B前向和有限反传、DAOT E21 L/U真实目标通过。新测试5项通过，包括四路按opaque ID重排/分批复现及真实E100 predictor+独立CPU scorer。独立P0/P1审查结论见启动追加记录。
