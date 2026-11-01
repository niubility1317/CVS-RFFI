# 相关性先验物理恢复：完整源训练核实

**SOURCE_TRAINED 已核实；完整实验仍在执行 Main 预测，尚无本版正式准确率。** 全部学习参数从零单阶段30轮/7410更新，原Ls6300/Us56700、20份R07模拟再重复前10、同原M20数据顺序与LR。

![完整源训练](E:/type10-7/automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01/full_training_curves.png)

| 项目 | 实际值 |
|---|---:|
| 训练秒数 | 1932.094824 |
| worker秒数 | 1935.278144 |
| 轮时中位数 | 64.442973 s |
| 峰值已分配显存 | 1003.646 MiB |
| 源Ls/Us曝光 | 189000 / 1701000 |
| 学习参数 | 1230341 |
| 源mu/C固定buffer | 1050624字节 |
| G更新期间F0前向 | 0 |
| G更新期间后验求解 | 0 |
| E30 Action全矩阵相对MSE | 0.0097250868 |
| E30噪声logMSE | 0.0497459843 |

训练损失为.1×真实源Action全512实basis矩阵相对MSE+.001×噪声logMSE，实际只调用G.estimate。逻辑batch256/micro32按真实逻辑batch计数累计，每更新只clip/step一次。图中误差不是恢复IQ的NMSE、真实TX识别率或信息恢复比例；没有以这些下降值选择轮次、改参数或取消Main。

原63000源clean拟合的统一mu/full512C为额外数据统计知识，固定.001向同trace先验收缩，FP32 buffer、FP64部署后验。不是G checkpoint祖先，也不是零先验知识；统计拟合及预存R07准备成本未单独重新测量。F0只在source2真实检查和冻结后预测使用，不参与G更新；方法依据确实参考过冻结源F0诊断，不扩大成全流程无F知识的主张。

CPU独立复核全部30日志/stdout、sourceorder/draw/LR/加权loss，最大加权loss差1.12e-09；实际E0学习状态等于factory，mu/C等于原FP32源统计，E01/E30先验不变、TF骨干与输出head更新、optimizer有限、30技术快照存在且E30与final一致。真实source2完整后验/F0、算子head梯度和严格E0重载已通过。复算0新神经前向/IQ或truth读取。

源训练worker终态0，实际预测worker已从/proc核实接续；同一地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×7继续自动执行，全部prediction固定后才连接独立truth。当前仅source训练完成，六景3pp与两个clean保持尚未取得本版证据。默认DURA和完整目标ACTIVE保持。
