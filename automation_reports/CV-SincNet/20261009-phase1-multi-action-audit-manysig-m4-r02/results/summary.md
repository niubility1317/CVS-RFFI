# 多解耦下一步：完整源端作用诊断

4个固定multi E200身份骨干×clean/source practical_mid。每行1920拟合包、960独立审查包；身份骨干原训练见过这些源包，因此不是全模型未见数据测试。辅助拟合总计14400步，所有行与步骤完整。
三档L/T从相同初始化和样本日程开始，固定200步；这是短预算固定参考诊断，不能等同原E200在线辅助模型的复现或新身份训练。没有使用目标评分、没有新测试集准确率。

下表使用另一TX在**相同干预参数**下生成的q。skill=1−预测MSE/零预测MSE；以百分数展示，越大越好，负数表示不如零预测。它不是识别准确率。±为四seed样本SD。

|源视图|分支|拟合目标|h作用skill（%）|G后作用skill（%）|margin变化MAE|
|---|---|---|---:|---:|---:|
|clean|linear|legacy_self|16.36 ± 2.20|40.25 ± 1.00|0.3836|
|clean|linear|block_z_self|5.04 ± 1.17|39.25 ± 5.15|0.4021|
|clean|linear|block_z_cross|4.63 ± 0.96|37.88 ± 4.86|0.4122|
|clean|temporal|legacy_self|21.42 ± 2.94|11.05 ± 3.96|0.4370|
|clean|temporal|block_z_self|5.55 ± 0.79|12.15 ± 2.15|0.4480|
|clean|temporal|block_z_cross|5.02 ± 0.67|11.67 ± 1.88|0.4509|
|source_practical_mid|linear|legacy_self|12.12 ± 1.19|11.54 ± 2.12|0.4529|
|source_practical_mid|linear|block_z_self|5.69 ± 0.76|14.67 ± 3.65|0.4511|
|source_practical_mid|linear|block_z_cross|7.22 ± 1.19|16.66 ± 4.15|0.4447|
|source_practical_mid|temporal|legacy_self|16.18 ± 0.71|10.76 ± 3.06|0.6179|
|source_practical_mid|temporal|block_z_self|9.16 ± 0.85|11.29 ± 0.53|0.6278|
|source_practical_mid|temporal|block_z_cross|8.26 ± 0.89|10.61 ± 0.53|0.6316|

## R组级作用诊断

R拟合整体条件匹配源RX变化，未扣除未经验证的L/T外推。组间没有逐包反事实配对。真实参考使用mean G(h_i)及平均逐包margin；完整产物另列质心近似误差。

|源视图|作用|h作用skill（%）|真实组margin变化预测MAE|
|---|---|---:|---:|
|clean|learned|22.57|0.9767|
|clean|zero|0.00|0.7657|
|clean|fit_mean|0.00|0.7657|
|source_practical_mid|learned|22.09|2.8639|
|source_practical_mid|zero|0.00|3.0615|
|source_practical_mid|fit_mean|0.00|3.0615|

## 解释与后续边界

源practical_mid的L分支在加入跨TX拟合后，G后skill比原目标更高；clean上没有同向改善。T没有显示同样收益。R在h空间能超过零预测，但clean组margin误差反而更大，支持继续区分作用拟合与身份方向约束。
不据这轮短拟合宣称新网络提升识别性能，也不直接提高辅助权重。报告下一阶段的身份训练、真实CE与可靠性拆分、原关系/类别锚定消融和完整truth-last测试仍需独立预登记；LT强化留待组合诊断证据。
完整逐seed/self/cross/shuffled/oracle/zero/mean指标见action_metrics.csv，R见receiver_metrics.csv；每行原JSON保留谱、路由、边界、组合、梯度与覆盖记录。所有负结果保留。
