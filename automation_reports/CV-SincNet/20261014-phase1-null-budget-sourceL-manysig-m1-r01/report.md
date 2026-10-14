# 周期参考正交补＋停止预算：完整源结果

ANALYZED / VERIFIED，科学决定`REJECT_NULL_REFERENCE14_BUDGET_PRETEXT`。完整6,300包可见有效，但新14维观测在所有源RX都不可稳定预测，跨RX身份关系也未通过。该版本关闭，不训练decoder/backbone，不扫bin/mask/预算倍数/阈值。

Run `20261014-phase1-null-budget-sourceL-manysig-m1-r01`；执行Git `3b92b5eff1de92e639f0b9f5b3c0379af86bd17d`。固定source L_s TX0至5、RX1/3/4/6/8、day1至3，90cell各70包，原物理IDexactmatch。无modelseed/checkpoint/U/V/target输入。固定visible0/1/3、held2，FFT20/sqrt20 bins7..13；visible phase/RMS，不拟合gain/delay。zero-reference、cycle1和mean同包同valid。仅周期稳态线性公共参考加DC子空间被数学投影，未证明TX独占、RX移除或缓存噪声独立。

## 可预测性与身份关系

zero-reference MSE .00528369；mean .01003627，为zero的1.89948倍（误差增加89.9480%）；cycle1 .01540849，为zero的2.91623倍。mean中位数 .00123521，q99 .09901102，前1%包占22.8089%误差；完整独立tail复算核实。可见均值总体改善≥10%与所有RX不反向均失败。

126,000固定关系全部有效，跨RX同TXcos .23669924、参照RX异TX .23799782、gap −.00129858；62,192条（49.3587%）gap≤0。只3/5 RX gap为正，未达到gap≥.02和至少4/5正条件。同RX异TXpair平均cos .27698248，仍高于跨RX同TX .23669924。该关联不是因果或新RX泛化结论。

| RX | mean误差下降 | 身份gap | mean预算active |
|---|---:|---:|---:|
| 1 | -311.6498% | +0.00155129 | 1.4286% |
| 3 | -84.0872% | -0.00741985 | 43.8095% |
| 4 | -71.6634% | +0.01589542 | 3.8095% |
| 6 | -118.5043% | -0.01998672 | 12.5397% |
| 8 | -89.9117% | +0.00346696 | 34.8413% |

## 停止预算与退化基线

固定v=可见三重复样本方差、b=4v/3、s=mean(m²)、D=s+b+1e−8、w=s/D、a=relu((MSE−b)/D)，损失.1*sum(w*a/(1+a))/完整包数。观测全detach，满足预算梯度0，单包贡献有界。这些数学检查通过，但只是观测容差。

meanweight .15602905，meanactive 1,215/6,300（19.2857%），loss .00134198；zero-referenceactive仅485（7.6984%），loss .00052566。zero预测在5,815包（92.3016%）已满足预算，显示预算统计有非零信号不能证明值得学习的身份监督。mean有界loss前1%贡献21.2984%。三个预算可用性条件通过，不能弥补可预测性和身份关系失败。

## 验证、成本与后续

一次P0/P1 review PASS；preflight、单archive SHA、compile、独立livePID/CWD/argv/log增长及自然terminal产物VERIFIED。独立freshCPU重读census，135项分组均值、全部126,000关系、6,300包visible signal/variance/budget、18项预算分组和24项overall tail核实，0新增IQ。

实际6,300源IQ读取、2零IQfixture、0network forward/optimizer/EMA/head/decoder fitting、0U/V/target/GPU，主要源处理3.3517秒。原census/runtime/log保留不可变根；公开[source_result_evidence.json](source_result_evidence.json)、[误差分组](baseline_source_metrics.csv)、[关系分组](source_relation_metrics.csv)、[预算分组](budget_source_metrics.csv)。

两次原始重复观测路线至此关闭，不能因为损失设计精细继续强迫骨干拟合这些目标。下一训练机制主线转向SIG_TIME真实中层特征的有停止条件语义排序，并检查辅助实际AdamW增量对其他源RX的作用；只用合法L训练证据，不加伪标签判定链、不另训头、不把临时拟合读出当部署特征。排序/SupCon及跨域梯度思想已有文献，拟研究的是固定默认头评估下、实际优化器状态中的跨RX辅助增量贡献，创新与效果均须独立消融确认。

SIG_TIME、COMPACT、LONG_SILU、Sinc既有正收益按各自配方保留，不相加、不按target选择父权重。默认分类头原joint训练保持，三seed84%研究目标active。
