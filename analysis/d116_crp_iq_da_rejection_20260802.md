# D116-CRP输入域接收机响应适应理论审查

状态：`DESIGN_REJECTED / P0=4 / P1=3 / IMPLEMENTATION_NOT_STARTED / NO_PERFORMANCE_RESULT`

日期：2026-08-02

## 1.原候选

D116-CRP拟在冻结encoder之前，用sealed Phase1旧类log-spectrum／group-delay模板减去当前row六个旧类K-shot support的类条件谱，鲁棒估计共同低阶实cepstrum与线性相位，再构造固定、稳定、最小相位FIR，对该row全部support和每条独立query的固定received IQ施加同一后接收数学变换。该设计不改feature、logit、qKNN、metric或head，query不参与滤波器拟合。

作者给出的K1条件是：六个旧类各提供一个独立物理support，类模板完全吸收TX谱形，LEO随机通道残差跨类近零均值，且leave-one-old-class-out滤波器稳定。没有与checkpoint共同封存的合规Phase1频谱／群时延模板时，候选不得实现。

## 2.独立反方P0

|编号|问题|裁决|
|---|---|---|
|P0-1|builder为每个物理样本独立抽取LEO弱信道参数；共同seed只保证随机序列可复现，不代表support与query共享同一信道|六类残差不能被解释为一个共同channel realization|
|P0-2|support残差同时含TX调制／符号谱、day变化、CFO、相噪和逐样本LEO信道；无pilot、clean或重复校准时，`R_receiver`与这些项不可唯一分离|共同receiver响应不可识别|
|P0-3|log幅度存在增益gauge，群时延存在常相位／线性相位gauge和all-pass非最小相位部分；当前协议与bundle没有给出最小相位或时间基准假设|稳定逆滤波器没有唯一理论对象|
|P0-4|现有sealed Phase1 bundle没有已验证的raw-IQ频谱／群时延aggregate；运行时回读source或clean来补模板违反`p2_min_v1`|当前资产面不足，不能先实现后补封存|

## 3.P1与历史差异边界

- 同一可逆LTI滤波同时施于support/query，在冻结encoder近似线性时会退化为固定频域PSD几何；若没有新的可识别target状态，容易重复D93/D94 transport和D110 metric的负路线。
- D113已证明共同feature平移可以大范围改变连续值但不跨决策边界；把共同作用面提前到IQ并不能替代receiver/channel可识别性证明。
- K1只有六个旧类support；new support不得进入公共状态，否则new5/new10/new20会改变old before/after预测状态。

## 4.裁决

`REJECT_D116_CRP_AS_UNIDENTIFIABLE_DA`。不创建Phase1模板、不实现FIR、不运行真实forward G0，不扫描cepstrum阶数、收缩、可信频带或逆滤强度。

反方提出的固定逐记录归一化自相关view`r_l/r_0`对复标量增益和常相位严格不变，只读每条fixed received IQ且query无状态；它最多是一个域不变输入view，不估计target域状态，不能直接标为D116域适应。若后续研究该view，必须作为新的表示／head因素单独分类并证明不重复现有same-IQ view，不能用它挽救D116。
