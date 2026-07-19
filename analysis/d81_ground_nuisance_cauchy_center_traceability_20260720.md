# D81地面扰动谱稳健原型追溯

## 预注册机制

D81只从当前84-cell int8地面类中心构造逐类去中心的跨域质心漂移谱，不读取ground类别分数、类锚点、radius或count。对地面残差协方差的正特征值计算participation-ratio effective rank，并固定取`ceil(effective_rank)`个主方向；没有rank扫描。

对每个target类及每次full/block、outer/held support fit，先以该次fit可见的support均值为中心，计算每个物理样本在地面扰动谱上的能量。以类内平均能量自标定，一次Cauchy权重`raw_w_i=1/(1+energy_i/mean_energy)`形成稳健target中心。随后只把该类全部support的z160共同平移到这个稳健中心，FFT96/RF32不变。该平移严格保留类内残差和target协方差；D62最终query度量仍只由target support决定，ground不直接进入query分数。

## 要求—实现—验证矩阵

|ID|要求|目标文件|状态|验证/停止条件|
|---|---|---|---|---|
|D81-R1|真实84-cell组件只读，入口/出口hash一致|D81 probe/D66 loader|verified|完整105-row run入口/出口NPZ和manifest SHA一致|
|D81-R2|只用逐类去中心ground漂移谱，不用类别锚点/radius/count|D81 core/probe|verified|真实84-cell smoke；84 cell、radius/count=false|
|D81-R3|rank由effective rank唯一导出，无rank/scale/温度扫描|D81 core/tests|verified|真实effective rank13.6446→rank14；专项测试|
|D81-R4|每类一次Cauchy自标定权重，旧/新类同式|D81 core/tests|verified|类置换等变；无class ID/role/scene分支|
|D81-R5|只平移z160类中心，严格保留所有类内残差/协方差|D81 core/tests|verified|FP64残差误差≤2e−12；FFT/RF bitwise不变|
|D81-R6|每个outer/held fit仅用当次可见support重算|D81 probe/tests|verified|合成D62完整full/block OOF闭包；无held/query输入|
|D81-R7|K1/K2严格identity、0新增step/parameter/query MAC|core/resource audit|verified|K1/K2 bitwise identity；资源公式与专项测试|
|D81-R8|完整开发实验20-1/new5/K10/713101、3场景×5fold、105行|run/summarizer|verified|105行、1,080 component、2,160 transform及完整日志全部解析|
|D81-R9|相对D62严格联合门|summarizer/report|verified|`A+0.56pp`、`N持平`、H`+0.31pp`、F`−0.56pp`、old→new`−1`，所有floor/场景/混淆无退化|
|D81-R10|formal ground bundle需联合封存及外部authority签名|loader/report|blocked|当前只能development diagnostic|

## 停止条件

不扫描rank、Cauchy尺度、权重温度、平移系数、类别或场景权重。D81已通过单seed开发联合门，因此按预注册进入第二独立seed；不得直接进入125或作正式性能声明。当前追溯状态：9项`verified`、1项`blocked`。
