# Phase3 未见接收机星地开发域

2026-10-29，数据run `20261029-phase3-development-pool-rx24-1-r01`。原同RX clean开发最高92.6667%，十轮OE为92.0635%，不能替代跨接收机星地证据。本轮只制作独立开发数据，零模型／训练／适应／预测／评分，不启动新性能候选。正式RX24-18的U768/Q1920保持未模型访问。

## 容量与选择

复用Full既有VALIDATED_ONCE验证及counts/members索引，不重扫ZIP或重验原始IQ。开发RX为24-1（Full index26），与原source五RX、正式24-18、当前Phase2七RX均不交，且不在四compact RX集合中。只按集合和容量筛选，未看IQ或分数。其他compact外RX在相应日期缺完整旧六类；24-1的3月8日/15日足够沿用完整六类与原三台source novel。不能由此宣称穷尽项目所有历史曝光。

|角色|日期|builder私有TX集合|每TX每景|记录数|
|---|---|---|---:|---:|
|U_dev无标签发现/适应|2021-03-08|旧6＋19-15、19-16|16|384|
|V_dev独立开发验证|2021-03-15|上述8＋后续未知7-1|32|864|

旧六类顺序14-10、14-7、20-15、20-19、6-15、8-20。三景固定low suburban/mid urban/low urban，全部1248条物理接收记录跨角色和场景互斥。U_dev已知288、发现96；V_dev旧类576、发现192、后续未知96。U_dev每个RX/TX/date需48条，实际最小141；V_dev需96条，实际最小124。旧六类对应两日最小541/930。没有为缺量单元补复制、重选简单类别或按质量筛样。

这三台开发未知从未进入当前F或辅助OE训练，但已在源五RX做过重复novel评价；本次是新的RX/接收/日期条件，不是新未知TX总体盲测。U_dev一旦开放，19-15/19-16属于无标签见过；7-1仅V_dev读取，持续评价未注册未知。开发日期仍被原source其他RX使用，不声称严格未见日期外推。

## 处理与权限

每条从Full float64[N,256,2]按原iq_record转FP32[2,256]、一次unit RMS，不重裁/重均衡；随后原ResidualChannel，25MHz、2462MHz、residual/post_sync/noEQ、referenceSNR30dB、reflect。输出不再RMS归一化、不保存clean参考；RX/date/eq定义接收机会话，信道seed取决于canonical ID与scene，不用真实old/new角色。采样、scene循环分配与整体混洗对全部TX同式。

每role公共精确manifest.json/ids.npy/received.npy，phase3_min_v1仅初始六类/opaque ID，不公开真实未知数、配额、TX、scene或源路径。truth及canonical映射留独立评分/builder侧；V_dev状态冻结后逐样本预测，完整后truth-last，不回收训练，不用标签调阈值或早停。算法从U_dev自行估计簇数；正式四个未知11-4/3-13/12-20/5-20不进入本开发池。

builder只读取索引中的两PKL成员，检查本次选中记录/角色/有限值并标VALIDATED_ONCE。原source、原正式池不重建。此数据切片的成果只证明入口合法可用，不代表95%拒识、无标签适应或注册成功。本机CPU构建、GPU新增0，不停止其他owner。实际代码、测试、一次P0/P1、进程/日志/产物证据与状态维护在同名登记，后续方法另立run。
