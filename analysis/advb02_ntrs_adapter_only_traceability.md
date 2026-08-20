# ADVB02 NTRS Adapter-Only追踪记录

来源：`E:/codex/home/attachments/eeadadf2-ccb4-4785-b2a3-7fd1d17b70e3/pasted-text.txt`

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|AON-01|七、八|q可训练且adapter-only反向隔离raw骨干/head|`code/ntrs.py`、dual model、训练器|pending|q梯度、raw零漂移测试|A1-R保留q冻结对照|
|AON-02|十八.2|q-only rank-8类共享低秩残差，不读取z_anchor|`code/ntrs.py`|pending|输入依赖和梯度测试|无独立头、无LayerNorm|
|AON-03|十八.3|卫星robust CE、clean-zero、relative correction|NTRS loss bundle|pending|损失分离测试|clean robust CE禁止|
|AON-04|十八.3|A2增加frozen/raw teacher KL和margin|训练器、launcher|pending|loss bundle测试|不使用target信息|
|AON-05|十八.4|alpha从0.02/0.05开始并记录修正分布/旋转角|模型、evaluation|pending|边界和遥测测试|不再使用0.20首屏|
|AON-06|十八.5|修正有效前always-on比较，不训练learned gate|模型、evaluation|pending|train/eval主路径测试|A3前无门控混杂|
|AON-07|十八.6|伪标签及原Core90几何使用raw/frozen空间|训练器|pending|训练主路径测试|adapter损失是唯一robust梯度入口|
|AON-08|十八.7|A4仅在通过后以0.01–0.05倍core LR联合微调且head冻结|训练器、launcher|pending|optimizer组测试|默认比例0.02|
|AON-09|十九|实现A0/A0-B/A1-R/A1/A2/A3/A4矩阵|launcher、报告|pending|dry-run逐profile|A3/A4顺序门控|
|AON-10|二十|六个性能门＋q梯度＋raw零漂移|evaluation、报告|pending|机制artifact测试|1–2pp需3次重复|
|AON-11|协议|seed392034、0.07/0.63/0.15/0.15、训练测试LEO_WEAK|launcher、负测|pending|launcher/协议测试|禁止mixed_orbit|
|AON-12|评测|E200后clean和三LEO独立测试及raw/robust转移|launcher、evaluation|pending|真实checkpoint smoke|缺测试不算完成|
|AON-13|声明|仅Phase1代理信道证据|报告|pending|报告复核|不声明真实在轨/Phase2/unknown|

最高风险项：adapter-only从旧D1 checkpoint加载时，必须证明新增q/adapter参数正确初始化、原raw骨干/head位级不变且不进入优化器。

