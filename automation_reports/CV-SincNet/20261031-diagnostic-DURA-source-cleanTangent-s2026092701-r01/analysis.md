# clean局部身份敏感度：通过固定源研究判据

VERIFIED / ANALYZED，SOURCE_ONLY。固定纯repeat E30源native IQ误差，512条训练池记录（128L/384U）×原缓存01/20。原F50/defaulthead冻结，原头rank5对比方向Q不拟合，phi(x)=4·normalize(Ffeatures(x))·Q，在sourceclean计算5×2×256 Jacobian。新局部度量为||Jclean·(restored−clean)||²，J与参考detach，训练时可只计算固定二次型。原有F特征/后验方案在restored处反复计算非线性梯度，两者不是同一个目标。

|源缓存|tangent与exact梯度平均cos|IQ与exact平均cos|tangent非负|各RX最小均值|线性近似相对误差中位数|
|---|---:|---:|---:|---:|---:|
|01|0.643863|0.071039|493/512|0.538053|0.531591|
|20|0.661080|0.070946|484/512|0.543571|0.525913|

两view全部512cos可定义；平均tangent≥0.5且比IQ至少高0.1、至少384条非负、五源RX均值非负，均通过事前固定规则。输入梯度范数比中位数分别74.547379/75.460251，据预定公式固定alpha=0.0033130025035573727，不是根据目标分数或后续训练调整。它使较大中位比view的新增梯度约为IQ梯度0.25倍，不保证每条或G参数梯度都有该比例。

局部近似相对误差中位数仍约0.53，不能称全局精确的特征约束；正cos不保证优化完整G后泛化。512条属于源训练池，不是独立V，更不是真实TX准确率、目标提升或恢复极限。资格只支持另行登记从零单阶段30轮，源规则没有自动触发训练；正式收益仍须完整七景truth-last，默认DURA保持。

实际CPU循环5.796346秒，其中512sourceclean J缓存3.690889秒、5.24288MB。主F输入1536、首步smoke6；cleanVJP2560、smokeJVJP20、restoredVJP1024；G新输入/训练更新/TX/V/target均0。F全state不变、无参数梯度累积。全63000缓存645.12MB、30次读取19.3536GB仅为数值体积推算，尚非GPU成本/加速结论。源F既有50轮/11100更新知识和成本保持披露，不能称F-free。

三份私有完整NPZ已下载，512源ID/角色/RX顺序匹配，tangent loss/inputgradnorm和IQnorm由IQ/J独立复算，2048条cos由dot/norm/defined独立复算，两view全部汇总及alpha一致。4聚焦测试（finite-diff/perpacket/梯度隔离/边界与固定规则）和唯一P0/P1通过，真实2Ls原F smoke核实batched-vs-single J和有限6类输出。执行ff6926facd886afa7843ca67e1942153994d86e7，一次archive本远SHA/compile、PID1280413/CWD/argv/CUDA空、日志55→1253字节及自然结束已核实。未新增GPU工作或修改其他owner。相关[Jacobian知识迁移文献](https://proceedings.mlr.press/v80/srinivas18a.html)仅提供研究背景，不直接验证本IQ损失。目标ACTIVE。
