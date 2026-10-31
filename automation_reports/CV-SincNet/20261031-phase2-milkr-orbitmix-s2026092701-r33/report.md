# MILKR C4与C8等权局部核组合

R34 固定C4与C8两种RF分辨率，不按目标分数选择混合权重。每条记录仍只做八次F：全八次生成C8矩，偶数四次生成原C4矩，head只用原四次。G只在原received处理一次后再旋转；两种RF表征不增加物理观测、标签或额外F/G调用。Tri-SIG-TIME SSL、DURA v1.0和默认神经头保持冻结。

备选是继续加密相位或做support中心化，已有独立负结果保留。本轮只检验两种分辨率互补，等权不保证收益。每个RF分辨率在合法源L上重新拟合一个普通CCA64，不复用target-adapted状态。两编码器共557440个FP64参数，约4.25MiB，另有当前SUP第二路RF特征存储；公开报告实际时间和额外RAM。新checkpoint双CCA schema只含source投影/均值偏置，不含样本、协方差、标签/ID或类原型库。

当前SUP中分别计算tau4与tau8，再分别构造原linear/RBF/multiscale核，最终K=.5K4+.5K8。是两个核的算术平均，不是把相似度先平均后过RBF。只解一套Y−prior的KRR及无惩罚截距，R31多维平衡ridge gate(.1)、priorcap.5保持。raw/G融合仍用R26条件规则，RF距离定义为先对同一物理SUP的两路距离等权，再取最近SUP；不能分别挑两路不同原型形成虚假接近。没有按query真实old/new角色分支或类别配额限制。

K1源五RX/TX-held、120核/正则规则按Q0/QF宏H选择/ref6classscale；sourceK20及target K>1真实物理三折OOF，两个带宽、gate、核系数均只trainSUP。最终当前全SUP重拟合，query逐条冻结应用，无query统计/拟合/选择，不跨K/N/S借SUP。默认head概率只来自C4，源held新类head列在softmax前移除；两编码器都排除held RX/两TX。F已见源六TX、source3景/new2与target6景/new20、已曝光开发基准限制如实保留。源L6300、ep91/92共12600配对包，只源合法/数值门槛，无源H收益否决。

support严格S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整。无R32中心或R30学习度量。首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。两S每角点raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才同冻结方法补12KN；all16后补六RX。最终七RX等权逐KN验收不变，G-only匹配控制与QF跨输入差分开。

sourceGPU4/targetGPU0实际容量核实，其他owner保持。commit/命令/环境CWD/来源/data句柄/输出/技术停止条件见experiment.json；真实F/G无query smoke为launcher首步，包含C4/C8组件、原四相位head和单包检查，PASS继续。唯一P0/P1审查、单包传输/remotecompile、实际PID/CWD/argv/GPU/log增长及独立truth评分保持。

## 发布前核验

18项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：两路各SUP带宽后等权，单残差KRR；RF距离先同一SUP合并再min；source/worker双提取器/双CCA schema/新fitter，物理OOF两尺度和gate均只trainSUP。实际F/G无query smoke在launcher首步，包含两个RF组件/原head/单包检查，PASS直接继续。
