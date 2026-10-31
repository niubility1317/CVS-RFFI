# MILKR support置信度校准先验与query融合

R27 固定完整Tri-SIG-TIME SSL、DURA和原神经分类头冻结，C4moments/sourceCCA64、先验最大幅度0.5及R26已验证的query融合规则均保持。新假设是固定先验容易偏向旧类，应只从当前support校准其可信度；不修改R26已完成产物或按query成绩调其参数。

置信度c(p)=1−H(p)/log(oldcount)，逐record计算。当前SUP中旧类置信度均值a、新类均值b：B无新类时g=1；C只有a−b超过float64机器epsilon才g=clip((c−b)/(a−b),0,1)，否则g=0，并记录no_positive_gap模式、a/b及实际组数。新prior旧类槽为0.5·g(p)·(p−1/oldcount)，新类槽为0，当前SUP仍拟合Y−prior的KRR残差和无惩罚截距。只增加当前SUP的两个标量统计，不增加神经头、物理K、标签或其他K/N/S信息。query只逐条应用冻结a/b；没有query真实old/new角色、类别数、配额、批统计或参数更新。

K>1的真实三物理折OOF中，gate均值、RF带宽和系数均只由训练SUP拟合，held使用训练gate；选120核/正则规则后当前全SUP重拟合。K1仍由源五RX/TX-held、old4/new2的120规则按Q0/QF四主路径H、较大ridge/固定序选择，ref6 lambda随当前注册类数/6缩放。源old4head先去掉heldnew两列再softmax，骨干仍见过源六TX的限制保留。sourceL6300、既有ep91/92共12600配对，sourceCCA不读headpayload，最终只普通CCA参数进入target；不带source样本/特征库/类条件统计，NN/F/G/head更新0。源三景与目标六景、old4new2与old6new20差异不称新盲测。

用户已允许同记录query raw/G融合并保留G-only，support仍pureS0/S1。Q0=raw、QG(机器Q1)=仅G、QF(机器Q2)=融合。R26规则原样保持：只有G到SUP的最近RF距离更小且全类top2分数间隔更大才两路半权，否则raw；规则不读query标签。G对原received只执行一次再C4，QF使用两路既有四次F前向，未重抽信道或增加观测。原始分数/距离cache只按同case同物理query对齐融合，不改变任何SUP状态，全部三路预测后清空。

source合法/数值/真实checkpoint noquery smoke后，首RX1-1/P01/MIX01/seed2026092701同四角点K1/20×N2/20，B12+C24=36行全部固定后独立truth-last。每corner两S Q0较匹配原Q0≥5个百分点且QF>Q0、raw旧新均值guard≥原版−1点才补剩余12KN，最终all16后才其他六RX。QG对原QG同输入配对完整保留，QF对原QG标明路径改变，不用融合代替raw+5要求。

sourceGPU4/targetGPU0启动前核实容量，不动其他owner。完整commit、命令、环境/CWD、来源/data句柄、路径、停止条件见experiment.json；最小独立P0/P1审查、实际checkpointsmoke、一次archiveSHA/compile及PID/GPU/log增长核实保持。技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；support熵置信度/旧新均值/B全开/非正差关闭、校准prior残差方程与截距、query逐条独立/均值不变、物理折gate只读训练SUP、旧三路worker/head crop/非法输入等19项相关检查通过。真实F/G/head checkpoint无query smoke为launcher第一步，PASS后继续源校准及36行先导。
