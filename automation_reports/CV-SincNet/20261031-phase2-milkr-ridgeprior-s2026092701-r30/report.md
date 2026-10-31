# MILKR 当前support多维平衡先验校准

R31 固定Tri-SIG-TIME SSL、DURA v1.0、原默认神经头、C4moments/sourceCCA64、先验幅度上限.5和R26融合规则。新假设是仅用熵置信度的old/new均值会丢失概率形状，也会在均值差很小时形成陡峭gate。因此只在当前SUP上拟合多维平衡岭回归可信度，不改R27已完成参数，不使用R30的512权重。

q(p)=[sqrt(m)(p−1/m), c(p), margin(p)]，m是合法已知头列数，c为1−H(p)/logm，margin为最大两概率差。SUP已知/新增角色来自已授权标签，t=1/0；旧组、新组分别总权重.5。加权mean q，centered cov和rhs，ridge=.1·trace(cov)/d，beta=(cov+ridge I)^−1rhs，g(p)=clip(.5+(q−mean)beta,0,1)。这是带无惩罚截距的平衡二组岭回归软权重，不宣称概率校准。B无新类g=1；C covtrace≤float64eps时明确constant_unseparated/g=.5并审计。非法输入和数值求解失败报错。当前SUP旧类prior=.5·g(p)·(p−1/m)，新增槽0；KRR拟合Y−prior保持。只增加至多8维mean/beta，不增加观测、标签或神经身份头。

K>1真实三物理折OOF，每折gate均值/协方差/系数、RF带宽和KRR都只用trainSUP，held逐条应用冻结参数；最终当前全SUP重拟合。query不提供真实old/new角色，不参与统计/拟合/选择，也不跨K/N/S借SUP。每条query只用冻结映射概率。公开fit审计仅计数、mode、ridge、系数范数和statebytes，不提交适配权重。

source L6300、ep91/92共12600配对，CCA64五RX/TX-held只选120个K1核/正则规则（Q0/QF宏H、ref6 classscale），K20保持物理OOF；held新类head两列在softmax前剔除。F仍见源6TX、源三景/目标六景、源new2/目标new20的限制保留。本轮是新固定SUP机制，沿R26/R27的直接四角点证伪政策，源阶段只作合法/数值检查及K1选择，无源H收益否决门槛；不放宽R30门槛或复用其被拒权重，不因源小代理替代目标验证。

support严格S0raw/S1G；Q0raw/QG仅DURA/QF同记录融合完整保留。原received经G一次再C4，融合仍只在G到SUP最近RF距离更小且全类margin更大时半权，否则raw。原F/G/head不更新。首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw、raw旧新均值guard≥原版−1pp才冻结方法补12KN；全部16格后再补另6RX。最终七RX等权逐KN目标不变，QG匹配控制和QF跨路径差明确区分。

sourceGPU4/targetGPU0启动前核实容量，其他owner不动。commit/命令/环境CWD/来源/data句柄/输出/技术停止规则见experiment.json；真实F/G noquery smoke为launcher首步PASS继续，唯一独立P0/P1审查、单包传输/remotecompile及实际PID/CWD/argv/GPU/log增长核实保持。

## 发布前核验

21项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：加权中心化ridge与0.5截距一致，B全开/零方差0.5均有明确模式；OOF gate/cov/核/系数仅trainSUP，query冻结应用；普通CCA路由，不加载R30 gains或进入其源收益门槛。实际F/G noquery smoke将在launcher首步运行，PASS直接继续。
