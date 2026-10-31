# MILKR 当前support类别校准先验

R35 保留R34的C4/C8等权核、源双CCA64、8次F/原4相位head/G原记录一次，Tri-SIG-TIME SSL和DURA及神经head权重不变。新假设是只校准old/new成员可信度，仍可能保留跨RX的旧类混淆，因此在MILKR先验内增加当前SUP的闭式类别校准。它不改神经head输出，但会改变MILKR先验和最终判决；不能称整个分类器不变。

q与R31相同，包含sqrt(m)(p−1/m)、熵置信度和top2间隔。SUP旧/新组分别总权重.5（B只有旧类时总1），联合目标为[old成员指示，类概率目标]。旧SUP类目标为其旧类one-hot，新SUP类目标为旧m列上的均匀向量。以当前SUP加权mean/cov、lambda=.1trace(cov)/d进行一次多输出岭回归，截距不惩罚。成员分量在两组齐全时与R31二组ridge相同，B全开；类分量逐record做欧氏单纯形投影得到pcal。prior旧槽=.5·g(p)·(pcal−1/m)，新槽0，故仍有固定幅度界和每行prior和0。零方差明确constant_unseparated/old_only_constant模式，非法数值报错，不静默回退。

这是当前SUP的闭式先验状态，不新增神经身份头，不更新F/G/buffers，不写回源checkpoint。每物理OOF只用trainSUP拟合mean/cov/classmatrix/tau/KRR，held/query只冻结应用；不读取query真实old/new身份/类别数/配额或批统计，不跨K/N/S借数据。target m6时q维8、输出7，mean8+coef56+targetmean7共71个FP64/568bytes；公开审计只计数/维度/范数/mode，不提交适配系数。

RF仍两种源CCA64新拟合，合557440参数，source L6300、ep91/92共12600配对；五RX/TX-held两编码器均排除held RX/两TX。源old4/new2先移除held新类head列再softmax，类校准只对源旧4列，新增2类目标均匀。F已见源六TX、source3景/new2与target6景/new20、已曝光开发基准限制保留。source120规则选K1 Q0/QF宏H/ref6scale，K20物理OOF，源合法/数值后直接角点证伪，无源H收益否决；不改R34已完成规则或输出。

support纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整。RF两带宽后各半权、单残差KRR与R26融合条件保持；RF距离先同一SUP两分辨率平均再最近，head仍原C4。首RX1-1/P01/MIX01/seed2026092701 K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结方法补12KN；all16后补六RX。最终七RX等权逐KN要求不变，G-only同输入控制与QF跨输入差分开。

sourceGPU4/targetGPU0实际容量核实，不动其他owner。commit/命令/环境CWD/来源/data句柄/输出/技术停止条件见experiment.json；真实F/G无query smoke在launcher首步，检查原head/双RF/单包独立，PASS继续。唯一P0/P1审查、单包传输/remotecompile、实际PID/CWD/argv/GPU/log增长和独立truth评分保持。

## 发布前核验

24项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS，独立执行新增4项测试通过：旧类onehot/new uniform列与当前旧类概率顺序一致；prior只写旧槽、逐record simplex后零和且幅度受限；OOF gate/mean/cov/tau/KRR仅trainSUP；source/worker class rule及双CCA/fitter路由正确。神经head不更新，MILKR prior/最终决策有意改变。实际F/G无query smoke在launcher首步，PASS继续。
