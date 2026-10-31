# MILKR 当前support半中心化几何

R32 固定Tri-SIG-TIME SSL、DURA v1.0、原神经头、C4/sourceCCA64、R31多维平衡岭回归先验和R26融合规则。新假设是目标RX的共同特征偏移仍可能进入RF距离；只由当前SUP估计该偏移，不能用query中心。固定八个RF block分别单位化，减去.5倍当前SUP均值后再次单位化。五个RF分支及三112维SIG视图均处理，默认head概率完全不变。每个当前SUPcase增加1072个均值标量/8576字节，不增加物理K或标签。均值是当前SUP统计，不是源样本库；公开审计只含范数/计数/strength/statebytes，不提交拟合均值。

备选是重训更大的度量映射或继续改变head先验；前者R30源收益不足，后者R31已改善多数角点。本轮检验固定RF几何变化，不扫描center强度。历史R10是CCA前仅SIG视图全中心化/坐标标准化，source未通过；其负结果保留，不将本轮当作已证有效。本轮在新C4/CCA几何上覆盖全部RF，半中心化避免全中心化可能删除类别方向。

K>1真实物理三折OOF中，每折仅trainSUP估计RF中心，再以同中心转换train和held；gate/带宽/KRR也均只trainSUP。选规则后在当前全SUP重拟合。query只逐条应用冻结中心，S0/S1和不同K/N不借SUP、不计算query均值或方差。原RF单位向量减去半均值后理论范数至少.5，数值非法时报错；两侧同映射构造内积核。R31 prior仍只用未改变的head概率，固定balancedridge .1和priorcap.5。

source L6300、ep91/92共12600配对，CCA五RX/TX-held排除held RX/两TX，普通CCA64最终权重；held新类head列在softmax前剔除。source120个K1核/正则按Q0/QF H、ref6classscale选择，K20真实物理OOF。原F见源6TX、源3景/目标6景、源old4/new2与targetold6/new20限制保持。本轮沿R31固定新机制直接角点证伪政策，源只作合法/数值检查和核规则选择，无sourceH收益否决，不重开旧run或回调其参数。

support纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整。G原received一次在C4前，融合公式不变（G最近SUP距离更小且全类margin更大才半权，否则raw）。首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结方法补12KN；all16通过后补另6RX。最终七RX等权逐KN目标不变，QG匹配控制和QF对原QG的跨路径差分开。

sourceGPU4/targetGPU0启动前核实容量，其他owner不动。commit/命令/环境CWD/来源与data句柄/输出/技术停止条件见experiment.json；实际checkpoint无query smoke为launcher首步PASS继续，唯一独立P0/P1审查、单包传输/remotecompile及实际PID/CWD/argv/GPU/log增长核实保持。

## 发布前核验

19项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：State保存中心化SUP，query原特征只映射一次；OOF两侧同train中心，无双重中心化；head概率原样，规则字段传递完整；1072个FP64均值8576bytes。实际F/G noquery smoke在launcher首步运行，PASS直接继续。
