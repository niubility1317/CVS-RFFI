# MILKR 源CCA典型相关强度加权

R40独立检验源表示中的方向可靠性，沿用固定R34双C4/C8核、R31当前SUP成员先验及R26融合，不混入R38成员核gate或R39旧类方向校准。原CCA白化后的64个方向等权；新源编码器固定对第i个坐标乘sqrt(rho_i)，rho_i为源raw/G配对CCA的典型相关系数。相应线性几何对方向按rho_i加权，rank64、投影维度和参数量不变，没有新增混合强度或目标参数搜索。

每block输入仍为原单位化2d均值/标准差。源Wx/Wy保持原.01trace/d正则inverseRoot，SVD WxSxyWy=U diag(rho)Vᵀ，Px=WxU64diag(sqrt(rho64))、Py=WyV64diag(sqrt(rho64))，再按原规则逐record单位化、pad回d。rho理论0..1，数值舍入clip到此区间，超过1+1e−8或非有限值明确报错，无旧权重回退。FFT/f_emb的真实零std半块仍由原CCA处理，不新增余量。权重完全由合法源配对拟合；target不访问源样本/cov/classstatbank，只加载557440个FP64投影参数。新dual checkpoint schema明确为source_C4_C8_correlation_weighted_CCA_v1，不将普通CCA静默认作新版。

源L6300、ep91/92共12600配对，五源RX/两TX留出同时保护两个encoder及rho权重；held-new head列先裁除再softmax。F已见源六TX，源3景/new2与目标6景/new20及曝光开发集边界保留。源120K1规则由Q0/QF宏H选择，ref6类数缩放；K>1当前SUP物理OOF。源仅合法/数值门槛，固定新机制直接验证，不用target选rank/power或回改旧run。

Tri-SIG-TIME SSL、DURA v1.0、原C4 head及buffers冻结，每路径8次F/G原记录一次。R31 gate输入原p、prior.5/R26融合保持。每物理OOF带宽/gate/KRR仅trainSUP；源encoder不由target拟合。support严格纯S0raw/S1G，query三路逐条独立，无真值/角色/配额/批统计。G-only同输入对照与融合跨输入差分开。

首RX1-1/P01/MIX01/seed2026092701全部K1/5/10/20×N2/5/10/20，B24+C96共120预测、40fit/40fusion，全部固定后独立truth-last。两S每KN raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才同冻结方法扩其余六RX；最终七RX等权all16目标不变，单RX不代表完成。

sourceGPU4/targetGPU0启动前核实容量并保护其他owner。真实F/G无query smoke为launcher首步PASS继续；唯一P0/P1审查、一次归档传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。实际commit、命令、环境CWD、来源、数据句柄、矩阵及技术停止条件见experiment.json。

## 发布前核验

32项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增3项+worker alias共4项PASS：Px/Py列乘sqrt(rho)与公式一致；真实FFT/f_emb零std结构可fit/reload，557440参数；新weighteddual schema与普通旧schema隔离，普通默认保持；sourcefold同时保护C4/C8及rho，target加载新桥；原head裁剪/映射及R31gate保持；rho摘要读取finalfit而非load后对象；无RFgate/RFclass/density/center串入；16KN/120row沿通用worker计数。真实F/G无query smoke为launcher首步PASS继续。
