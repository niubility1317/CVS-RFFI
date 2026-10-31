# MILKR 当前 support RF 先验证据门控

R38 在R34等权C4/C8核、R31成员ridge先验和R26融合上检验新假设：仅由默认head概率校准旧类先验，可能忽略RF空间中的新类证据。新RF门控只会减弱父先验，不增大先验；不改双CCA、核权重、神经F/G/head或buffers，不采用R35类别校准、R36密度或R37正交余量。

当前训练SUP二组权重old/new各总.5，目标t为合法SUP旧类成员指示。v=w·diagK−wᵀKw，lambda=.1v。求解带未惩罚截距的加权核岭回归：(K+lambda diag(1/w))beta+b1=t，1ᵀbeta=0。r(x)=clip(k(x,S)beta+b,0,1)，factor=min(2r,1)，新版prior=R34prior*factor。r=.5为无RF反证，factor=1，保留父先验；r<.5只抑制父先验。没有可调混合强度。B只有旧SUP时factor恒1、不存beta；零方差明确constant_unseparated r=.5。严重负方差/非法数值报错，小负舍入容忍1e−12后置零，不静默隐瞒求解失败。C状态为beta、截距与lambda，共8*(physicalSUP+2)bytes，B16bytes。

每物理OOF独立拟合两带宽、原head gate、新RF gate和残差KRR，仅trainSUP。held/query只计算到固定trainSUP的crosskernel，不读取query角色/真值/配额/批统计或拟合状态；全部已注册类竞争。原prior每行和0与幅度界仍保持，RF最近距离与融合条件保持。support严格纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整保留。

源L6300、ep91/92共12600配对，新拟合双CCA64、557440FP64参数。五源RX/两TX留出同时保护两编码器，held-new head列先裁除再softmax；F已见源六TX，源3景/new2与目标6景/new20及已曝光开发集边界保持。源120K1规则由Q0/QF宏H选择，ref6类数缩放，K>1当前SUP物理OOF。源只作合法/数值门槛，固定新机制直接角点证伪，不回改旧run。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。两S每角点raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结法补12KN，再六RX；最终七RX等权all16逐格目标不变。G-only同输入配对与融合跨输入差分开。

sourceGPU4/targetGPU0事前检查余量，其他owner保持。实际commit、命令、环境CWD、数据句柄、来源、矩阵与技术停止条件见experiment.json。真实F/G无query smoke为launcher首步PASS继续，唯一P0/P1审查、单包传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。

## 发布前核验

31项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项+worker alias共5项PASS：加权RKHS方程及截距sum(alpha)=0、方差尺度一致；B和常量核factor=1，其他factor有界0..1且prior零和保持；源old4/目标old6一致；各物理OOF两tau/两gate/KRR只trainSUP，held/query冻结crosskernel；源120规则保留RFmarker，源/worker真实双CCA/fitter，无density/center串入。真实F/G无query smoke为launcher首步PASS继续。
