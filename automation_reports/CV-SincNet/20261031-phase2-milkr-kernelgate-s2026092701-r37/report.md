# MILKR 当前 support RF 先验证据门控

R38 在R34等权C4/C8核、R31成员ridge先验和R26融合上检验新假设：仅由默认head概率校准旧类先验，可能忽略RF空间中的新类证据。新RF门控只会减弱父先验，不增大先验；不改双CCA、核权重、神经F/G/head或buffers，不采用R35类别校准、R36密度或R37正交余量。

当前训练SUP二组权重old/new各总.5，目标t为合法SUP旧类成员指示。v=w·diagK−wᵀKw，lambda=.1v。求解带未惩罚截距的加权核岭回归：(K+lambda diag(1/w))beta+b1=t，1ᵀbeta=0。r(x)=clip(k(x,S)beta+b,0,1)，factor=min(2r,1)，新版prior=R34prior*factor。r=.5为无RF反证，factor=1，保留父先验；r<.5只抑制父先验。没有可调混合强度。B只有旧SUP时factor恒1、不存beta；零方差明确constant_unseparated r=.5。严重负方差/非法数值报错，小负舍入容忍1e−12后置零，不静默隐瞒求解失败。C状态为beta、截距与lambda，共8*(physicalSUP+2)bytes，B16bytes。

每物理OOF独立拟合两带宽、原head gate、新RF gate和残差KRR，仅trainSUP。held/query只计算到固定trainSUP的crosskernel，不读取query角色/真值/配额/批统计或拟合状态；全部已注册类竞争。原prior每行和0与幅度界仍保持，RF最近距离与融合条件保持。support严格纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整保留。

源L6300、ep91/92共12600配对，新拟合双CCA64、557440FP64参数。五源RX/两TX留出同时保护两编码器，held-new head列先裁除再softmax；F已见源六TX，源3景/new2与目标6景/new20及已曝光开发集边界保持。源120K1规则由Q0/QF宏H选择，ref6类数缩放，K>1当前SUP物理OOF。源只作合法/数值门槛，固定新机制直接角点证伪，不回改旧run。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。两S每角点raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结法补12KN，再六RX；最终七RX等权all16逐格目标不变。G-only同输入配对与融合跨输入差分开。

sourceGPU4/targetGPU0事前检查余量，其他owner保持。实际commit、命令、环境CWD、数据句柄、来源、矩阵与技术停止条件见experiment.json。真实F/G无query smoke为launcher首步PASS继续，唯一P0/P1审查、单包传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。

## 发布前核验

31项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项+worker alias共5项PASS：加权RKHS方程及截距sum(alpha)=0、方差尺度一致；B和常量核factor=1，其他factor有界0..1且prior零和保持；源old4/目标old6一致；各物理OOF两tau/两gate/KRR只trainSUP，held/query冻结crosskernel；源120规则保留RFmarker，源/worker真实双CCA/fitter，无density/center串入。真实F/G无query smoke为launcher首步PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `17eeb89296da5751f0f7e0420c6a84aaa22c7eca`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'nearest', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5, 'rf_prior_gate_rule': 'support_balanced_kernel_evidence_v1'}；源流程77.974秒。

target worker [142.1365523613058]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，RFgate额外状态按实际SUP计数核实，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 55.5016 | +9.3827 | +2.8977 | -0.6536 |
| S0Q1 | 54.8480 | +12.4553 | +2.8977 | -0.6536 |
| S0Q2 | 58.3993 | 不适用：融合新路径 | +2.8977 | -0.6536 |
| S1Q0 | 48.9262 | +30.9053 | +13.4694 | +16.3920 |
| S1Q1 | 65.3182 | +7.0142 | +13.4694 | +16.3920 |
| S1Q2 | 62.3956 | 不适用：融合新路径 | +13.4694 | +16.3920 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.493 (+19.373) | +3.263 | +2.398 | 45.930 (+9.338) | +11.144 | +15.564 |
| 1 | 20 | 30.813 (+4.838) | +1.403 | -0.314 | 28.435 (+12.529) | +7.327 | +10.249 |
| 20 | 2 | 71.410 (+5.849) | +2.974 | -3.412 | 60.633 (+56.641) | +18.521 | +20.872 |
| 20 | 20 | 69.291 (+7.471) | +3.950 | -1.287 | 60.707 (+45.113) | +16.886 | +18.883 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.837843949657822, 'fusion_gain_pp': 1.4034094058156699}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
