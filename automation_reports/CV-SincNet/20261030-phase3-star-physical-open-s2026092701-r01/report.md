# PAMF同记录星地视图身份结构学习

## 预登记

Run `20261030-phase3-star-physical-open-s2026092701-r01`；实际执行代码 `31fa172946aeaecd2d3f1e784aee7366b7b5bee1`。P_L/P_LOE为同结构/同数据/同预算对照，唯一主P_LOE::relation_max。完整机制和科学边界见`docs/PAMF_PHASE3_PHYSICAL_OPEN_PLAN_20261030.md`，逐row矩阵唯一维护experiment.json。

从原Tri-SIG-TIME SSL E50/11100/defaultsix F0独立初始化，不继续训练旧M/C0/C1或根据目标成绩挑父。两组精确完整四贡献640/常量condition C0网络，新增wrapper只捕获同次projector[-1] raw64，无新增weights/forward。原默认head、.1有界身份残差g0、NoExplicitCFO27保持。活动315106/new82113/native982835/unused667729；activeF+G582057，实际eval含native+原F0ref+G1482779，未剥离或测B1。

各2220fresh CENTER_UNITS AdamW/lr1e-4/原wd/clip5，累计13320；initseed2701、sourceM sampler3017/slot3018。原完整L128六前向768（CRIC.1/ordinary+raw KL/AsCE KL0/LS.01）、P1KD128 T2/w1、episode support96/query96/OE64 P1不变。每step另同nativeL前64/OE同oi64的P0共128；1280Frows/step。P1同source roles，P0received不表示clean，As64同P1重放不新抽channel。source U/PL/EMA在增量关闭。

两组都以unitopen64对Lsupport/query192作跨RX且跨day正对SupCon .1/temp.1，denominator只eligiblepositive或异类；raw64同recordP0P1 L正则.1。P_LOE额外OE正则.1，P_L权重0但同forward/诊断；每正则25MSE+25std+cov，samplevariance/unbiased、eps1e-4，covoffdiag平方和/64。不同OE记录不赋同类/异类身份，未用上轮失败MNN边/真TX/K。fullL+KD保护anchor，其余全部附加loss投影/normcap；relationaux独立。实际OE raw64目标梯度/全loss组件、实际unique源索引数记录；不由weightdecay假推未知学习。

sourceL6300/OE6134来自原5RX/3day的G10MIX01 P0/P1，祖先F0/DURA合同不变。OE只有outside-six/RXday；原Us56700仅原SSL祖先不新增读。V27000只known42风险/六predclassτ校准，NV3150只冻结后评价。所有新IQ都星地，输入一次unitRMS、fc2.45GHz/25MHz/600km/receiver2027/scene3001/channel4001，没有clean新输入、重抽信道或G训练。

每行2220固定后L六原型/V42τ冻结，再sourceNV和RX24-1 futureQ864全部预测，独立CPUdualtruth最后连接。eval实际只P1，原F0同P1closed不可换；sourceunknown90不取消Q。目标U384/formalRX24-18未消费，Q不拟合，未知y/K/配额0；已暴露开发不盲测。CK90/RU90/F0loss2pp、主trueTX宏AK/RU/H与六scene覆盖按v3；high郊区缺必需unknown6保持N/A，不重抽凑指标。仅DA0REG0、AN/HOW None；完整适应/发现/未来匿名注册目标ACTIVE。

- Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-physical-open-s2026092701-r01-release`；owner `codex-phase3-physical-open-20261030`。
- `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_physical_open_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-physical-open-s2026092701-r01-release/runtime.json`。
- 独占root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-physical-open-s2026092701-r01`；两row并发，每次actualfree≥8192MB、每卡≤2新增own，foreign不整卡排除、不停止/修改/重启。每row train→fullNV+Q prediction→CUDAempty CPU score自动接续，其他row健康继续。
- 两组训练真实F5683200；eval新F每组37314/原F0864/G864，总F5759556真实+24合成，G1728真实+4合成。train实际checkpointsmoke8合成F/arm，先source values，PASS直接继续。sourcepool42584+Q864=43448，不把多view/重复曝光当新记录。
- model/loss10tests18sub PASS3.85s；evalscore13tests39sub PASS4.14s后唯一wrapper接口定向1 PASS2.88s；trainer3 PASS22.60s（实际可微单step两臂）；launcher3 PASS4.02s，非一次合并suite。12AST/实际runtime，唯一directP0P1 PASS。一次preflight只元数据/文件存在/GPU，真实IQ/truth0；单archive比较/compile后启动、独立PID/CWD/argv/GPU/log增长和全部预测后truthlast，不增审批/签名/hash链。
- 预期train final/resolved/steps/summary/smoke/complete；predict prototypes/thresholds/known/NV/Q/calibration/complete；score results/complete；launch/pipeline/row/config/logs/observations。所有raw日志/weights/features私有不Git。非法角色/祖先、非有限、输出碰撞/不可运行/无法合法预测保留，无自动重试、按科学结果停或热改；UNKNOWN核原PID/root。当前不宣称已启动或新性能成功。
