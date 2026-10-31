# PAMF 公共类传输对齐

VERIFIED / ANALYZED_DEVELOPMENT：新444训练、八列三DURA全部预测和独立评分完成；科学NO_PROMOTION，完整目标ACTIVE。

Run `20261031-phase3-common-transport-s2026092701-r01` / actualcode `4038105cb7f7c2defe75eb7fbfa825e7f932b40f`。只新增TARGET_CCD444；源r03初始化，固定Sinc/buffers/F0/defaulthead/G，保留sourceKD10及PCD.1，step101加入固定源六原型adaptive-UOT CCD.1。旧SOURCE/PCD训练a336/r01、六列预测及统计4273/r02精确复用。新known90/OE32源fit1，V两组六τ12新值/旧36exact全冻，再八列三DURA全部预测后CPU三truth；Utruth0。完整公式/数据/祖先/三对照晋级/成本/停止规则见`docs/PAMF_PHASE3_COMMON_TRANSPORT_PLAN_20261031.md`，六类seed/row在experiment.json。

Interpreter `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-common-transport-s2026092701-r01-release`；owner `codex-phase3-common-transport-20261031`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_common_transport_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-common-transport-s2026092701-r01-release/runtime.json`。新root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-common-transport-s2026092701-r01`。科学低分仍完成预登记全测试，无formalRX/发现/匿名REG，A_N/H_OW=N/A，原主基线保持。

## 完整结果与科学判断

source_DURA: 89.755556/96.785185/65.904762/76.002910%; F0loss1.225926pp；clean_DURA: 69.965278/86.631944/99.652778/82.210992%; F0loss10.069444pp；star_DURA: 64.583333/86.458333/77.430556/70.425835%; F0loss8.333333pp。主对匹配PCD：source正确+4/未知拒对+12、clean正确−18/未知+1、star正确−4/未知−2；star AK/CK/RU各−.694444pp，H−.700073pp。不因对较弱E0的star正确+3/未知+1或SOURCE的+1/+6而晋级；sourceknown12过、target保护/必需覆盖和胜PCD规则失败。原4af基线/PCD对照保持，AN/HOW NA、正式RX未读。

全444 source/U索引exact匹配，CCD101..444启用344步，每次400logUOT，完整方程/选择/伪hist/beta/填充/残差/首100数值差及24点见`docs/PAMF_PHASE3_COMMON_TRANSPORT_RESULTS_20261031.md`。源几何额外只读诊断L positive99.460317%/OE8.493642%，无U/Q/新IQ/NN权重/forward/truth，源可分不证明目标伪标签纯度。

新NN444而旧control888仅继承，新增real signal398122/G2112/F01728，新GMM1/新τ12+旧36共48值384B；原F0/defaulthead/G/Sinc/buffers保持。过程秒数{"train": 111.65449, "predict": 31.903433, "score": 24.684922, "pipeline": 168.438859}，非B1。driver1480122/train1480142独立live/fullargv/CWD/CUDA0/UUID56adac86，GPU406→940MiB为离散非peak，日志517→852→1058B；predict1483523/GPU1与CPU1484878仅pipeline/产物，无独立live，不继承上轮证据；终态owned空/foreign0。

发布前P1 completion字段由实际train→disk→launcher RED→GREEN定点修复并复审PASS；初0105归档未发布执行、保留，实际403。private Win206在CreateProcess未spawn，改stdin简化probe PASS后只补输入preflight；729files/data0单实际archiveSHA/compile/runtime VERIFIED，无真实run技术失败。未来注册4TX只有metadata容量/早U须整类排除，packet历史排除UNKNOWN，未建池不作为当前输入或盲测。
