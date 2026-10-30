# PAMF-OW 动态注册集合训练

## 预登记与一次技术恢复

Run `20261030-phase3-star-open-episodic-s2026092701-r02`，执行代码 `1029bc4f4ab86022c75ae0f3332b5dfafc1fe909`。唯一主候选 `M::relation_max`，固定 S0/S1 控制；三行矩阵维护在同目录 experiment.json，具体机制见 `docs/PAMF_PHASE3_STAR_OPEN_EPISODIC_PLAN_20261030.md`。本记录不宣称已启动或取得性能。r01三行真实checkpoint smoke PASS后，在首步原型slot int64 NumPy2/Torch2.1转换技术失败；均0参数更新/0目标或真值读取。r02只将int64 metadata经Python整数列表转换，科学矩阵不改，失败产物保留。兼容边界先RED后2项定向GREEN、一次直接P1 targeted rereview纳入。

三行独立复用原完整 Tri-SIG-TIME SSL E50/11100，通过原生可微构造器更新 Sinc、完整骨干和原默认六类头；新增同次 t2/PA/SIG 开放64、零初始化有界身份残差及共享关系评分。识别器284834（新增51841），训练wrapper952563，未使用原生域参数667729另列。S0关系输入detach，S1固定六类联合训练，M轮换五类集合。未知OE不加入默认六类uniform目标，不使用VICReg/Gram。

每行2220次fresh CENTER_UNITS AdamW，lr1e-4/原wd/clip5，累计13320步；同采样2026103017、槽排列2026103018、初始化2026092701。原L128完整六次前向（CRIC.1、ordinary/raw KL、AsCE且AsKL0、LS.01）、同P1原F0 KD T2权重1。As64重放原cachedP1，无新增频道。额外每类16 support和16 query、OE64；跨RX且跨day的support/query物理记录互斥。成员0.1乘等权pair/set均衡BCE，原型槽随机；共享梯度冲突投影/范数上限，relation独立。顶层train执行原Lhook、encoder及Dropout/BN eval保留梯度和固定统计。

每行训练F2557440行：原六L前向1704960、KD284160、episode568320。三行训练F7672320行；不将重复曝光算作独立物理样本。额外冻结推理、真实时间和G成本据完成产物报告，不推定B1时延。

源L6300/辅助OE6134/V27000/source novel3150全部为原G10 MIX01星地P1；源RX1/3/4/6/8、三日，源真旧类标签仅L/V合法。OE只outside-six与RX/day，不读真实TX/K。原Us56700保持隐藏且本增量不读。原F0已见全部源六类/RX，因此episode留类/留RX仅训练代理。source novel和目标池均历史开发暴露，不宣称新盲测。

冻结模型后L形成六个均值单位原型（仅1536B FP32），V仅按既定42组known风险校准六个predicted-class阈值（48B FP64），随后冻结source novel3150与目标RX24-1 futureQ864的全部预测，独立CPUscorer再读两份真值。Q来自2021-03-15，同原G10 received经同DURA；源RX与目标RX互斥。目标U384保持不读，正式RX24-18未消费；无Q拟合、无目标y/K/角色/配额。sourceRU90不取消预登记目标测试，控制高分不能替换M。

主报逐真实TX宏 A_K、R_U、H，并列 C_K、原同P1 F0 closed差、候选自身closed差、逐TX/RX/day/scene与同ID损伤分解。targets v3为CK90/RU90/原F0closed−AK≤2pp；H不替代单项。Q practical_high未知计数0/4/2，缺必需身份保持N/A，六场景完整认证不足。本轮只DA0_REG0，后续UDA/匿名注册未完成，AN/HOW=N/A，完整目标ACTIVE。

## 命令与证据

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-open-episodic-s2026092701-r02-release`，owner `codex-phase3-star-open-episodic-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_open_world_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-open-episodic-s2026092701-r02-release/runtime.json`。
- 独占root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-open-episodic-s2026092701-r02`，三行train→完整source/target预测→独立CPUscore自动串联；最多两行并发、每卡最多新增两个own训练/预测、每次启动实际free≥8192MB。其他用户任务保持，允许同卡其他用户任务。
- 初步聚焦检查：model7项及受影响FP32定向检查、episode/loss11项及7项受影响检查、root三arm一步联调/权限共4项、eval/scorer10项及结果契约定向检查、launcher6项19sub；各自日志留本地，非一次合并套件。14AST与实际runtime接口PASS。独立一次直接P0/P1review PASS，结果契约修复已纳入；不重复主套件。
- 一次N607 preflight核实原权重/源缓存/目标VALIDATED_ONCE capsule/split句柄及显存，未读IQ或truth值。实际显存见本run preflight.json。单代码commit/单归档比较与compile；首步真实checkpoint合成G10 noquery梯度smoke，评估首步原F0/新F/G合成smoke；启动后独立PID/CWD/argv/GPU/log增长；预测齐后独立truth-last。
- 预期train final/resolved/summary/steps/smoke/complete，predict prototypes/thresholds/known/novel/target/complete，score results/complete，row/pipeline state和logs。IQ、checkpoint、预测和原始日志不进Git。
- 技术失败保留对应row、健康row继续，无自动重试；系统失败需要停止时由owner先核实自己的PID/CWD/argv，绝不触碰foreign。UNKNOWN先读回同root/PID；科学未达不重选、重采样、延长或换主。


## 实际启动核验与进行状态

VERIFIED / RUNNING。执行代码 `1029bc4f4ab86022c75ae0f3332b5dfafc1fe909`；归档416文件/新增data0，SHA `eeeeda0b2b84211802ae38b706ea0c5f15d01a0b2f1ab2b93beacaa58c872cc8`，远端一次比较/compile和runtime校验PASS。driver3487225、S0 train3487260/CUDA7、S1 train3487263/CUDA1的PID/CWD/完整argv与实际GPU已独立读回；源步日志增长、真实梯度smoke和参数更新已核实。实际离散GPU显存样本约1.7GiB，非峰值。当前快照 `{"S0": {"status": "RUNNING", "latest_step": 550, "smoke": true}, "S1": {"status": "RUNNING", "latest_step": 525, "smoke": true}, "M": {"status": "QUEUED", "latest_step": 0, "smoke": false}}`；M按预登记队列，不改为控制主报。

本次兼容恢复未改变科学模型、数据、预算、RNG或阈值；原r01三臂0update失败产物保留，所有own进程自然退出。当前没有正式识别成绩，源训练loss不能替代开集性能。每row固定后自动进行L原型/V校准/NV与Q完整预测/独立CPU评分。目标v3 CK90/RU90/F0loss2pp和完整UDA/发现/匿名注册仍ACTIVE，其他用户干预0。
