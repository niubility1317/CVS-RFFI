# PAMF完整贡献与参考条件残差

## 预登记

Run `20261030-phase3-star-reference-open-s2026092701-r01`，执行代码 `13d063b98699b1aab9e8e05f3583a65299104515`。同结构/同全部trainable数量的C0和C1，主 `C1::relation_max` 固定；完整机制和边界见`docs/PAMF_PHASE3_REFERENCE_OPEN_DESIGN_20261030.md`，逐row矩阵唯一维护experiment.json。本报告不宣称已启动或取得性能。

两行同次捕获原时频base_norm160、带原gain PA160、带原gain reference160、已限幅SIG160，concat640。两行都保留原reference贡献；仅新增conditioner路径的输入不同：C0全零29、C1实际本条masked reference29。conditioner29→32/GELU→640最后层全0，两行所有参数可训练，C0仍可学常量残差。δ_j=.1norm(c_j)/sqrt160*tanh(h_j)，每分支≤10%原贡献，非RX不变或2pp准确率保证。NoExplicitCFO27保持0、内部局部补偿保留，没有新canonical/波形EQ。

开放projector640→64→64/LN/unit、原H64→160+zero g160有界身份残差、共享relation128→32→1。原default6head始终联合训练、默认头argmax决定旧身份。实构设计new82113/active315106/native982835/unused667729；activeF+G582057，实际eval还保留完整native及原F0参考共1482779参数，未剥离域分支或声称存储压缩。

各从原Tri-SIG-TIME SSL E50/11100独立strict-native开始，不从旧M或更高控制继续。两行2220fresh CENTER_UNITS AdamW/lr1e-4/原wd/clip5，累计13320（非IID自然epoch），same init2026092701、sampler2026103017/slots2026103018。同原M五类注册集合任务，sourceL支持96/query96跨RX且跨day且ID互斥、OE64。原完整L128六前向768行（CRIC.1/ordinary+raw KL/AsCE KL0/LS.01），As64重放cachedP1不新增频道，另rawL128 KD T2权重1；成员.1乘pair/set均衡BCE等权。共享梯度冲突投影/normcap，conditioner也受保护、relation独立；无新SupCon/uniformOE/VICReg/Gram/U/PL/EMA。

源L6300/OE6134/V27000/NV3150、源5RX/3day、原G10MIX01P1及F0/DURA祖先不变。旧6 y仅L/V合法、OE仅outside-six/RXday而TX/K隐藏、原Us56700不新读。全部postPhase1 IQ仍星地P1，ordinary只是原变换，sourceL/U祖先仍原TriSSL；默认phase1/phase2旧产物不改。网络结构面板的CE-only规范不由本同结构学习机制比较取代。

训练冻结后L六原型、V known-only42组六predictedclassτ，再固定sourceNV和RX24-1 futureQ864全部预测，独立CPU才开两truth。source RU90不是取消target的gate。targetU384/formalRX24-18保持0读，Q无拟合/未知y/K/配额，开发已暴露不盲测。主报trueTX macroAK/RU/H，伴CK≥90、RU≥90、AK≥原同P1F0closed−2pp，候选自身closed分开。highscene三未知0/4/2缺类保留N/A，不能认证全部六景。DA0REG0，AN/HOW/UDA/匿名注册未实施，fullgoalACTIVE。

### 执行和验证

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-reference-open-s2026092701-r01-release`，owner `codex-phase3-reference-open-20261030`。
- `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_reference_open_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-reference-open-s2026092701-r01-release/runtime.json`。
- 独占 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-reference-open-s2026092701-r01`。两行并发、每次实际free≥8192MB、每卡最多新增2own，foreign不整卡排除且不停止/修改/重启。每row train→fullsource+target prediction→独立CUDAemptyCPU score自动接续。
- 每行nativeL抽取284160/support和query各213120/OE142080次；训练F2557440行，两行5114880。eval各新F37314/originalF0864/G864真实行，synthetic各2；train烟测4/arm。总F5191236真实+16合成，G1728真实+4合成；43448登记physical pool容量与actual sampled unique未记录分开，不预报B1/峰值或加速。
- 模型7tests7sub PASS3.12s/两受影响定向2.74s、evalscore14tests29sub PASS4.23s、roottrain4 PASS2.85s、launcher3 PASS2.86s；不是一次合并suite。10AST/actualruntime PASS，唯一直接P0/P1 PASS。额外设计审批/builder-review/固定规划链NONBLOCKING/REJECTED_EXTRA_GATE，按用户自主授权与项目最小流程继续。
- 一次N607preflight复用数据句柄、源合同、原F0/DURA文件和显存，source/target/truth值0读。单codecommit/单archive比较+compile、真实checkpoint首步noquery烟测、独立PID/CWD/argv/GPU/log增长、全部预测后truthlast。C0 firstW grad必须0、两行CFOcol27 grad0；C1日志保留actualconditioner输入梯度，并记录真实更新而不凭weightdecay推断参考使用。
- 预期train resolved/final/summary/steps/smoke/complete；predict prototypes/thresholds/known/NV/Q/cal/complete；score results/complete；pipeline/row/log/observations。数据/weights/原日志私有不Git。技术错误保留对应行、健康行继续，不科学调参/续训/重排/重采样或自动重试；UNKNOWN核原PID/root，foreign0操作。


## 实际启动与初步运行核实

VERIFIED / RUNNING，执行code13d063b98699b1aab9e8e05f3583a65299104515。434files/data0，archive539aa5ca0cb1d89dffb43092000b99961413cb5d3460ecb9d18e4ab213fa14c6远端单次比较/compile通过。driver3681335及C0 train3681467/CUDA6、C1 train3681473/CUDA7的livePID/CWD/完整argv/实际GPU已独立核实；实际checkpoint梯度smoke PASS、首步与日志增长齐。每卡一新增own进程，显存离散样本各1722MiB（非峰值），foreign操作0。

主C1和控制C0保持同architecture/alltrainable315106active/982835container、相同L/OE采样与M五类任务。C0零条件仍可学习constant残差；C1只多使用本条maskedr29的条件输入。现无正式识别成绩，完成后自动full prediction和独立双truth评分，不依据source/target score调参/停其它row。目标U/正式RX未访问，AN/HOW/UDA/REG尚未实施，完整v3目标ACTIVE。


## 两组全部完成后的实际结论

VERIFIED C0/C1 2x2220 source updates, all source/crossRX-G10P1 predictions and independent truth-last scores; primary C1 AK=61.4583%, CK=84.2014%, RU=51.0417%, H=55.7677%, F0closed-loss=11.4583 个百分点; NO_PROMOTION; DA/REG not performed, fullgoal ACTIVE.

详见 `docs/PAMF_PHASE3_REFERENCE_OPEN_RESULTS_20261030.md`。
