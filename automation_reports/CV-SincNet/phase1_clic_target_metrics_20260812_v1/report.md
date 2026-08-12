# Phase1 CLIC target LEO-weak 12臂指标封存v1报告

## 状态与目标

- 实验ID：`phase1_clic_target_metrics_20260812_v1`。
- 当前状态：`ANALYZED / ARTIFACTS_COMPLETE=12/12 / FORMAL_LAUNCH=1 / PERFORMANCE_READ=YES / RETRY=NO`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：对已完成且truth-blind封存的`phase1_clic_target_prediction_20260812_v1`共12份prediction进行一次独立truth-side计量，输出target LEO-weak known、unknown rejection、open-set和scene／RX／class／day DG完整证据。
- 本run不训练、不适配、不更新、不重跑预测、不改变阈值、不选择候选，也不要求与后续ADV3B02共用同一物理包；只消费既有prediction和同一truth sidecar。

## 冻结输入与输出

- 输入prediction：`runs/phase1_clic_target_prediction_20260812_v1/predictions/F{1..6}{C,G}_CLIC12.prediction.json`共12份；上游报告commit=`08326e2a`，每份3120行、forward_count=3120、同一package SHA=`64373f39e45cb7dcfc5b5d0a989e43ba17f3ff16d85577631434e65e6235fff5`，truth未打开且fit/update/retry/selection全0。
- truth sidecar：`runs/phase1_clic_target_prediction_20260812_v1/sealed_target/truth_sidecar.json`；只能由本独立scorer在prediction/predictor/package验签后首次打开。
- 输出：`runs/phase1_clic_target_metrics_20260812_v1/metrics/F{1..6}{C,G}_CLIC12.metrics.json`共12份；日志：`logs/phase1_clic_target_metrics_20260812_v1`；两根启动前必须不存在。
- launcher：`code/scripts/launch_phase1_clic_target_metrics12_v1_20260812.sh`；N607`CVS-RFFI`Python，CPU执行，CUDA禁用，OMP／MKL／OpenBLAS各2线程。
- evaluator commit=`17e46320`，SHA-256=`845658D432891314447EFE171E91EC772181366F9A2B317595CE53EDFACA8052`；launcher SHA-256=`23BB602F5A026CC17E5886E4A6F7B52ED9756C56C0CA57463FDBECB6771F7A63`。launcher专测`2／2`、`bash -n`和dry-run12行（C6/G6）通过，禁ADV／combined score／prediction重跑／package／threshold／class-order／retry参数为0。

## 指标与证据边界

- known：registered-known中的unknown／defer均按错误计；按scene、receiver、class、day及三scene等权、sample-pooled封存准确率、macro、minimum、false reject、defer和accepted-known coverage。
- unknown：显式拒识只计`decision=unknown`，defer单列且不计分子；全局和每scene冻结分母、分子、defer和拒识率，预注册floor为0.70。低于floor是有效失败结果，仍写receipt而不抛异常。
- open-set：封存AUROC-unknown、AUPR-out、FPR95；prediction的`e_unknown`和decision只在truth连接后用于计量，绝不回流。
- 每份receipt的`passed`只表示unknown floor，不表示对ADV3B02非劣或方法综合晋级；固定`baseline_compared=false`、`comparison_status=ADV_COMPARISON_PENDING`，不得写ADV通过结论。
- 12份全部闭合后才读取性能并形成同一candidate/run行表。每行必须同时呈现target LEO-weak、unknown rejection、open-set、known/DG、defer/coverage和技术状态；不得拼接不同臂的单项极值。

## 本地验证与正式停止规则

- target-only sealer API／CLI、正例公式、69 unknown＋31 defer gate-false、prediction／truth／predictor篡改、truth打开顺序、真实G文件调用和不可覆盖测试`7／7`通过；完整postfreeze`152／152`通过；`py_compile`、`git diff --check`通过。
- 原combined ADV scorer保持严格，不接受缺失reference；当前无配置等价ADV原件，所以本run只封target metrics，后续另行生成6fold ADV baseline并完成非劣比较。
- formal launcher唯一调用1次，retry=`NO`。发生协议泄漏、hash/checkout漂移、覆盖风险或至少2fold同一确定性异常时，只停止本run确切进程并保留证据；不得按accuracy、unknown rejection或其他性能值停止。

## 正式执行与封存证据

- 预启动检查：精确release=`/home/szu2070436088/2510044040/releases/phase1_clic_target_metrics_20260812_v1_9dfe67f2`；项目根唯一outer=`/home/szu2070436088/2510044040/CV-SincNet/phase1_clic_target_metrics_20260812_v1_outer.out`；run、log和outer启动前均不存在，相关进程为0。
- 物理release evaluator SHA=`b1a80d20a7c8020d4651151bab2d8463d497dce7c55968af8e88d5f2b921740d`（归档CRLF字节193708）；规范化LF SHA=`845658D432891314447EFE171E91EC772181366F9A2B317595CE53EDFACA8052`，与冻结本地代码一致；launcher SHA=`23bb602f5a026cc17e5886e4a6f7b52ed9756c56c0ca57463fdbecb6771f7a63`。
- 静态检查：远端`py_compile`、`bash -n`、`--help`、`bash launcher --dry-run`12行、12份prediction+truth输入存在性和fresh roots均通过。launcher虽保留归档权限`664`，正式入口明确使用`bash`执行，不改变release。
- 唯一正式命令：`nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_target_metrics_20260812_v1_9dfe67f2/code/scripts/launch_phase1_clic_target_metrics12_v1_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_target_metrics_20260812_v1_outer.out 2>&1 &`；`FORMAL_INVOCATION=1`、launch PID=`2878452`、`RETRY=NO`。
- 完成证据：6个fold worker各生成C/G两份receipt，12/12 metrics文件闭合；6份worker日志各有两条成功路径且无`Traceback`、`ERROR`或`Exception`；相关进程结束后为0，8卡保持空闲，SSH/TCP22清零。outer为空文件仅表示worker输出均定向到fold日志，不表示缺少receipt。
- 结构QA：12/12文件JSON可解析、top-level schema/keyset闭合、fold/arm映射闭合、`sealed=true`且truth sidecar已按scorer打开；known三scene、unknown三scene、open-set、receiver/class/day结构和所有数值finite。每份均`target_fit_rows=0`、`target_update_rows=0`、`target_retry_count=0`、`target_selection_count=0`、`target_selection_feedback=false`，`baseline_compared=false`、`comparison_status=ADV_COMPARISON_PENDING`。

## 同一receipt行的target LEO-weak结果

以下表格的scene顺序固定为`clear/low-elev/rain`；`G/C/L/R`分别表示global、clear、low-elev、rain。`target`列明确每一行都来自目标RX=`20-1`、day0/1/2和三种LEO weak场景。known的unknown/defer按错误计；unknown rejection只计显式`decision=unknown`，defer不计入分子；coverage是accepted-known coverage。DG列给出同一receipt的global receiver/class/day准确率映射，scene列给出三scene准确率。

|候选|机制|target|known overall/macro/min_class/min_rx/min_day|known scene accuracy C/L/R|unknown reject G/C/L/R|open-set AUROC/AUPR/FPR95|known FR G/C/L/R；defer G/C/L/R；coverage G/C/L/R|unknown FAR G/C/L/R；safe handling G/C/L/R|DG receiver；class；day（global）|unknown gate|ADV|
|---|---|---|---|---|---|---|---|---|---|---|---|
|F1C_CLIC12|C raw phase control|20-1；d0/1/2；LEO C/L/R|0.6729/0.6729/0.5750/0.6729/0.6265|0.7125/0.6562/0.6500|0.0387/0.0437/0.0213/0.0512|0.5593/0.8368/0.9375|0.0667/0.0750/0.0437/0.0813；0.0583/0.0375/0.1187/0.0187；0.8750/0.8875/0.8375/0.9000|0.9200/0.9450/0.8900/0.9250；0.0800/0.0550/0.1100/0.0750|RX`20-1`:0.6729；class`20-15`:0.6750,`20-19`:0.8333,`6-15`:0.6083,`8-20`:0.5750；day`2021_03_01`:0.6265,`2021_03_08`:0.6688,`2021_03_15`:0.7261|FAIL（0.70）|PENDING|
|F1G_CLIC12|G complex local invariant curvature|20-1；d0/1/2；LEO C/L/R|0.6521/0.6521/0.4917/0.6521/0.5964|0.7125/0.6500/0.5938|0.0612/0.0563/0.0700/0.0575|0.5352/0.8365/0.9187|0.0688/0.0688/0.0688/0.0688；0.0542/0.0688/0.0312/0.0625；0.8771/0.8625/0.9000/0.8688|0.8862/0.8675/0.9038/0.8875；0.1138/0.1325/0.0963/0.1125|RX`20-1`:0.6521；class`20-15`:0.4917,`20-19`:0.7667,`6-15`:0.7333,`8-20`:0.6167；day`2021_03_01`:0.5964,`2021_03_08`:0.6433,`2021_03_15`:0.7197|FAIL（0.70）|PENDING|
|F2C_CLIC12|C raw phase control|20-1；d0/1/2；LEO C/L/R|0.6104/0.6104/0.1750/0.6104/0.5677|0.6062/0.5813/0.6438|0.0800/0.0825/0.0825/0.0750|0.5796/0.8720/0.9021|0.0292/0.0250/0.0437/0.0187；0.0125/0.0250/0.0000/0.0125；0.9583/0.9500/0.9563/0.9688|0.8996/0.8825/0.9012/0.9150；0.1004/0.1175/0.0988/0.0850|RX`20-1`:0.6104；class`14-10`:0.5750,`20-19`:0.1750,`6-15`:0.8083,`8-20`:0.8833；day`2021_03_01`:0.6727,`2021_03_08`:0.5677,`2021_03_15`:0.5875|FAIL（0.70）|PENDING|
|F2G_CLIC12|G complex local invariant curvature|20-1；d0/1/2；LEO C/L/R|0.6167/0.6167/0.3500/0.6167/0.5355|0.6000/0.6062/0.6438|0.0271/0.0175/0.0238/0.0400|0.5885/0.8687/0.9000|0.0208/0.0187/0.0125/0.0312；0.0083/0.0187/0.0000/0.0063；0.9708/0.9625/0.9875/0.9625|0.9525/0.9600/0.9663/0.9313；0.0475/0.0400/0.0338/0.0688|RX`20-1`:0.6167；class`14-10`:0.5083,`20-19`:0.3500,`6-15`:0.7833,`8-20`:0.8250；day`2021_03_01`:0.6909,`2021_03_08`:0.5355,`2021_03_15`:0.6188|FAIL（0.70）|PENDING|
|F3C_CLIC12|C raw phase control|20-1；d0/1/2；LEO C/L/R|0.7188/0.7188/0.5917/0.7188/0.6203|0.7625/0.7000/0.6937|0.0192/0.0312/0.0063/0.0200|0.5343/0.8439/0.9688|0.0229/0.0312/0.0063/0.0312；0.0208/0.0063/0.0563/0.0000；0.9563/0.9625/0.9375/0.9688|0.9575/0.9575/0.9313/0.9762；0.0425/0.0425/0.0688/0.0238|RX`20-1`:0.7188；class`14-10`:0.5917,`14-7`:0.7250,`6-15`:0.7167,`8-20`:0.8417；day`2021_03_01`:0.7857,`2021_03_08`:0.7468,`2021_03_15`:0.6203|FAIL（0.70）|PENDING|
|F3G_CLIC12|G complex local invariant curvature|20-1；d0/1/2；LEO C/L/R|0.7188/0.7188/0.5833/0.7188/0.5759|0.7625/0.7312/0.6625|0.0258/0.0262/0.0175/0.0338|0.5172/0.8282/0.9208|0.0458/0.0500/0.0250/0.0625；0.0271/0.0187/0.0375/0.0250；0.9271/0.9313/0.9375/0.9125|0.9567/0.9575/0.9625/0.9500；0.0433/0.0425/0.0375/0.0500|RX`20-1`:0.7188；class`14-10`:0.6167,`14-7`:0.5833,`6-15`:0.9000,`8-20`:0.7750；day`2021_03_01`:0.7917,`2021_03_08`:0.7857,`2021_03_15`:0.5759|FAIL（0.70）|PENDING|
|F4C_CLIC12|C raw phase control|20-1；d0/1/2；LEO C/L/R|0.7146/0.7146/0.2917/0.7146/0.6863|0.7125/0.6875/0.7438|0.0271/0.0250/0.0400/0.0163|0.5981/0.8737/0.9000|0.0083/0.0000/0.0125/0.0125；0.0187/0.0375/0.0000/0.0187；0.9729/0.9625/0.9875/0.9688|0.9542/0.9475/0.9525/0.9625；0.0458/0.0525/0.0475/0.0375|RX`20-1`:0.7146；class`14-10`:0.2917,`14-7`:0.9083,`20-15`:0.8167,`8-20`:0.8417；day`2021_03_01`:0.7485,`2021_03_08`:0.7063,`2021_03_15`:0.6863|FAIL（0.70）|PENDING|
|F4G_CLIC12|G complex local invariant curvature|20-1；d0/1/2；LEO C/L/R|0.6917/0.6917/0.1333/0.6917/0.6863|0.6937/0.6750/0.7063|0.0129/0.0125/0.0112/0.0150|0.6726/0.9019/0.8792|0.0042/0.0000/0.0000/0.0125；0.0187/0.0250/0.0312/0.0000；0.9771/0.9750/0.9688/0.9875|0.9433/0.9337/0.9363/0.9600；0.0567/0.0663/0.0638/0.0400|RX`20-1`:0.6917；class`14-10`:0.1333,`14-7`:0.9667,`20-15`:0.8083,`8-20`:0.8583；day`2021_03_01`:0.6946,`2021_03_08`:0.6937,`2021_03_15`:0.6863|FAIL（0.70）|PENDING|
|F5C_CLIC12|C raw phase control|20-1；d0/1/2；LEO C/L/R|0.4229/0.4229/0.1917/0.4229/0.3212|0.4000/0.4188/0.4500|0.0196/0.0275/0.0063/0.0250|0.5300/0.8509/0.9646|0.0167/0.0125/0.0187/0.0187；0.0437/0.0187/0.0688/0.0437；0.9396/0.9688/0.9125/0.9375|0.9142/0.9237/0.8975/0.9213；0.0858/0.0762/0.1025/0.0788|RX`20-1`:0.4229；class`14-10`:0.2500,`14-7`:0.1917,`20-15`:0.8000,`20-19`:0.4500；day`2021_03_01`:0.5671,`2021_03_08`:0.3212,`2021_03_15`:0.3775|FAIL（0.70）|PENDING|
|F5G_CLIC12|G complex local invariant curvature|20-1；d0/1/2；LEO C/L/R|0.4458/0.4458/0.1250/0.4458/0.3515|0.4437/0.4250/0.4688|0.0271/0.0400/0.0275/0.0138|0.5249/0.8519/0.9563|0.0125/0.0125/0.0187/0.0063；0.0125/0.0125/0.0250/0.0000；0.9750/0.9750/0.9563/0.9938|0.9504/0.9375/0.9325/0.9812；0.0496/0.0625/0.0675/0.0187|RX`20-1`:0.4458；class`14-10`:0.1750,`14-7`:0.1250,`20-15`:0.8667,`20-19`:0.6167；day`2021_03_01`:0.5732,`2021_03_08`:0.3515,`2021_03_15`:0.4106|FAIL（0.70）|PENDING|
|F6C_CLIC12|C raw phase control|20-1；d0/1/2；LEO C/L/R|0.4833/0.4833/0.3250/0.4833/0.3789|0.4313/0.5188/0.5000|0.0267/0.0288/0.0300/0.0213|0.5279/0.8538/0.9646|0.0083/0.0063/0.0063/0.0125；0.0375/0.0750/0.0312/0.0063；0.9542/0.9187/0.9625/0.9812|0.9263/0.8700/0.9375/0.9712；0.0737/0.1300/0.0625/0.0288|RX`20-1`:0.4833；class`14-7`:0.5083,`20-15`:0.7250,`20-19`:0.3750,`6-15`:0.3250；day`2021_03_01`:0.5542,`2021_03_08`:0.3789,`2021_03_15`:0.5163|FAIL（0.70）|PENDING|
|F6G_CLIC12|G complex local invariant curvature|20-1；d0/1/2；LEO C/L/R|0.5021/0.5021/0.3833/0.5021/0.3602|0.4688/0.5563/0.4813|0.0267/0.0512/0.0125/0.0163|0.5002/0.8318/0.9646|0.0271/0.0563/0.0000/0.0250；0.0437/0.0625/0.0187/0.0500；0.9292/0.8812/0.9812/0.9250|0.9371/0.8725/0.9788/0.9600；0.0629/0.1275/0.0213/0.0400|RX`20-1`:0.5021；class`14-7`:0.3833,`20-15`:0.7750,`20-19`:0.4667,`6-15`:0.3833；day`2021_03_01`:0.6084,`2021_03_08`:0.3602,`2021_03_15`:0.5359|FAIL（0.70）|PENDING|

## 结果判定与边界

- 12/12`passed=false`是预注册unknown floor的真实失败结果，不是技术失败。全局unknown显式拒识率范围为C=`0.0192–0.0800`、G=`0.0129–0.0612`；三种LEO weak逐scene均低于0.70，因此当前没有任何候选通过真实unknown确认门。
- known target LEO-weak、unknown rejection、open-set、known false reject/defer/coverage和receiver/class/day DG全部来自同一candidate receipt；没有跨候选拼接极值。该run不含ADV reference，所有行的ADV状态保持`PENDING`，不得据此写非劣或晋级结论。
- 这是有效的target-only计量闭环和失败证据，不触发按性能停止规则；不修改阈值、不重跑prediction、不重训、不把target结果反馈到候选选择。后续ADV3B02必须使用配置等价的独立baseline流程完成比较，不能把本表的unknown失败改写成ADV结论。

## 封存receipt索引

|候选|fold_config_key|metrics receipt SHA-256|
|---|---|---|
|F1C_CLIC12|`61aaf8326e9ba72c577ad65caa2e39dc602b3a312589e6bfa50a85ec3c04c4fa`|`1ebf954ab1dcf33e4ea62e873172bc062161c1f46d9f3f570d63e284b8cdbe1a`|
|F1G_CLIC12|`61aaf8326e9ba72c577ad65caa2e39dc602b3a312589e6bfa50a85ec3c04c4fa`|`e95b8216a4c581a4a28ff3e14bb315f1091dafd15b2d1e881526a0655c52c32a`|
|F2C_CLIC12|`b47d7cc0686ace54df1e602949e3e185b341af2bc2533c2fb2a6e088d02d4953`|`63e40d67b0d55bc6864d94e81f5e8456ab0a7f649308723fda2df6678aa455e5`|
|F2G_CLIC12|`b47d7cc0686ace54df1e602949e3e185b341af2bc2533c2fb2a6e088d02d4953`|`73c39cb85069d9cbbc40e47d1261c17c7b0670d524cbf756b26eed3da0e0cb57`|
|F3C_CLIC12|`b9779bf5aaa54874a14f35e890020074f9dcfac774a23664e4dc8a9e847959ec`|`ccb423fd330921a1d17342290411b81d3766c27059c6e49c9343fd3efdebb9d0`|
|F3G_CLIC12|`b9779bf5aaa54874a14f35e890020074f9dcfac774a23664e4dc8a9e847959ec`|`ce721cb9c04108dc89959f18cce2af77a51fce0e26ef2952d2372512a41983b2`|
|F4C_CLIC12|`eab70193e46a29fe8f2a48fa73e572894d9000d433e50ce6db047b9d2ad0135a`|`657a312d646bfdaaa252c0cdeba2adef98deeb01b4f659c04dd0ad530e708700`|
|F4G_CLIC12|`eab70193e46a29fe8f2a48fa73e572894d9000d433e50ce6db047b9d2ad0135a`|`97e42ca60b6b0d295d8cb6b9e75769f3256f6e5e65f91cb73ded209628f20150`|
|F5C_CLIC12|`3201a4adc7f95f75fb82775ac4303f9a5f43c58b9c3f0eb388ac7eb6d91e240d`|`a1c6e84587c34ce7ba1fc3816f0e39f600aedf743e8b79910658d659631553e7`|
|F5G_CLIC12|`3201a4adc7f95f75fb82775ac4303f9a5f43c58b9c3f0eb388ac7eb6d91e240d`|`fa034d7937562a07a7cac37b8d401a20087c2d913824f01cbf0cd774fede6d94`|
|F6C_CLIC12|`05352c3b102513a85db64ff905e47f0227314f4523b69710ccb25d6262d7619d`|`50570d8dffa3eb5657236f9818a120828c9ea8b3621daf83827e4a6b00181c2b`|
|F6G_CLIC12|`05352c3b102513a85db64ff905e47f0227314f4523b69710ccb25d6262d7619d`|`0742c1c6e12baadae2a926d5e982b333ee8f202bc5d0e0772acbd8d6dcac4cd1`|

完整receipt位于N607`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_target_metrics_20260812_v1/metrics/`；日志位于同一项目根的`logs/phase1_clic_target_metrics_20260812_v1/`。本地报告与Git镜像必须保持同一SHA。
