# P1-RECTE后冻结公平评价合同与实现追踪

## 目的与边界

本文件冻结`phase1_recte12_20260810_v1`的P1-RECTE后冻结评价接口。训练实现锚定提交`48a2c284b2cca8430320da16560748898ed3b9d5`，训练receipt固定为`cvs.phase1.recte_receipt.v1`。后冻结接口只复用已签RCAT/RCRMD链承载的公平Gaussian-NLL、source/LEO物理绑定和F6原始artifact重算模式；它不复用RCAT、RCRMD、ICMT或CAGM的训练身份、候选身份、运行根、schema、命令或性能结论。

本轮实现source-only后冻结接口、聚焦测试和42步launcher，并完成本地静态、纯函数、负例与dry-run验证。已封存的12臂RECTE训练原件只证明训练技术合同；在唯一N607 Runner实际执行42步前，本文件不能证明后冻结性能改善、Phase2结果、Phase3能力或真实在轨验证。任何完整矩阵即使通过全部非补偿门，也只能标记为`PENDING_MAIN_REVIEW_FULL_6_FOLD`，不得自动晋级或自签P0/P1。

训练根固定为`phase1_recte12_20260810_v1`，后冻结矩阵固定为`phase1_recte_postfreeze_20260810_v1`，候选固定为`F{1..6}{C|G}_RECTE12`。独立schema固定为`cvs.phase1.recte_lv_export.v1`、`cvs.phase1.recte_leo_binding.v1`和`cvs.phase1.recte_postfreeze_pair.v1`。为兼容已签公平核，可沿用历史NPZ叶名`icmt_clean_l_v_proxy_final_only.npz`；这只是一项叶名兼容，manifest、binding、pair JSON、receipt字段、候选ID、运行根和命令均必须是RECTE身份，不能输出`icmt_*`、`rcat_*`、`rcrmd_*`或`cagm_*`身份字段。

## RECTE训练receipt与C/G公平合同

P1-RECTE的训练核只读source-known-train L的TX、同物理RX和同一clean/单LEO输出。固定`R_s=0..6`、local4，共28个RX×class格；无序格对固定分母为`378=28×27/2`，不按occupied pair或positive pair重归一。G的`lambda_recte=0.02`，C的`enabled=false,lambda=0`且aux字段必须为N/A/0。G必须重验：

- `feat_joint`作为`z_id`，clean raw logits stop-gradient；`torch.func.functional_call`对当前exact CosFace head使用detach参数与clone/detach buffer重读，functional logits与live LEO logits逐元素相等；
- 每个clear/low/rain scene均存在positive-tail pair；每batch为固定canonical cell order、`n_rc`、occupancy、fixed378、无active重归一和functional equality留下receipt；
- 每scene首个positive-tail pair的未缩放`L_RECTE`独立VJP：`feat_joint_leo`及shared encoder有限非零；exact classifier head aux VJP严格为None/zero；诊断不触碰AMP、optimizer或RNG；
- C/G共同receipt逐项闭合：same physical、RX、class、scene、order、`n_rc`、warm-start、head/class order、split、source RX provenance/SHA、batch order、new AdamW初态和共同live`L_base→feat_joint→exact head→tx_logits`路径。

该receipt只证明训练合同。后冻结不重新训练、不读性能来选择实现，不新增forward、模型、状态、cache、重采样或第二LEO view。

## 冻结评价核与数据边界

每个候选仅以source-L clean `feat_joint`拟合float64 totalized-L2对角Gaussian：正范数行映射为`z/||z||₂`，精确零范数行保留为零，feature、范数、NLL、score或聚合量任一nonfinite均fatal。每类方差按`ddof=1`计算，再以`0.9×class+0.1×class-equal pooled`收缩并设`1e-6`下限。评分是完整Gaussian-NLL；连续unknown量为稳定logsumexp上的`u=log(4)-logsumexp(-NLL)`。V和proxy绝不参与fit，L、V、proxy的零行及全部行均保留。

固定proxy条件为days=`2021_03_01,2021_03_08`、RX=`1-1,1-19,14-7,18-2,19-2,2-1`、seed=`7281148`、每TX最多400、总数400。source-only三LEO导出必须逐scenario核验ManySig path/SHA、selection、physical key，以及完整TX/RX/day绑定。U不允许iterate、forward、loss、backward或optimizer；V只允许只读诊断，RECTE对V零fit、零loss、零backward、零optimizer、零calibration和零model-selection反馈；proxy和held也零训练。

分类门包括clean6折四floor、LEO18格四floor、每fold三scene equal-weight overall与global18-cell equal-weight overall。proxy门要求每fold`ΔAUROC>0`及`Δ(mean u_proxy-mean u_V)>0`，均为6/6。clean、LEO、四floor、fold/global overall和proxy双门全部非补偿：任何失败为`REJECT`；通过仅保留pending-main语义。

F6必须重开F1--F5的raw clean NPZ、LEO NPZ、LEO binding、proxy metrics JSON、proxy scores CSV和当前C/G checkpoint；在当前SHA下重算summary、delta、floor、proxy gate和共同receipt。任何prior pair自报摘要、缓存摘要、同步后SHA或手工字段都不能代替原件重算。

## 42步、资源与真实接口

冻结42步为12个clean export、12个LEO export/binding、12个proxy JSON/CSV和6个same-fold C/G pair。本轮文件面包括设计卡、三个后冻结脚本、聚焦测试和42步launcher；Runner、真实artifact与N607执行仍保持独立职责，不在本地实现中伪造或预判。

资源预算沿用已签source-only导出核：clean/LEO导出只对冻结checkpoint和固定source切片执行既有forward；Gaussian拟合与F6重算在评价侧使用float64；不增加训练forward、模型、保存状态、采样或视图。真实接口需要每个候选的`final_ssdg.pth`、sealed clean/LEO/proxy原件、ManySig输入和独立Runner路径；在这些原件实际到位且由独立P0/P1复核前，所有真实性能和42步状态均为deferred。

## 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|RECTE-PF-01|身份冻结|RECTE专用训练根、候选、后冻结矩阵、schema和仅叶名兼容说明|全部4文件|verified|`ssr-gpu`导入、常量和身份静态核验|禁止历史身份字段|
|RECTE-PF-02|训练receipt|`cvs.phase1.recte_receipt.v1`、Rs0..6、local4、28格、fixed378、lambda=.02、C aux N/A/0|clean export、LEO export、pair|implemented|原始receipt重开校验路径已编译|真实checkpoint待独立复核|
|RECTE-PF-03|functional公平|exact CosFace functional equality、共同live路径、每scenepositive-tail、独立VJP与head aux None/zero|clean export、pair|implemented|per-scene audit与head None/zero校验路径已编译|不将head aux误判为非零|
|RECTE-PF-04|C/G公平|physical/RX/class/scene/order/`n_rc`、warm-start、head/class、split、AdamW与batch order逐项共同绑定|pair|implemented|same-fold common projection和原始receipt重开路径已编译|G-only字段不强制C相等|
|RECTE-PF-05|clean导出|L-only float64 totalized-L2 Gaussian、zero保留、nonfinite fatal、V/proxy零fit|clean export、pair|implemented|签字公平核委托与zero-row导入检查|`feat_joint`；真实NPZ待独立复核|
|RECTE-PF-06|source-LEO|ManySig SHA/path、selection、physical key、逐scenario TX/RX/day完整绑定|LEO export、pair|implemented|冻结binding/receipt校验路径已编译|source-only|
|RECTE-PF-07|fixed proxy|days/RX/seed/max-per-TX/total=400、JSON/CSV/NPZ physical SHA闭合|clean export、pair|implemented|冻结常量及raw-logit重算路径已编译|CLI不可调|
|RECTE-PF-08|门|clean6、LEO18、四floor、fold/global overall、proxy双门6/6，全部非补偿|pair|implemented|签字fold/matrix aggregate委托和RECTE verdict映射已编译|只REJECT或pending-main|
|RECTE-PF-09|F6|重开F1--F5原始clean/LEO/binding/proxy/checkpoint并按当前SHA重算|pair|implemented|F6 raw reload、proxy CSV/JSON重算、当前checkpoint SHA路径已编译|拒绝prior自报|
|RECTE-PF-10|42步与资源|12clean+12LEO/binding+12proxy JSON/CSV+6pair；不新增训练资源|launcher、tests、本文件|verified|`bash -n`通过；dry-run精确12+12+12+6=42且无历史候选身份|真实执行仍由唯一Runner负责|
|RECTE-PF-11|本地验证|`ssr-gpu`串行py_compile、聚焦/共享回归、负例与diff检查|三个脚本、tests、launcher|verified|四Python文件py_compile；RECTE postfreeze 33 passed；RCAT+RCRMD共享回归60 passed；`git diff --check`通过|仅证明本地接口与冻结控制流|
|RECTE-PF-12|真实接口|真实checkpoint、ManySig、sealed原件和独立P0/P1|外部Runner|deferred|RECTE训练12臂原件已闭合；后冻结独立复核与Runner待执行|不预读或预判性能|

## 当前实施状态

当前实现状态为verified=3、implemented=8、deferred=1、rejected=0、blocked=0。实际运行的本地命令包括：

```text
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; python -m py_compile code\export_phase1_recte_features.py code\export_phase1_recte_leo_features.py code\evaluate_phase1_recte_postfreeze_pair.py code\tests\test_phase1_recte_postfreeze.py
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; python -m pytest -q code\tests\test_phase1_recte_postfreeze.py
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; python -m pytest -q code\tests\test_phase1_rcat_postfreeze.py code\tests\test_phase1_rcrmd_postfreeze.py
bash -n code/scripts/launch_phase1_recte_postfreeze_20260810.sh
bash code/scripts/launch_phase1_recte_postfreeze_20260810.sh --dry-run
git diff --check
```

这些检查只证明本地后冻结接口、负例和42步控制流，不替代真实artifact执行。最高风险仍是以真实C/G checkpoint、ManySig和sealed clean/LEO/proxy原件闭合F6 raw重开；该工作必须由独立P0/P1与唯一Runner完成。本追踪卡不替代独立审查、Git版本化、42步Runner交接或真实性能分析。
