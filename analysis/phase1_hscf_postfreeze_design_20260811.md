# P1-HSCF后冻结公平评价合同与实现追踪

## 目的与边界

本卡冻结训练根`phase1_hscf12_20260811_v2`的P1-HSCF后冻结接口。它仅借用已签RECTE、RCAT和RCRMD链中的source-only导出、float64 Gaussian-NLL、公平门与F6原始工件重开承载；候选、receipt、manifest、binding、pair、命令和运行根均为HSCF专属，禁止持久化`icmt_*`、`rcat_*`、`rcrmd_*`、`cagm_*`或`recte_*`身份。历史NPZ叶名`icmt_clean_l_v_proxy_final_only.npz`仅是公平核文件叶兼容，不代表运行身份。

本轮只实现并发布本地后冻结接口，不读取真实训练性能、不伪造checkpoint或ManySig原件。独立actual-diff复核已给出`P0=0/P1=0/ALLOW_LOCAL_GIT_RELEASE`；真实42步、性能、Phase2、Phase3和真实在轨声明仍保持外部deferred，完整矩阵即使通过全部门也只能为`PENDING_MAIN_REVIEW_FULL_6_FOLD`。

## HSCF训练receipt与C/G公平合同

训练receipt固定为`cvs.phase1.hscf_receipt.v1`，方法固定`P1_HSCF`，候选固定`F{1..6}{C|G}_HSCF12`。原始receipt必须逐臂重开并由`phase1_hscf.validate_hscf_terminal_receipt`重新校验：`B=128`、local4、固定分母512、`lambda_hscf=.02`；C为`enabled=false,lambda=0`且辅助计数/场景/VJP为N/A或零；G为`enabled=true`且clear/low/rain均有正项与一次raw-unscaled VJP，LEO raw logits、shared encoder与exact head weight均有限非零，clean raw logits和head bias为None/数值零。

pair必须从C/G原始receipt逐项比较共同GeoSat-C`training_final_only`warm-start、严格model keys、新AdamW空初态、AMP、source partition/L物理顺序SHA、TX/class order、同物理clean/LEO batch/order、三scene循环、B128/local4/denom512、共同live`L_base→feat_joint→exact head→tx_logits`路径。HSCF训练辅助不读取RX/day/fold；source-L、LEO导出与ManySig binding仍必须封存physical TX/RX/day和逐scene完整性，不能以训练receipt缺少RX/day字段替代这些原件。

## 冻结评价核

每候选只用source-L clean`feat_joint`拟合float64 totalized-L2对角Gaussian：正范数行映射为`z/||z||₂`，精确零行保持零，feature、范数、NLL、score或聚合量任何nonfinite均fatal。每类方差按`ddof=1`，`0.9×class+0.1×class-equal pooled`收缩，最小`1e-6`；评分为完整Gaussian-NLL，连续unknown量为稳定logsumexp的`u=log(4)-logsumexp(-NLL)`。V/proxy零fit，L/V/proxy全部行和零行保留。

proxy固定days=`2021_03_01,2021_03_08`、RX=`1-1,1-19,14-7,18-2,19-2,2-1`、seed=`7281148`、每TX最多400、总数400；JSON、CSV、NPZ、physical key和当前SHA必须闭合。四floor、clean6、LEO18、每fold三sceneoverall、global18-celloverall及每fold`ΔAUROC>0`和`Δ(mean u_proxy-mean u_V)>0`均为非补偿门，任何失败永久`REJECT_P1_HSCF_PERMANENT`。

F6必须重开F1--F5的raw clean NPZ、LEO NPZ、LEO binding、proxy JSON/CSV和当前C/G checkpoint，重新计算摘要、delta、floor、proxy gate与C/G共同receipt；不得接受prior pair自报摘要、缓存或同步后的自报SHA。

## 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|HSCF-PF-01|身份冻结|HSCF专用root、candidate、matrix、schema；仅NPZ叶名兼容|全部4文件|verified|导入与旧identity输出拒绝路径|旧identity fail|
|HSCF-PF-02|训练receipt|原始`hscf_receipt`、B128/local4/512/lambda、C零与G三scene正项/VJP|clean、LEO、pair|verified|12份v2真实terminal receipt由当前validator重开；合成篡改拒绝|checkpoint bytes仍由Runner核验|
|HSCF-PF-03|C/G公平|warm-start、new AdamW、AMP、physical/order、TX/class、common live path逐项一致|pair|implemented|same-fold common projection合成闭合|G-only不强行相等|
|HSCF-PF-04|clean|L-only`feat_joint`、V/proxy零fit、float64 totalized-L2/zero/nonfinite|clean、pair|verified|float64正行/零行纯函数检查|不读取性能|
|HSCF-PF-05|LEO/ManySig|source-only、三scene、physical TX/RX/day、selection、ManySig SHA/path|LEO、pair|implemented|冻结binding/receipt校验路径已编译|训练RX/day零读不削弱导出绑定|
|HSCF-PF-06|fixed proxy|days/RX/seed/400、JSON/CSV/NPZ current-SHA闭合|clean、pair|implemented|常量及raw-logit重算路径已编译|CLI不可调|
|HSCF-PF-07|门|四floor、fold/globaloverall与proxy双门均非补偿|pair|implemented|签字fold/matrix aggregate委托与HSCF verdict映射已编译|仅REJECT或pending-main|
|HSCF-PF-08|F6|F1--F5 raw reload和当前receipt/checkpoint重验|pair|implemented|F6 raw reload、receipt与proxy重算路径已编译|拒绝自报|
|HSCF-PF-09|42步|12clean+12LEO/binding+12proxy+6pair|launcher/Runner|implemented|`bash -n`与dry-run严格`42=12+12+12+6`|N607唯一launch仍deferred|
|HSCF-PF-10|本地验证|官方`ssr-gpu`编译、导入、窄纯函数、SHA|六文件|verified|py_compile、focused12、HSCF joint32、dry-run42、diff-check|独立复核`P0=0/P1=0`|
|HSCF-PF-11|真实接口|真实checkpoint、ManySig、sealed42步原件、独立P0/P1|外部Runner|deferred|独立P0/P1已通过；真实原件由Runner启动前核验|最高剩余风险|

## 资源与真实接口

后冻结总步数保持42=`12 clean+12 LEO/binding+12 proxy JSON/CSV+6 same-fold pair`。本地接口不增加训练forward、模型、持久state、cache、采样或第二LEO view；Gaussian与F6只做评价侧float64计算。真实接口仍需要每臂`final_ssdg.pth`、ManySig输入、sealed clean/LEO/proxy原件和唯一Runner路径；这些缺失时必须fail-closed/deferred，不能由本地合成检查解除。

## 当前实施状态

当前状态为`LOCAL_VERIFIED_V2_TRAINING_ROOT / P0=0 / P1=0 / ALLOW_LOCAL_GIT_RELEASE`；性能解释状态为`NO_PERFORMANCE_INTERPRETATION`。v2训练根已完成技术闭合，但不构成训练性能结果、候选晋级或任何性能声明。已执行的本地验证为：

```text
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; python -m py_compile export_phase1_hscf_features.py export_phase1_hscf_leo_features.py evaluate_phase1_hscf_postfreeze_pair.py
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <导入三模块、float64 totalized-L2正行与零行、合成HSCF C/G terminal receipt及common binding检查>
git -c core.autocrlf=false diff --check；并对四个未跟踪目标文件执行`git diff --no-index --check -- NUL <file>`
```

以上验证允许进入Git发布，但不替代真实42步。最高风险和唯一真实接口依赖是12臂`final_ssdg.pth`、ManySig bytes、sealed clean/LEO/proxy JSON/CSV和F6 prior原件的首次闭合；它必须由唯一Runner在启动前与运行后重开，不能由本卡或合成夹具自签。
