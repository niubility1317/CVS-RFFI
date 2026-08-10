# P1-RCMMC后冻结公平评价合同与实现追踪

## 目的与边界

本卡冻结`P1_RCMMC`的后冻结接口，而非训练性能结论。训练根固定为`phase1_rcmmc12_20260811_v1`，后冻结根和矩阵固定为`phase1_rcmmc_postfreeze_20260811_v1`，候选固定为`F{1..6}{C|G}_RCMMC12`。专属schema为`cvs.phase1.rcmmc_receipt.v1`、`cvs.phase1.rcmmc_lv_export.v1`、`cvs.phase1.rcmmc_leo_binding.v1`和`cvs.phase1.rcmmc_postfreeze_pair.v1`。历史clean NPZ叶名`icmt_clean_l_v_proxy_final_only.npz`只是不变的公平核文件叶；manifest、LEO binding、pair JSON及训练receipt不得持久化`icmt_*`、`hscf_*`、`rcat_*`、`recte_*`、`rcrmd_*`或`cagm_*`旧候选身份字段。

本轮只实现本地接口，不读取真实checkpoint、ManySig、sealed artifact或任何性能数值，不访问N607。独立actual-diff审查已在最终六文件冻结面给出`P0=0 / P1=0 / ALLOW_LOCAL_GIT_RELEASE`；真实12臂、42步原件、性能、Phase2和Phase3能力仍由唯一Runner保持deferred。即使将来的六fold门全部通过，也只能给出`PENDING_MAIN_REVIEW_FULL_6_FOLD`，不得把本地技术放行写成性能或晋级结论。

## 冻结训练与评价合同

每个训练receipt必须从原始`rcmmc_receipt`重开并调用`phase1_rcmmc.validate_rcmmc_terminal_receipt`。它必须证明：`B=128`、`d=160`、local4、来自source split receipt的有序7个source receiver的count/SHA、三scene各28个receiver×class cell及合计84-cell共同coverage、固定分母28、同物理clean/单LEO行序、共同source split/warm-start/new AdamW/AMP/head路径。C必须为`enabled=false,lambda=0`且RCMMC辅助字段N/A或0；G必须为`enabled=true,lambda=.02`、三scene均有positive-D批次，并有一次四参raw-unscaled VJP：`clean_feat_joint`与exact head为None-or-zero，LEO`feat_joint`与shared encoder为finite-nonzero。receipt只封存标量、计数及SHA，不封存RX token、IQ、feature或moment matrix。

后冻结每臂只以clean L的`feat_joint`拟合float64安全totalized-L2对角Gaussian：先mask正范数再相除，零行保留为零，不加epsilon；feature、范数、拟合、NLL、unknown量或聚合值任一nonfinite立即fatal。每类variance固定`ddof=1`，`0.9×class+0.1×class-equal pooled`收缩与`1e-6`下限；完整Gaussian-NLL和稳定logsumexp定义连续`u=log(4)-logsumexp(-NLL)`。V、proxy、held、target、day、fold及U均零fit/零训练反馈；L、V、proxy的已封存行全部保留。

42步科学面固定为`12 clean+12 LEO/physical TX-RX-day binding+12 fixed400 proxy JSON/CSV+6 same-fold pair`。proxy固定days=`2021_03_01,2021_03_08`、RX=`1-1,1-19,14-7,18-2,19-2,2-1`、seed=`7281148`、每TX上限400、总数400。F6必须由sealed原件重开F1--F5 clean/LEO/binding/proxy JSON/CSV和当前C/G checkpoint，核对当前SHA并重算所有summary、four-floor、fold/global overall、AUROC与u-gap双strict非补偿gate；不得接受prior自报摘要。

## 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|RCMMC-PF-01|身份冻结|专属root、candidate、matrix和四schema；拒绝旧候选identity|全部4文件|verified|`ssr-gpu`编译、导入、CLI help与旧identity纯负例|稳定NPZ叶名不是身份|
|RCMMC-PF-02|receipt重开|原始RCMMC schema/method、B128/d160/local4、Rs count/SHA、fixed28、C/G和terminal重验|clean、LEO、pair|verified|合成C/G terminal；顶层/嵌套receipt、clean manifest和LEO binding共8个raw receiver token反例拒绝；旧identity篡改拒绝|不存RX token；真实checkpoint deferred|
|RCMMC-PF-03|G辅助|3×28共同/aux coverage、每scene positive-D、一次四参raw VJP；clean/head None-or-zero|clean、LEO、pair|verified|合成84-cell与四参VJP payload闭合/缺clean负例|无新forward/state|
|RCMMC-PF-04|C/G公平|同physical/order/source split/warm-start/new AdamW/AMP/head路径逐字段绑定|pair|verified|C/G共同投影与SHA闭合|G-only字段不强行相等|
|RCMMC-PF-05|clean与Gaussian|仅clean L fit；float64 safe totalized-L2、zero保留、nonfinite fatal、diagonal Gaussian/NLL|clean、pair|verified|正行/零行/nonfinite及L-only合成Gaussian/NLL|V/proxy/U零fit|
|RCMMC-PF-06|LEO/ManySig|source-only单LEO、三scene、physical TX/RX/day、selection和当前SHA闭合|LEO、pair|implemented|冻结binding/physical覆盖接口已编译|真实ManySig由Runner|
|RCMMC-PF-07|fixed400 proxy|固定days/RX/seed/max/总数400，NPZ/JSON/CSV/physical SHA闭合|clean、pair|implemented|常量与raw-logit重算路径已编译|CLI不可调；真实JSON/CSV deferred|
|RCMMC-PF-08|门|clean6/LEO18四floor、fold/global overall、AUROC与u-gap双strict且非补偿|pair|implemented|签字公平核委托与RCMMC verdict映射已编译|无性能结论|
|RCMMC-PF-09|F6重开|F1--F5 sealed raw artifact/current SHA/receipt/prior重算|pair|implemented|F6 raw reopen、receipt/common binding与proxy重算路径已编译|不得信自报|
|RCMMC-PF-10|真实接口|真实checkpoint、ManySig、sealed42步原件和独立P0/P1|外部Runner|deferred|Runner预检/独立审查|本轮不得解除|

## 当前实施状态

当前为`LOCAL_VERIFIED / INDEPENDENT_REVIEW_PASSED / NO_PERFORMANCE_INTERPRETATION`，追踪计数仍为`verified=5,implemented=4,deferred=1,rejected=0,blocked=0`。已实际执行`ssr-gpu`串行`py_compile`、三模块导入、三份CLI help、postfreeze聚焦18项、RCMMC核心+postfreeze联合33项、float64零/非有限/L-only Gaussian纯函数、合成C/G terminal/common binding、8个raw receiver token对抗反例、42步dry-run和`git diff --check`。独立审查在最终实现内容上复验四参VJP、C/G绑定、F6原件重开及上述负测后给出`P0=0 / P1=0`。真实checkpoint、ManySig、sealed clean/LEO/proxy JSON/CSV、42步运行及性能解释始终保持deferred；最高风险是首次真实receipt与sealed42步artifact闭合，不得由模板迁移或本地合成检查替代。
