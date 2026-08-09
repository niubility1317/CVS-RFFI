# P1-CAGM后冻结公平评价合同与实现追踪

## 目的与边界

本文件冻结`phase1_cagm12_20260810_v2`的P1-CAGM后冻结评价接口。它只复用已签ICMT v2的公平评价尺度、数据绑定与反篡改链，不复用ICMT的训练机制、损失或结论。当前代码与本地测试只证明接口和负例防线，不构成12臂训练、42步执行、性能提升、Phase2结果或Phase3能力结果。

训练根固定为`phase1_cagm12_20260810_v2`，后冻结矩阵固定为`phase1_cagm_postfreeze_20260810_v2`，候选固定为`F{1..6}{C|G}_CAGM12`。独立schema固定为`cvs.phase1.cagm_lv_export.v1`、`cvs.phase1.cagm_leo_binding.v1`、`cvs.phase1.cagm_postfreeze_pair.v1`。v1从未执行正式42步或产生性能结果，并因训练receipt缺少P1共同绑定与逐臂joint-zero-mask闭包而阻断；v2是唯一冻结执行面，不覆盖也不续跑v1。

## 冻结评价核

只由clean L的`z_id=feat_joint`拟合float64 totalized-L2对角高斯：正范数行映射为`z/||z||`，精确零范数行映射为零；任意非有限特征或范数立即失败。每类用ddof=1估计方差，再以`0.9*class+0.1*pool`收缩，并施加`1e-6`下限；评分是完整Gaussian-NLL，连续未知量为稳定`logsumexp`上的`u=log(4)-logsumexp(-NLL)`。V和proxy绝不fit，所有L/V/proxy行均保留。

proxy正式输入逐字固定为days=`2021_03_01,2021_03_08`、RX=`1-1,1-19,14-7,18-2,19-2,2-1`、seed=`7281148`、每TX上限=`400`、总数=`400`，不提供可调的正式CLI自由度。

## 追踪矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|PF-01|冻结合同1|固定root、matrix、F1..F6 C/G、三份CAGM schema|全部6文件|verified|`py_compile`、import和CAGM pair闭合|不回落为ICMT候选/root/schema|
|PF-02|冻结合同2|精确42步：12 clean、12三LEO+binding、12 logits proxy、6 pair；沿用GPU映射|launcher|verified|`bash -n`；dry-run=`42=12+12+12+6`|未执行真实性能|
|PF-03|冻结合同3|L-only`feat_joint`公平Gaussian-NLL、totalized-L2、float64、ddof1、收缩、floor、stable logsumexp、V/proxy零fit|pair evaluator|verified|float64/zero/nonfinite与端到端pair测试|只复用公平评价尺度|
|PF-04|冻结合同4|固定proxy输入和NPZ/manifest/JSON/CSV/physical/pair/F6全链重验；同步一行篡改失败|clean exporter、pair evaluator、tests|verified|固定CLI拒绝；pair篡改和F6 SHA同步后一行负例|正式输入不可调|
|PF-05|冻结合同5|ManySig路径/SHA、source selection、physical keys、NPZ SHA、逐scenario TX/RX/day完整与独立CAGM sidecar|LEO exporter、pair evaluator|verified|合成全链sidecar及原ICMT v2模板回归|继承信任链，不继承训练机制|
|PF-06|冻结合同6|CAGM`training_final_only`checkpoint、arm/candidate/root/head/class及receipt/terminal重验|clean exporter、LEO exporter、pair evaluator|verified|checkpoint篡改负例和pair每臂重载|读取原始checkpoint，不信summary|
|PF-07|冻结合同6|C=.00/G=.02、divisor10、detach、joint zero mask、common batch/scene/order、新AdamW、4+6 term、finite、G VJP、head none/zero、C零/N/A|clean exporter、pair evaluator、tests|verified|C/G terminal、权重/divisor、VJP/head、term coverage负例|C维持N/A/0|
|PF-08|冻结合同7|F6重读F1-F5clean/LEO/binding/proxyJSON+CSV，重算摘要、delta、gate并逐字段比对|pair evaluator|verified|六fold F6 raw-artifact recompute；一行特征+SHA同步仍失败|不信prior派生delta|
|PF-09|冻结合同8|非补偿门：clean6/6、LEO18/18四floor、每fold/全18overall、6/6两项strict proxy|pair evaluator|verified|CAGM sixfold aggregate和gate命名测试|失败固定`REJECT_P1_CAGM_PERMANENT`|
|PF-10|冻结合同9|明确公平评价合同而非ICMT机制复用且无性能结论|本文件、launcher输出|verified|文案审阅；测试断言无`ALLOW`|不自签ALLOW|
|PF-11|冻结合同10|迁移ICMT P0/P1负例并增加CAGM receipt/arm/schema/root、权重/divisor、VJP/head、zero-mask/term coverage负例|tests|verified|CAGM focused38 passed；合并ICMT模板69 passed|不改训练面|
|PF-12|P1-1|训练与postfreeze根非覆盖升级到v2；只接受`cvs.phase1.cagm_receipt.v2`|全部6文件|verified|`py_compile`、联合69项、bash语法、v2 dry-run42通过|候选名与三份postfreeze schema不变|
|PF-13|P1-1|原始receipt的`joint_zero_mask_aux_only`必须逐臂严格存在：G=True、C=False；manifest/sidecar只复制已证明值|clean、LEO、pair、tests|verified|缺失/反值/manifest伪报负例通过|不得默认补True|
|PF-14|P1-2|C/G共同训练绑定逐字段严格比较，双方optimizer必须为AdamW，键集合与类型严格|pair、tests|verified|G-only sequence/rows/scenario/baseline/init/optimizer及类型/键负例通过|current pair与F6 prior raw重算均调用|
|PF-15|P1-2|pair持久化共同绑定摘要与通过标志；F6从原始receipt重算并逐字段核对|pair、tests|verified|raw G篡改与prior自报篡改负例通过|不信prior pair自报共同绑定|

## 交付与验证边界

授权写入仅限本文件、`code/export_phase1_cagm_features.py`、`code/export_phase1_cagm_leo_features.py`、`code/evaluate_phase1_cagm_postfreeze_pair.py`、`code/scripts/launch_phase1_cagm_postfreeze_20260810.sh`与`code/tests/test_phase1_cagm_postfreeze.py`。不修改`train_ssdg.py`、`phase1_cagm.py`、任何CAGM训练launcher/report或ICMT/GD/CB/CP/SCB/CARE/CIRF文件；不访问N607、不运行性能、不提交Git。

最终只运行串行`ssr-gpu`语法/窄测试、ICMT模板回归、`bash -n`、42步dry-run和`git diff --check`。截至本地核验，追踪项为`verified=15,deferred=0,rejected=0,blocked=0`。未发现`BLOCKED_BY_REAL_INTERFACE`；真实CAGM final checkpoint/完整训练artifact尚未由本任务读取或评价，训练工件可用后仍须在不改冻结合同的前提下执行正式42步，不能以本地合成测试代替性能结果。
