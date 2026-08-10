# P1-RCAT后冻结公平评价合同与实现追踪

## 目的与边界

本文件冻结`phase1_rcat12_20260810_v1`的P1-RCAT后冻结评价接口。它只复用已签RCRMD后冻结实现所承载的ICMTv2公平Gaussian-NLL核、source/LEO物理绑定与F6原始artifact重算，不复用RCRMD、CAGM或ICMT的训练损失、状态或性能结论。代码和本地合成验证只能证明接口、冻结常量和负例防线；它们不证明12臂训练完成、42步真实执行、性能改善、Phase2结果或Phase3能力。任何完整矩阵通过仍只能写为`pending-main`。

训练根固定为`phase1_rcat12_20260810_v1`，后冻结矩阵固定为`phase1_rcat_postfreeze_20260810_v1`，候选固定为`F{1..6}{C|G}_RCAT12`。独立schema固定为`cvs.phase1.rcat_lv_export.v1`、`cvs.phase1.rcat_leo_binding.v1`和`cvs.phase1.rcat_postfreeze_pair.v1`；训练receipt固定为`cvs.phase1.rcat_receipt.v1`。clean artifact沿用已签公平核的稳定文件叶`icmt_clean_l_v_proxy_final_only.npz`，但manifest、binding、pair JSON和所有receipt必须是RCAT身份。

## 冻结评价核

每个候选只以clean L的`z_id=feat_joint`拟合float64分段totalized-L2对角Gaussian：正范数行映射为`z/||z||₂`，精确零范数行映射为零，任何非有限feature或范数立即失败。该规则与训练float32的分段规则相同，但不宣称字节相同。每类方差采用`ddof=1`，再以`0.9×class+0.1×class-equal pooled`收缩并设置`1e-6`下限。评分为完整Gaussian-NLL；连续unknown量为稳定logsumexp上的`u=log(4)-logsumexp(-NLL)`。V和proxy不参与fit，L、V、proxy全部行保留。

proxy正式常量为days=`2021_03_01,2021_03_08`，RX=`1-1,1-19,14-7,18-2,19-2,2-1`，seed=`7281148`，每TX上限400，总数400。source-only三LEO导出固定核验ManySig SHA/path、selection、physical ID以及逐scene TX/RX/day绑定。F6必须逐项重读F1--F5的clean、LEO、binding、proxy JSON/CSV和当前checkpoint，核对当前SHA并重算summary、delta、floor及proxy gate；不得信任prior pair自报摘要。

RCAT训练绑定必须由原始receipt重新证明：frozen mode；C=`enabled=false,lambda=0`，G=`enabled=true,lambda=.02`；source receiver集合`0..6`及count/SHA/provenance；每scene固定28个RX×local4格、三scene共84格；固定分母`1/28`；共同same-physical、RX/class/scene的`n_rc`与batch order；相同warm-start、head/class/order/split及新AdamW初态。C的aux必须N/A或0；G必须有`positive_q>0`、有限loss和首个positive-q批次的raw-unscaled `feat_joint/shared encoder`有限非零VJP。exact head对aux应为None/zero且不要求非零；另须证明C/G共同`L_base→feat_joint→exact head→tx_logits`路径为live。训练float32账本由RCAT终态validator复核。

RCAT与共同clean-detach→LEO logits KL的零集合不相同：当clean/LEO特征均非零时，RCAT的`q=0`要求`z_leo=a·z_clean`且`a>0`；两者同为精确零向量时也为零。exact head零空间中的方向漂移可使logits KL为零而RCAT为正，这是RCAT相对共同KL唯一新增的可识别部分；反之，正径向缩放可使RCAT为零而KL仍为正。后冻结只审计该训练身份，不据此预称修复RX、day、unknown或性能。

分类门为clean6折四floor、LEO18格四floor、每fold三scene equal-weight overall与global18-cell equal-weight overall。proxy门为每fold`ΔAUROC>0`且`Δ(mean u_proxy-mean u_V)>0`，要求6/6。所有门均非补偿；任一失败永久`REJECT`，全部通过也只可标记`pending-main`。

## 可追溯性矩阵

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|RCAT-PF-01|身份冻结|训练根、候选、后冻结矩阵、四个schema与稳定clean叶固定|三个脚本、本文件|verified|常量导入与CLI必填项检查|仅RCAT身份|
|RCAT-PF-02|公平核|L-only、float64 totalized-L2、zero保留、nonfinite fatal、ddof1、0.9/0.1、1e-6、full NLL、stable LSE|pair|verified|zero/nonfinite合成核；RCRMD后冻结27项回归|V/proxy零fit；精确委托signed kernel|
|RCAT-PF-03|proxy常量|固定days/RXs/seed/max/TX/总数400，闭合NPZ/JSON/CSV/physical SHA|clean export、pair|verified|固定常量与raw-logit重算路径静态检查|真实artifact由42步runner产生|
|RCAT-PF-04|source-LEO|ManySig SHA/path、selection、逐scene physical/TX/RX/day完整绑定|LEO export、pair|verified|binding字段及signed-kernel委托检查|未读取真实ManySig性能|
|RCAT-PF-05|RCAT receipt|C/G、Rs0..6、28×3、1/28、same-physical、warm-start/head/class/order/split/AdamW|三个脚本|verified|原始checkpoint validator静态检查与合成receipt闭合|真实checkpoint兼容性待runner|
|RCAT-PF-06|RCAT aux边界|C aux N/A/0；G positive-q、feat_joint/shared encoder VJP；head aux None/zero且共同head路径live|clean export、pair|verified|C/G合成receipt与head非零VJP负例；RCAT 11项回归|不误称head aux非零|
|RCAT-PF-07|C/G公平绑定|共同physical/RX/class/scene n_rc、batch order及基础训练字段逐字段相同；G-only字段不强制相等|pair|verified|合成C/G common projection闭合|非补偿|
|RCAT-PF-08|F6原件重算|重读F1--F5 raw clean/LEO/binding/proxy JSON/CSV/checkpoint并按当前SHA重算|pair|verified|signed prior raw re-open路径复用；RCRMD后冻结27项回归|不信prior摘要；真实F6待runner|
|RCAT-PF-09|门与终态|clean6/6、LEO18/18四floor、fold/global overall、proxy双门6/6；仅REJECT或pending-main|pair|verified|signed fold/matrix聚合核复用；RCRMD后冻结27项回归|无性能结论；不自签P0/P1|
|RCAT-PF-10|本地验证|`ssr-gpu`串行py_compile、最窄既有回归、diff-check、四文件SHA|全部4文件|verified|py_compile；RCAT 11；RCRMD后冻结27；合成核；diff-check|不访问N607|

## 授权边界与当前状态

当前冻结发布仅新增本文件、`code/export_phase1_rcat_features.py`、`code/export_phase1_rcat_leo_features.py`、`code/evaluate_phase1_rcat_postfreeze_pair.py`、`code/tests/test_phase1_rcat_postfreeze.py`和`code/scripts/launch_phase1_rcat_postfreeze_20260810.sh`。不得修改trainer、RCAT core、既有共享实现、report或`conversation_index`；不得在本地实现与复核阶段访问N607或运行真实性能。正式发布与N607执行须另经独立P0/P1终裁、Git版本化、预注册和唯一Runner交接。

当前实现追踪项为verified=10、deferred=0、rejected=0、blocked=0。真实性能与artifact证据另行deferred：本轮未读取真实RCAT12终态checkpoint、真实ManySig导出或任何后冻结指标。最高风险是实际receipt/artifact与本接口的首次闭合；它属于正式42步runner及独立P0/P1复核，不能由接口迁移或合成验证解除。
