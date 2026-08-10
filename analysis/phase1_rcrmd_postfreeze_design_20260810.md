# P1-RCRMD后冻结公平评价合同与实现追踪

## 目的与边界

本文件冻结phase1_rcrmd12_20260810_v1的P1-RCRMD后冻结评价接口。它只复用已签ICMTv2的公平Gaussian-NLL核、source/LEO物理绑定与F6原始artifact重算，不复用ICMT或CAGM的训练机制、损失或结论。代码和本地合成测试只能证明接口、冻结常量和负例防线；它们不构成12臂训练完成、42步真实执行、性能改善、Phase2结果或Phase3能力结果。任何通过的完整矩阵仍只写为pending-main。

训练根固定为phase1_rcrmd12_20260810_v1，后冻结矩阵固定为phase1_rcrmd_postfreeze_20260810_v1，候选固定为F{1..6}{C|G}_RCRMD12。独立schema固定为cvs.phase1.rcrmd_lv_export.v1、cvs.phase1.rcrmd_leo_binding.v1和cvs.phase1.rcrmd_postfreeze_pair.v1。clean artifact沿用已签公平核的稳定文件叶icmt_clean_l_v_proxy_final_only.npz，但manifest、binding、pair JSON和所有receipt必须是RCRMD身份。

## 冻结评价核

每个候选仅以clean L的z_id=feat_joint拟合float64 totalized-L2对角Gaussian。正范数行映射为z/||z||，精确零范数行映射为零，任何非有限feature或范数立即失败。每类方差采用ddof=1，再以0.9×class+0.1×class-equal pooled收缩并设置1e-6下限。评分是完整Gaussian-NLL；连续unknown量为稳定logsumexp上的u=log(4)-logsumexp(-NLL)。V和proxy绝不参与fit，L、V、proxy的全部行均保留。

proxy正式常量为days=2021_03_01,2021_03_08；RX=1-1,1-19,14-7,18-2,19-2,2-1；seed=7281148；每TX上限400；总数400。source-only三LEO导出固定为三场景、完整TX/RX/day绑定与ManySig SHA/path验证。F6逐项重读F1--F5的clean、LEO、binding、proxy JSON/CSV和当前checkpoint，重算summary、delta、floor与proxy gate；不信任prior pair自报摘要或同步更新后的SHA。

RCRMD训练绑定必须从原始receipt重新证明：frozen mode；C=false/lambda=0与G=true/lambda=.02；source receiver物理绑定、receiver集合0..6、count/SHA/provenance；每scene28个receiver×class格、三场景84格终态覆盖、固定分母1/28；共同physical/RX/class/scene n_rc与batch-order；相同warm-start、head/class/order/split和新建AdamW初态。C的aux保持N/A或0；G必须有active_q、loss、首个active批次的raw-unscaled shared encoder和exact classifier head VJP，float32账本误差界由原始terminal validator复核。pair只逐字段比较上述C/G共同公平字段，绝不把G-only aux字段强制等同于C。

分类门为clean6折四floor、LEO18格四floor、每fold三场景equal-weight overall与global18-cell equal-weight overall。proxy门为每foldΔAUROC>0与Δ(mean u_proxy-mean u_V)>0，均要求6/6。全部门非补偿；pair与矩阵仅产生REJECT或pending-main语义，不产生自动晋级结论。

## 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|RPF-01|冻结42步|12clean+12LEO/binding+12proxy JSON/CSV+6pair，沿用GPU映射|launcher|verified|bash-n；dry-run=42=12+12+12+6|不执行训练|
|RPF-02|公平核|L-only、float64 totalized-L2、zero保留、nonfinite fatal、ddof1、0.9/0.1、1e-6、full NLL、stable logsumexp|pair|verified|focused zero/nonfinite与Gaussian-NLL测试|V/proxy零fit|
|RPF-03|proxy常量|固定days/RXs/seed/max/TX/总数400并闭合NPZ/JSON/CSV/physical SHA|clean export、pair、tests|verified|固定CLI、source/proxy绑定及单行raw-logit攻击负例|正式CLI不可调|
|RPF-04|source-LEO|ManySig SHA/path、physical selection、逐scenario TX/RX/day完整绑定|LEO export、pair、tests|verified|LEO sidecar字段篡改负例|source-only|
|RPF-05|RCRMDreceipt|frozen C/G、source-RX provenance/count/SHA、28×3、1/28、warm-start/head/class/order/split/AdamW|clean export、LEO export、pair|verified|C/G原始receipt、schema、lambda、VJP、84格负例|读原始checkpoint|
|RPF-06|C/G公平比较|公共physical/RX/class/scene n_rc与batch-order逐字段相同；G-only aux不误比|pair、tests|verified|common mismatch与G-only边界测试|非补偿|
|RPF-07|F6|重读F1--F5 raw clean/LEO/binding/proxy JSON/CSV/checkpoint并重算|pair、tests|verified|F6五个prior重算、prior摘要与同步SHA后的raw artifact篡改负例|不信prior自报|
|RPF-08|门与语义|clean6、LEO18、fold/global overall、proxy双门6/6；仅pending-main|pair、tests|verified|six-fold合成矩阵与pair verdict测试|无自动晋级|
|RPF-09|验证|py_compile、focused、模板联合回归、bash-n、42步dry-run、diff-check|全部6文件|verified|py_compile；focused27；CAGM+RCRMD联合65；dry-run42|串行ssr-gpu|

## 授权边界

本轮只写入本文件、code/export_phase1_rcrmd_features.py、code/export_phase1_rcrmd_leo_features.py、code/evaluate_phase1_rcrmd_postfreeze_pair.py、code/scripts/launch_phase1_rcrmd_postfreeze_20260810.sh和code/tests/test_phase1_rcrmd_postfreeze.py。不得修改RCRMD core、训练接线或训练launcher，不访问N607，不提交Git，不以训练性能选择任何常量。

截至本地核验，追踪项为verified=9，deferred=0，rejected=0，blocked=0。最高风险仍是尚未读取真实RCRMD12终态checkpoint或真实ManySig artifact；该风险属于正式42步前的独立P0/P1复核和runner交接，不可由合成测试或本文件自行解除。
