# D107-SCMKRR/r1完整125研发与实验报告

状态：`ANALYZED / PERFORMANCE_WEAK / REJECTED_NO_RERUN`

## 目标与终止背景

|字段|值|
|---|---|
|experiment ID|`d107_scmkrr_target125_20260801_r1`|
|日期/operator|2026-08-01；主agent负责集成、数据与结果分析；Terra Max子agent分别实现核心、执行面和truth面|
|目标|以机制独立的新方法完成完整125，并与D62、D91、D92、SVRN-qKNN-BCRR全面比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|Git|仅本地提交；不push、不上传GitHub|

D106在r1-r7始终没有形成完整performance manifest，连续暴露执行接口缺陷。按用户要求，该路线已经终止，不创建r8。D107不得导入D106 K路由、RCMR qKNN或ReLU plus-view链。

## 冻结方法

候选：`D107-SCMKRR/r1`，即Support-Centered Mean-Embedding Kernel Ridge。

1. 只使用当前row的signed z_id support；所有行先L2归一化。
2. before的6个旧类support构成冻结domain anchor A；after新增类不能反向改变A。
3. 每个注册类构造归一化support均值原型p_c；K1直接使用唯一support，不估协方差、不fallback。
4. 带宽b为全部注册类原型两两平方距离的中位数，只允许机器精度下界totalization。
5. RBF核`k(x,y)=exp(-||x-y||²/b)`；以冻结A作RKHS均值中心化，消除当前receiver/scene公共分量。
6. 从密封Phase1 D106资产只读取tau/spectrum摘要生成无量纲ridge比率；ground原型不得直接参与query打分。
7. simplex目标`Y=I-11ᵀ/C`，`B=(K+λI)⁻¹Y`；每条query独立计算`k(q,P)ᵀB`并在全部注册类竞争。
8. 四臂固定：`M0=未中心化kernel prototype`、`M_DA=中心化kernel prototype`、`M_HEAD=未中心化simplex-KRR`、`M_JOINT=中心化simplex-KRR`。无ROUTED臂、无K路由。
9. 无query truth、role、quota、fit、update、selection、batch-count或global reassignment；old/new公式完全相同。

该方法不是D92的old/new协方差LDA，不是D62的Fisher行拼接，不是D91的OOF梯度残差，也不是D106/SVRN的qKNN/BCRR变体。K1仍产生新的非线性决策面。

## 资源预算

最大`C=26,K=10,|A|=60,D=160`时，建态核计算约0.934M MAC，单query约14.4k量级；持久数值状态预计约15.4KB，无梯度、epoch或新checkpoint。只复用已密封Phase1摘要和已验证target received-IQ输入。

## 冻结完整125

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102,713103,713104,713105,713106`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|arm/state|4臂；before/after|
|闭包|125 outer、375 scene、1500 arm pairs、3000 prediction surfaces、500 arm级聚合行|

K5必须复用matched K10物理池并只取前5shot；`receiver=3-19/rain/K1-new20`只作预登记压力切片，不用于选参或分支。

## 精简研发与发布流程

实现分为三个不重叠面：SCMKRR typed核心/数学负例、Target125矩阵与真实input/不可覆盖prediction、独立truth-side构建与评分。主agent只做diff集成和最小聚焦验证。通过一个真实no-truth state smoke后直接发布完整125，不设置source-held性能gate、候选扫描或额外签名流程。只有协议/确定性执行故障可以停止，性能弱则完整结束后淘汰并进入下一方法。

## 本地实现与验证

|项目|证据|
|---|---|
|核心|`92fcabca`；四臂SCMKRR、K1非fallback、冻结anchor、canonical wire|
|执行面|`a571328c`；125/375/1500/3000矩阵、真实D92封包、不可覆盖prediction|
|truth面|`0faf4f5d`,`1202d0b2`；先封存验证再构建750个truth surface并输出375个scene同row、500个outer-arm聚合|
|真实接口修正|`baeaa023`,`bda8de01`；signed视图改用`pre_relu`；D92和RDCE运行时身份分离；RDCE从固定wire内解析lineage；删除错误的before/after全query相等假设|
|method lock|`configs/stage2_d107_scmkrr_r1.json`；SHA256=`ae0fec3d82d3eaa6e135ae6340af723575ede1ccb3c12120628ffcad25a38788`|
|本地验证|`ssr-gpu`下7个D107文件`py_compile`通过；三组专项测试合计`44 passed`；`git diff --check`通过|
|发布边界|仅本地Git；不push、不上传GitHub；一个真实无truth smoke通过后立即完整125|

真实N607资产已只读核对：D92 matrix SHA256=`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；RDCE wire SHA256=`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`。D92生产truth sidecar与offline build receipt的路径、SHA、receiver、seed、K和new-count字段已用真实首行核对。

N607发布源码固定为commit=`79305f25114c91390dd0efca6683d60f966a2036`；本地Git archive=`E:\type10-7\code\snapshots\d107_scmkrr_target125_20260801_r1_source_79305f25.tar`，SHA256=`ad8e3c889767c9fa2b9e3846b5ef8d32e2ff0ed6b02e241f07db016898595d07`。远端不可覆盖run root预登记为`/home/szu2070436088/2510044040/CV-SincNet/runs/d107_scmkrr_target125_20260801_r1`，Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，GPU=`cuda:0`；日志、PID、prepared、smoke、predictions、truth和score均必须位于该run root内。

## 性能目标与分析

沿用当前目标：K10三slice要求`A_old≥92%`、`F_old≥85%`、`N≥92/90/86%`；K5/new20相对matched K10/new20的`A/F/N/H`下降均≤5pp；K1/new20相对同row D92要求`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确严格增加。

完成后必须保留同rowreceiver、seed、slice、scene、before/after old、old floor、seen-new、H、forgetting和correct count。D62、D92、SVRN用完整125比较；D91单列15行development证据。
+## N607运行与完整闭包

2026-08-01 direct preflight通过，8张RTX3090均为空闲。唯一run root首次创建，archive、method lock、D92 matrix、checkpoint和RDCE wire的远端SHA均与预登记一致；7个D107文件远端`py_compile`通过。

|阶段|结果|
|---|---|
|prepare|`TARGET125_D92_PACKAGES_LOCATED`；125 outer、375 scene、1500 arm-pair、3000 surface；query truth/role/fit/update/selection均为false|
|真实smoke|row0/clear；2 phase×4 arm闭合；异常0；receipt file SHA256=`10e0589278c55ce7785213c758b09b5ecc9f09b6d62928b2782b978ff6a46e82`|
|完整predict|wrapper PID=`3407594`，Python PID=`3407595`；19:23:52启动，19:26:38 exit=`0`，166秒；3000/3000 prediction surface；manifest sealed=true、truth_open=false|
|truth/score|prediction完整封存后才生成750个truth surface；score包含375个scene同row、1500个scene-arm、500个outer-arm|
|收尾|wrapper/child均退出；GPU0—7回到0%/1MiB；本地SSH/TCP22清空；无技术异常|

## D107完整125性能

为与D62/D92/SVRN历史主表一致，主比较采用125个outer-row指标均值；post correct为全量计数。补充的query-weighted值只解释全量正确数，不能与同row均值混排。

|arm|before old|after old|before floor|after floor|seen-new|H|forgetting|post correct|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|74.63%|49.86%|47.89%|24.37%|30.40%|37.11%|24.78pp|51,616/157,500|
|M_DA|74.63%|46.92%|43.80%|18.53%|29.52%|35.44%|27.71pp|49,633/157,500|
|M_HEAD|74.90%|51.12%|46.99%|22.93%|29.56%|36.66%|23.77pp|51,347/157,500|
|M_JOINT|74.77%|47.26%|44.03%|17.87%|30.11%|35.95%|27.51pp|50,262/157,500|

四臂均未接近D92；support centering的M_DA和M_JOINT进一步损伤after old与floor，simplex head仅把after old提高到51.12%，但seen-new和H没有形成联合收益。未根据结果选择arm或改参数；冻结主臂仍为M_JOINT。M_JOINT按全部query计数加权时seen-new=25.77%、H=33.36%；该值仅解释post correct。

### M_JOINT按slice

|slice|before old|after old|before floor|after floor|seen-new|H|forgetting|post correct|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|K10/new5|77.16%|59.10%|47.53%|26.53%|48.15%|51.73%|18.06pp|8,930/16,500|
|K10/new10|77.16%|52.51%|47.53%|19.93%|35.53%|41.76%|24.64pp|10,056/24,000|
|K10/new20|77.16%|49.38%|47.53%|18.40%|24.53%|32.62%|27.78pp|11,803/39,000|
|K5/new20|75.50%|45.88%|45.13%|16.13%|23.22%|30.69%|29.62pp|11,096/39,000|
|K1/new20|66.87%|29.43%|32.40%|8.33%|19.09%|22.96%|37.43pp|8,377/39,000|

K10三个slice的绝对目标全部失败。K5/new20相对K10/new20的after old、after floor、seen-new和H分别下降3.50pp、2.27pp、1.31pp和1.93pp，shot降级稳定性尚可，但绝对性能过低。K1/new20相对同口径D92分别低14.60pp(after old)、5.87pp(after floor)、8.06pp(seen-new)、10.45pp(H)，且forgetting高13.32pp，K1晋级条件全部失败。

### M_JOINT按receiver与scene

|receiver|before old|after old|after floor|seen-new|H|forgetting|
|---|---:|---:|---:|---:|---:|---:|
|20-1|70.43%|44.23%|12.53%|33.66%|37.58%|26.20pp|
|3-19|62.12%|35.17%|8.73%|18.28%|23.69%|26.96pp|
|7-14|85.56%|55.72%|24.93%|32.82%|41.01%|29.83pp|
|7-7|79.82%|57.08%|31.20%|29.72%|38.90%|22.74pp|
|8-8|75.90%|44.10%|11.93%|36.05%|38.58%|31.80pp|

|scene|before old|after old|after floor|seen-new|H|forgetting|
|---|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|77.80%|50.10%|14.36%|32.59%|38.39%|27.70pp|
|leo_low_elev_weak|72.91%|45.80%|13.88%|28.41%|34.01%|27.11pp|
|leo_rain_weak|73.59%|45.88%|12.44%|29.32%|34.91%|27.71pp|

最弱receiver为3-19，H=23.69%；所有receiver和scene均表现为after floor与new同时偏低，不是单一场景异常。

## 与D62、D91、D92、SVRN全面同口径比较

|方法|证据范围|before old|after old|before floor|after floor|seen-new|H|forgetting|post correct|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D92|完整125|81.55%|65.56%|59.88%|36.81%|58.93%|61.57%|15.99pp|91,270/157,500|当前完整125最强|
|D62|完整125|81.51%|64.39%|59.77%|35.15%|59.11%|61.09%|17.11pp|91,071/157,500|与D92接近|
|D107 M_JOINT|完整125|74.77%|47.26%|44.03%|17.87%|30.11%|35.95%|27.51pp|50,262/157,500|淘汰，不重跑|
|SVRN-qKNN-BCRR|完整125|73.10%|43.03%|45.17%|11.21%|23.46%|29.25%|30.07pp|40,938/157,500|弱于D107，但同样不可晋级|
|D91|15行development；receiver20-1、seed713101、K10/new5|92.78%|82.22%|73.33%|50.00%|84.67%|82.62%|10.56pp|未形成完整125|15行prediction与matched D62逐哈希一致，不能归因于D91或参与完整125排名|

D107 M_JOINT相对D92：after old低18.30pp、after floor低18.95pp、seen-new低28.83pp、H低25.61pp、forgetting高11.52pp、post少41,008个正确预测。相对D62：after old低17.13pp、after floor低17.28pp、seen-new低29.00pp、H低25.14pp、forgetting高10.40pp、post少40,809个正确预测。D107虽优于SVRN的after old、after floor、seen-new、H和forgetting，但仍远离项目目标，不能因“优于一个弱基线”保留该路线。

## 证据artifact

|artifact|SHA256|
|---|---|
|`prediction_manifest.json`|`51084f17e76376d58044f19bd61da69e153565657c6f2285774e15d8f09dbe5f`|
|`truth_catalog.json`|`cd78bef1317a396ef238fe5e6c3b6f90cb927393d6527bb6e88b772ad0f20ae8`|
|`score/score_manifest.json`|`d9b04fe97d4613decb851bdf7906b8ba54e33b9454532bc3598d872137058995`|
|`logs/predict.log`|`efb72abe2dd7095bd77c1d75fc2ba85ae1a45ce39d110f234631a38fcd6a3062`|
|`logs/score.log`|`1d8c16207c780c5aa0f4c9a3a693af8adad2164c7d386c7d1e756b75a2e48043`|
|`control/predict.exit`|`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`|

## 最终判定与下一方法

D107状态为`ANALYZED / PERFORMANCE_WEAK / REJECTED_NO_RERUN`。失败机制清楚：signed pre-ReLU原型本身使before old从D92的81.55%降到74.77%；support-centered RKHS变换继续损伤old floor；simplex KRR没有恢复new。D107不调参、不修arm、不创建r2。下一方法D108必须保留D92的强ReLU/diag-cov表示，把研发集中在support-only的注册平衡head与floor保护上。
