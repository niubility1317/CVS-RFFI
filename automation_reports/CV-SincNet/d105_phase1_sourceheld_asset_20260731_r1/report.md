# D105 Phase1 source-only压缩知识与source-held门实验报告

状态：`R2_LOCAL_VERIFIED / RELEASE_REVIEW_PENDING / NOT_LANDED / NO_PERFORMANCE_RESULT`

## 1.实验标识与目标

|字段|值|
|---|---|
|experiment ID|`d105_phase1_sourceheld_asset_20260731_r1`|
|日期|2026-07-31|
|operator|主agent整合；terra-max Phase1功能agent实现；正式N607由另设唯一terra-max runner执行|
|协议边界|仅Phase1 source-only；`target_rows=0`、`query_rows=0`|
|目标|从真实ADV3B02 checkpoint和经验证的source weak-IQ生成D105严格`pre_relu/z_dom`tap，完成receiver-held K1/K5/K10、class-LOCO、TX probe与INT8量化门，生成class-free/receiver-free聚合bundle并独立封存|
|性能边界|本run只决定Phase1资产是否具备formal Phase2输入资格，不产生Target性能结论|
|GitHub|不push、不上传；仅本地Git版本化|

## 2.核心假设

D105共享DA所需的Phase1知识可以压缩为不含类名、receiver名、physical ID、原始IQ或逐样本特征的INT8/FP16聚合状态。该状态只有在完整source-held非劣门、低TX可预测性、量化一致性、独立复审与外部authority seal全部闭合后，才允许作为Phase2不可变输入。

D102的`PHASE1_HELD_FALSIFIER_REJECT` bundle、held结论和其formal=false资产一律禁止复用。可重用的只有独立SHA绑定的原始source cache、selection salt、真实checkpoint和通用读取实现；D105必须重新产生自己的strict tap、预测封存、truth-open、独立score、gate、component与seal。

## 3.预登记输入

|输入|N607路径|SHA256/状态|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|source validation cache set|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|selection salt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|dual archive parity reference|`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8/output/archive/phase1_singleobs_dual_feature_archive.npz`|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；仅作字节一致性参考，不直接转封为D105资产|

运行环境拟定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，GPU0执行严格tap和source-held预测。正式release前由唯一runner再次执行只读preflight并记录即时GPU占用。

## 4.不可变生命周期

```text
source weak-IQ
  -> D105 strict tap archive/receipt（无Target、无query）
  -> source-held prediction manifest（无truth，只读）
  -> truth-open receipt（绑定prediction manifest SHA）
  -> independent score artifact
  -> gate recomputation
  -> non-formal aggregate component
  -> independent review P0=0/P1=0
  -> external authority seal
  -> formal D105 Phase1 asset/runtime handle
```

缺少任一步、文件可写、哈希漂移、D102 REJECT lineage、truth先于prediction、held覆盖不完整或任一门失败时，必须保持`formal_phase2_eligible=false`，不得供Target25加载。

## 5.冻结source-held覆盖与门

- 每个source receiver均覆盖K1、K5、K10 receiver-held预测；
- 每个source receiver×class覆盖class-LOCO；
- receiver-held和class-LOCO均要求D105相对M0的balanced accuracy、最低类准确率和净正确数不劣；
- 每个receiver完成TX probe，最大balanced accuracy≤25%；
- INT8相对FP32 top1一致率≥99.5%，大margin flip为0；
- 不持久化原始IQ、clean IQ、逐样本FP32 sidecar、source replay或类/receiver/physical ID；
- 独立代码复审要求`P0=0、P1=0`。

这些门只授予Phase1资产资格，不等于D105四臂Target性能或`PROMOTABLE`。

## 6.拟定N607输出

|类型|路径|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_asset_20260731_r1`|
|source snapshot|`<run-root>/source`|
|input receipts|`<run-root>/input`|
|strict tap|`<run-root>/output/strict_tap`|
|prediction manifest|`<run-root>/output/source_held_prediction`|
|truth-open与score|`<run-root>/output/source_held_score`|
|component|`<run-root>/output/component`|
|formal sealed asset|`<run-root>/output/formal_asset`|
|main log/PID/exit|`<run-root>/logs/pipeline.log`、`pipeline.pid`、`pipeline.exit`|

精确服务器命令、Git commit、源码哈希、method lock/runtime SHA、authority receipt和expected artifact SHA将在本地实现、全组测试与独立release复审完成后补写。在此之前不得同步或启动。

## 7.健康停止与判定

性能高低不得触发提前停止。仅P0协议/安全错误、输出覆盖风险、错误checkout/hash、truth顺序违规或至少两个不同held row在预测前出现同一标准化确定性异常指纹时停止本run。停止后保留全部partial artifact，并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

技术完成要求：strict tap、prediction manifest、truth-open、score、gate、component、authority seal和formal validation全部同哈希闭合，完整日志可读，退出码0。若source-held性能门失败但技术闭合，则状态为`COMPLETED_SOURCE_HELD_REJECT / NO_TARGET_LAUNCH`，不得生成formal seal或启动Target25。

## 8.当前进展

- 已完成真实checkpoint随机IQ feature-tap smoke，checkpoint字节和`pre_relu/z_dom`形状闭合；
- 已完成strict source tap→无truth只读prediction→truth-open→独立score→重算gate→aggregate component闭环；
- R1独立release审查发现unsigned authority自述可将component升级为formal，判为`P0=1`并`NO-GO`；R1未提交、未落地；
- R3稳定版使用固定`somph_runtime_trust`Ed25519公钥和D105专用signature domain；formal asset完整保留authority envelope、detached signature、独立review receipt、签名D102 revocation manifest及signature；
- formal authority签名绑定N607预先创建的nonce ledger identity；该identity由N607账本绝对路径、run ID和签名域规范化产生，离线签名端只接收摘要，封存端必须在消费nonce前以本机路径重算一致；
- D102r6的bundle manifest、payload、seal、content root、method lock、runtime、held score和tap archive真实SHA均进入内容撤销项，改名副本不能绕过；
- candidate runtime manifest现绑定正式执行面递归可达的40个`cvsrffi`模块＋5个正式CLI，共45文件；SHA256=`639c16dd6a70620ca99fa960acb9e988aeba3cea92edcb7a9a158b26a6d958b5`；
- candidate method lock显式固定4轮IRLS、任务等质量、K1零系数、FP16部署和Target25开发声明；SHA256=`37dd03fcdb7cb01e6e545def11711b0c9c9ad35e3d505d75c18f314cb3ef3576`；
- `ssr-gpu`统一回归182项全部通过；45文件`py_compile`、5个正式CLI及4个关键子命令参数面、canonical loader和`git diff --check`通过；
- 同代真实checkpoint无query smoke收据SHA256=`cc08c4891b8c9112fc37dc9c752f7f53f99e4a3b83df22195f3f58e48696ef5f`；400个source-held meta step完成，K1恒等成立，query fit/update=0/0，Target访问=false，性能计算=false，stderr=0；
- 2026-07-31 14:24 HKT只读N607 preflight通过，8张RTX 3090均空闲；盘点结束后本地无残留SSH进程或到N607/bridge的ESTABLISHED连接；
- 上述GPU状态仅是历史只读盘点，正式release前必须重新preflight；
- 当前尚无N607真实D105 strict tap、source-held score、formal asset或性能数据；R2必须先达到独立`P0=0、P1=0`并完成本地Git提交。

## 9.R2发布门

|门|当前状态|
|---|---|
|统一代码/负测|182 passed|
|正式执行闭包|40个递归模块＋5个CLI；45文件`py_compile`通过；缺失/漂移逐项失败|
|真实checkpoint无query smoke|PASS；收据`cc08c489…96ef5f`；仅技术证据|
|candidate runtime/method lock|`639c16dd…d958b5`/`37dd03fc…ef3576`；canonical loader通过|
|可信外部authority签名|代码闭合；尚未生成生产signature|
|签名D102 revocation|代码和真实identity fixture闭合；生产signature待独立authority|
|R3独立release复审|稳定代码实证完成；待文档/receipt/commit闭合后最终签字|
|本地Git提交|待R3最终复审通过后执行|
|N607 landing|未执行|

生产私钥不得进入Git、报告、N607或formal asset。若无法获得与固定公钥匹配的独立签名，必须保持`NO_TARGET_LAUNCH`，不能退回unsigned JSON。

45文件闭包限定于正式Phase1/Target25执行面及5个入口。real-checkpoint smoke脚本本体在闭包内，但其额外训练helper不是正式Target25预测依赖，不能把45文件解释为覆盖全部smoke传递依赖。
