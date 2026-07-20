# D95窄实验Runner交接

## 执行与回收状态

|row|候选|PID/GPU|最终状态|性能资格|
|---|---|---|---|---|
|K10/new20|`d95_d81_coverage_residual`|`1173317`/GPU0|进程已退出；D43结构化协方差非正定，fail-closed|无pipeline receipt、fit/resource audit、预测或score，不存在性能结果|
|K1/new20|`d95_d81_coverage_residual`|`1173318`/GPU1|`DEVELOPMENT_ROW_COMPLETE`|仅development diagnostic；`formal_launch_authority=false`|

监控遵循N607短连接规则：直连preflight通过，两个PID均已退出，GPU0—GPU7均回到10MiB；每次SSH/SCP后本地`ssh.exe=0`，到N607及bridge的TCP22已建立连接数为0。未kill、重启、修改远端代码或干预其他任务。

预登记晋级门来自主报告：K10相对matched D81要求`H`和最低旧类同时提高且New下降不超过1pp；K1不得降低old或floor；任一失败均不得进入125。K1已明确失败，K10又无可评分结果，因此D95整体不得晋级125。

## 完整日志与异常扫描

|row|stdout|行数/字节|异常扫描|
|---|---|---:|---|
|K10/new20|`artifacts/d95_k10/d95_k10_new20.stdout.log`|32/3,603|1条既有PyTorch只读buffer warning；1个Traceback；末行`D43 structured covariance is not positive definite`；未命中OOM、Killed、NaN或Inf|
|K1/new20|`artifacts/d95_k1/d95_k1_new20.stdout.log`|3/1,376|1条既有PyTorch只读buffer warning；无Traceback、OOM、Killed、NaN或Inf；最终marker为`DEVELOPMENT_ROW_COMPLETE`|

K10失败发生在D95变换后的D81/D62内部D43 block covariance拟合，尚未生成任何query预测。按交接约束未重启；不能把K10技术失败与K1性能拼接，也不能从partial offline build推导性能。

## K1同row联合性能

单位均为百分比或百分点。

|方法|B-old|A-old|Min-B|Min-A|New|H|F|
|---|---:|---:|---:|---:|---:|---:|---:|
|matched D81|61.667|37.500|33.333|13.333|27.583|31.786|24.167|
|D95|56.389|33.333|23.333|8.333|28.167|30.533|23.056|
|D95−D81|-5.278|-4.167|-10.000|-5.000|+0.583|-1.253|-1.111|

遗忘数值虽少1.111pp，但D95注册前旧类已经低5.278pp、注册后仍低4.167pp，因此不是更好的旧类保持能力。K1的old和floor均违反预登记非退化门。

### 三场景

|场景|B-old|A-old|New|H|D95−D81 A|D95−D81 New|D95−D81 H|
|---|---:|---:|---:|---:|---:|---:|---:|
|clear|59.167|32.500|37.000|34.604|-8.333|+1.000|-3.660|
|low-elev|46.667|30.000|20.000|24.000|-2.500|+0.500|-0.375|
|rain|63.333|37.500|27.500|31.731|-1.667|+0.250|-0.409|

三个场景的注册后旧类和H均低于matched D81；clear旧类退化最大。

### 逐旧类

|旧类|D95 A|D81 A|差值|
|---|---:|---:|---:|
|`14-10`|40.000|58.333|-18.333|
|`14-7`|8.333|13.333|-5.000|
|`20-15`|48.333|45.000|+3.333|
|`20-19`|13.333|15.000|-1.667|
|`6-15`|16.667|18.333|-1.667|
|`8-20`|73.333|75.000|-1.667|

只有`20-15`改善；`14-10`下降18.333pp导致主要旧类退化，最低旧类`14-7`仅8.333%。

### 逐新类

|新类|D95|D81|差值|
|---|---:|---:|---:|
|`1-16`|23.333|23.333|0.000|
|`1-18`|23.333|20.000|+3.333|
|`1-8`|35.000|35.000|0.000|
|`10-10`|20.000|21.667|-1.667|
|`11-19`|25.000|21.667|+3.333|
|`13-14`|8.333|6.667|+1.667|
|`14-11`|25.000|28.333|-3.333|
|`16-19`|13.333|13.333|0.000|
|`18-10`|11.667|11.667|0.000|
|`18-8`|50.000|48.333|+1.667|
|`19-13`|23.333|23.333|0.000|
|`19-6`|50.000|51.667|-1.667|
|`19-8`|48.333|46.667|+1.667|
|`19-9`|18.333|16.667|+1.667|
|`2-16`|16.667|18.333|-1.667|
|`2-5`|38.333|35.000|+3.333|
|`20-12`|16.667|16.667|0.000|
|`3-8`|23.333|21.667|+1.667|
|`4-10`|35.000|33.333|+1.667|
|`8-3`|58.333|58.333|0.000|

新类总体只提高0.583pp，且逐类有升有降，无法抵消旧类和H的系统性下降。

## D81-base、coverage与算子审计

before/after各有3条场景fit audit，内容SHA逐位相同；每条均记录`d95_d81_base_used=true`，证明没有误跑成D94。K1下D81 Cauchy中心分支保持其锁定identity，`d95_before_center_shift_l2_max=d95_after_center_shift_l2_max=0`；D95自身ground→target residual算子则是非identity。

|场景顺序|ground nuisance coverage `rho`|span外能量比|operator更新谱范数|条件数|
|---:|---:|---:|---:|---:|
|1|0.203977|0.796023|0.101988|1.112812|
|2|0.104631|0.895369|0.052315|1.054734|
|3|0.136116|0.863884|0.066815|1.072100|

算子数值稳定且确实非identity，但79.60%—89.54%的target shift能量落在ground nuisance span之外，低coverage残差没有恢复D81判别结构。

## 量化、协议与资源

|项目|K1实测|判定|
|---|---:|---|
|地面聚合cell输入|84|真实读取|
|ground更新访问|false|PASS|
|ground逻辑状态|25,428B|只读|
|D81有效ground component fit|0|K1锁定identity；不是D95算子identity|
|support中心变换调用|18|完整|
|INT8/FP32 before support argmax变化|0/3场景|PASS|
|INT8/FP32 final support argmax变化|0/3场景|PASS|
|formal target FP32 sidecar|0|PASS|
|trainable parameters|2,260|低于80,000|
|adaptation epochs/steps|20/20|低于30/50|
|单场景估计适配MAC|53,706,240|记录完整|
|估计总MAC/query|21,344|其中D95 transport新增6,080|
|pipeline状态峰值|44,419B|低于256KiB|
|单fit final state|18,991B|INT8正式面|
|score matrix延迟|0.01234ms/query|K1本行实测|
|dense query graph|0B|PASS|
|query用于fit/transport|0/0|PASS|
|support/query view count|1/1|单物理IQ单观测|
|query backbone forwards/sample|1|PASS|

pipeline receipt还确认：query真值只在before/after不可变预测完成后连接，未回流predictor；候选claim scope为development only，地面组件当前并无formal launch authority。

## 回收artifact与SHA256

### K1完整成功面

|artifact|SHA256|
|---|---|
|`artifacts/d95_k1/pipeline_receipt.json`|`ab385cf8164e19792a93571db3f6293acacbfdb2b4cf37925187f3fe45321468`|
|`artifacts/d95_k1/diag_cosine_score.json`|`380eaaa7d9a6c668f1be1e7cf8d698787f8eb6e9494caf755e9612609849fae8`|
|`artifacts/d95_k1/diag/before/fit_audit.json`|`b84c2d0b93d54106c4e9bc54875acd44a4d120229002f6908cd164924664e714`|
|`artifacts/d95_k1/diag/after/fit_audit.json`|`b84c2d0b93d54106c4e9bc54875acd44a4d120229002f6908cd164924664e714`|
|`artifacts/d95_k1/diag/before/resource_audit.json`|`b7a306cfb477b96f600a324d40315b7c10e134a499f24b4da6a696df1a8b01f4`|
|`artifacts/d95_k1/diag/after/resource_audit.json`|`b7a306cfb477b96f600a324d40315b7c10e134a499f24b4da6a696df1a8b01f4`|
|`artifacts/d95_k1/diag/before/execution_receipt.json`|`8935ff54498181683da7d73d675695141ed95eb7e29c6ccdaa14503a97370cf7`|
|`artifacts/d95_k1/diag/after/execution_receipt.json`|`ec77579a4ed042aec53e8b45ea49d94e02a33d451355ebda6d1ef095fcfea8e9`|
|`artifacts/d95_k1/d95_k1_new20.stdout.log`|`2ddbd19699058530348be3bc373c272830e80bc78edda3db3b7e2c4e5a564fbf`|

before/after不可变prediction artifact及COMMIT也已完整回收到`artifacts/d95_k1/diag/`，未只保留汇总JSON。

### K10技术失败面

|artifact|SHA256|
|---|---|
|`artifacts/d95_k10/d95_k10_new20.stdout.log`|`a63c5a2ad85651433185373575f3a29c2de75ea7f7acf29460cadf6dda469a74`|
|`artifacts/d95_k10/offline_build_receipt.json`|`9f9f1a574be3340e7657781dc008625f92e733fe484f8958204bf885d1b5d1a5`|
|`artifacts/d95_k10/registration_pair.final.json`|`4210c4bb9d5e14c25973655123bcb629ec7064171e1d44c4be385d9313ee7635`|

K10没有`pipeline_receipt.json`、before/after fit/resource audit、COMMIT、prediction artifact或`diag_cosine_score.json`；这些不是回收遗漏，而是远端失败前未生成。

## 最终判定

`D95_COMPLETED_K1_DIAGNOSTIC_NEGATIVE_K10_TECHNICAL_FAILURE_NOT_PROMOTABLE`。

D95证明了“保留完整D81为base、叠加coverage-controlled ground→target非正交残差”能够在K1真正产生非identity算子，并保持量化、资源和query隔离闭包；但它以旧类、最低类和H退化换取不足1pp的新类增益。K10又在任何query预测前因结构化协方差数值失败停止。D95不得运行125，也不能称为性能更强版本。
