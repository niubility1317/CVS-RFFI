# D106外部基线证据与全面对比底稿

状态：`HISTORICAL_ARTIFACTS_AUDITED / D106_RESULT_PENDING / NO_CROSS_RUN_BEST_VALUE_SPLICE`

## 1.证据范围

本文只整理已完成的D62、D92、SVRN完整125矩阵和D91开发cell，为D106固定Target25提供外部matched基线。D62、D92、SVRN均覆盖5个receiver×5个seed×5个slice=125row，每row含3个互斥`leo_*_weak`场景；D91只覆盖development seed下K10/new5的3场景×5fold=15row，不能与125总体直接排序。

|方法|机制类别|证据规模|证据等级|最终状态|
|---|---|---:|---|---|
|D62 cross-fitted Fisher row splice|纯target-support交叉拟合Fisher/LDA行拼接|125row/375场景|完整历史125诊断|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|D91 crossfit consensus sigma margin|地面压缩sigma头＋跨折梯度共识收缩|15row development cell|开发诊断，不是125|未进入125|
|D92 registration-balanced covariance|注册后旧/新support分组协方差固定0.5/0.5融合|125row/375场景|完整历史125诊断|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|SVRN-qKNN-BCRR/r4.2|support-only SVRN分支＋INT8 qKNN＋BCRR，主臂`M_JOINT`|125row/375场景|完整历史125诊断|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|D106 RDCE/GTSM-r3|Phase1共享低秩DA＋待冻结support-only HEAD|Target25待运行|本地DATA/DA G0通过|无性能结果|

## 2.完整125总体

单位均为%；`B-old`为注册前旧类准确率，`A-old`为注册后旧类准确率，`F-old`为同row最低旧类准确率，`N`为seen-new准确率，`H`为旧/新调和均值，`Forget=B-old−A-old`。

|方法|row|B-old|A-old|F-old|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|---:|
|D62|125|81.51|64.39|35.15|59.11|61.09|17.11|
|SVRN-qKNN-BCRR/r4.2|125|73.10|43.03|11.21|23.46|29.25|30.07|

D92报告只把各slice与matched D81一起公开，不能把五个slice的独立数字重新拼成一个未经原artifact验证的“125总体”。因此D92在主比较中保留逐slice同row表。

## 3.逐slice同row性能

### 3.1 D62

|slice|B-old|A-old|F-old|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|86.02|76.33|50.47|73.57|74.60|9.69|
|K10/new10|86.02|71.53|42.27|66.75|68.84|14.49|
|K10/new20|86.02|68.68|37.93|68.78|68.56|17.34|
|K5/new20|81.32|61.39|30.87|59.28|60.03|19.93|
|K1/new20|68.14|44.03|14.20|27.15|33.41|24.11|

### 3.2 D92

|slice|B-old|A-old|F-old|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|86.111|76.189|49.800|74.133|74.803|9.922|
|K10/new10|86.111|72.533|44.200|66.353|69.106|13.578|
|K10/new20|86.111|71.333|42.667|68.150|69.555|14.778|
|K5/new20|81.267|63.711|33.200|58.883|60.955|17.556|
|K1/new20|68.144|44.033|14.200|27.150|33.410|24.111|

### 3.3 SVRN-qKNN-BCRR/r4.2

|slice|B-old|A-old|F-old|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|75.36|52.26|15.53|41.77|45.28|23.10|
|K10/new10|75.36|46.13|11.67|28.15|34.36|29.22|
|K10/new20|75.36|42.78|9.13|17.30|24.40|32.58|
|K5/new20|73.37|41.59|10.80|15.42|22.15|31.78|
|K1/new20|66.08|32.41|8.93|14.67|20.07|33.67|

## 4.严格paired结论

### 4.1 D62相对D81

125/125row按receiver、seed、K、new-count严格匹配。D62−D81的`A-old=-0.0067pp`、`F-old=-0.0533pp`、`N=-0.0040pp`、`H=-0.0028pp`，全部95%CI跨0。注册后375个场景状态中仅24个真正激活D62，K10/new20和K5/new20各只有1/75场景保留D62行；K1为75/75精确回退。因此D62的主要问题是formal decision surface几乎没有生效，不是均值波动不足。

### 4.2 D92相对D81

|slice|ΔA-old|ΔF-old|ΔN|ΔH|ΔForget|
|---|---:|---:|---:|---:|---:|
|K1/new20|0.000|0.000|0.000|0.000|0.000|
|K5/new20|+2.311|+2.400|−0.410|+0.920|−2.311|
|K10/new5|−0.133|−0.867|+0.520|+0.197|+0.133|
|K10/new10|+1.000|+1.933|−0.340|+0.291|−1.000|
|K10/new20|+2.622|+4.600|−0.653|+0.964|−2.622|

D92在大注册规模下确实改善旧类和floor，但每个旧类正向slice都牺牲新类；K1逐值不变。它是“有机制作用但任务交换伤害”的阴性基线。

### 4.3 SVRN相对D62

125/125row严格匹配。SVRN−D62为：`B-old=-8.40pp`、`A-old=-21.36pp`、`F-old=-23.93pp`、`N=-35.64pp`、`H=-31.84pp`、`Forget=+12.96pp`；`N/H`在125/125row均更差。SVRN的完整125已回答稳定性问题，无理由重复运行同一r4.2矩阵。

### 4.4 D91开发cell

D91在15个K10/new5 development row上为`B-old=92.778%`、`A-old=82.222%`、`F-old=53.333%`、`N=84.667%`、`H=82.624%`、`Forget=10.556pp`。相对D89，`A-old=-0.556pp`、`H=-0.313pp`、`Forget=+0.556pp`，仅1/15row改变；相对同cell D62则15/15row全部逐值一致。D91没有125证据，不能用其较高development绝对值压过D62/D92/SVRN正式矩阵。

## 5.相对当前D106门的差距

当前门为K10`A-old≥92%`、`F-old≥85%`，且new5/new10/new20的`N≥92%/90%/86%`。历史最强的D92在K10/new20仍只有`A-old=71.333%`、`F-old=42.667%`、`N=68.150%`，分别差20.667pp、42.333pp和17.850pp。D62与D92的K5/new20相对matched K10/new20在`A-old/F-old/N/H`上都下降超过5pp；两者K1也都没有正向DA功能。

因此D106不能把“略优于某个历史均值”当作成功。它必须先证明：

1. DA在同一四臂row中形成可测、方向一致的简单效应；
2. HEAD改善`N/H/F-old`而不把旧类收益换成新类退化；
3. K1相对同row D92同时满足`ΔH≥2pp`、`ΔF-old≥2pp`、`ΔA-old≥0`、`ΔN≥0`和总正确数严格增加；
4. 完整Target25后再与D62、D92、SVRN按相同receiver、seed、K、new-count配对，D91只保留机制级开发参照。

## 6.资源与部署含义

|方法|适配资源|持久状态|query侧|含义|
|---|---|---:|---|---|
|D62|2,016参数、20epoch/step；估算0.107B–42.152B MAC/job，均值26.056B|8,583–18,503B|6,624–15,264MAC/query；均值0.00857ms|状态小、query轻，但support闭式交叉拟合很重|
|SVRN-r4.2|0训练参数、0epoch、0optimizer step|峰值208,069B，均值126,981B|评分均值0.07868ms|无需梯度训练，但性能显著退化|
|D106 RDCE DA|0query fit；rank3低秩变换|INT8 basis＋FP16 attenuation|960MAC/query，basis每评分上下文解码1次|当前仅为本地功能与资源闭包，无性能结论|

## 7.证据文件

- `automation_reports/CV-SincNet/d62_comprehensive_125_20260720/report.md`
- `automation_reports/CV-SincNet/d91_crossfit_consensus_sigma_margin_20260720/d91_full_performance_summary.json`
- `code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_registration_balanced_125_20260720/report.md`
- `automation_reports/CV-SincNet/svrn_qknn_bcrr_125_r4_retry2_20260724/report.md`

以上历史数字不替代D106的四臂同row预测、独立truth-side评分、逐类下尾和paired CI。
