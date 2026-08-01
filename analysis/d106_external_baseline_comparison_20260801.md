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
|D106 RDCE＋RCMR-2V-qKNN|Phase1共享低秩DA＋纯target-support双视图秩拥挤头|Target25待运行|DATA/DA本地G0通过；HEAD真实588条production G0待闭合|无性能结果|

## 2.完整125总体

单位均为%；`B-old`为注册前旧类准确率，`A-old`为注册后旧类准确率，`B/A row-floor`为每个job内旧类floor再跨job平均，`N`为seen-new准确率，`H`为旧/新调和均值，`Forget=B-old−A-old`。这里的`row-floor`不是当前G2在固定seed、固定slice内先跨5receiver×3scenario聚合每个旧类再取最小值的`pooled F_old`，两者不得替换。

|方法|row|B-old|A-old|B row-floor|A row-floor|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D62|125|81.507|64.393|59.773|35.147|59.107|61.089|17.113|
|D92|125|81.549|65.560|59.880|36.813|58.934|61.566|15.989|
|SVRN-qKNN-BCRR/r4.2|125|73.102|43.033|45.173|11.213|23.463|29.251|30.069|
|D91|15个development row|92.778|82.222|73.333|50.000|84.667|82.624|10.556|

D92总体由`retry2/row_metrics.csv`的125个唯一键直接重算，不是从五个边际最优值拼接。D91另有跨15个development row聚合后的旧类class-floor`53.333%`；它与表中的mean row-floor`50.000%`口径不同，只作开发诊断。

## 3.逐slice同row性能

### 3.1 D62

|slice|B-old|A-old|A row-floor|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|86.02|76.33|50.47|73.57|74.60|9.69|
|K10/new10|86.02|71.53|42.27|66.75|68.84|14.49|
|K10/new20|86.02|68.68|37.93|68.78|68.56|17.34|
|K5/new20|81.32|61.39|30.87|59.28|60.03|19.93|
|K1/new20|68.14|44.03|14.20|27.15|33.41|24.11|

### 3.2 D92

|slice|B-old|A-old|A row-floor|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|86.111|76.189|49.800|74.133|74.803|9.922|
|K10/new10|86.111|72.533|44.200|66.353|69.106|13.578|
|K10/new20|86.111|71.333|42.667|68.150|69.555|14.778|
|K5/new20|81.267|63.711|33.200|58.883|60.955|17.556|
|K1/new20|68.144|44.033|14.200|27.150|33.410|24.111|

### 3.3 SVRN-qKNN-BCRR/r4.2

|slice|B-old|A-old|A row-floor|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|75.36|52.26|15.53|41.77|45.28|23.10|
|K10/new10|75.36|46.13|11.67|28.15|34.36|29.22|
|K10/new20|75.36|42.78|9.13|17.30|24.40|32.58|
|K5/new20|73.37|41.59|10.80|15.42|22.15|31.78|
|K1/new20|66.08|32.41|8.93|14.67|20.07|33.67|

## 4.严格paired结论

### 4.1 D62相对D81

125/125row按receiver、seed、K、new-count严格匹配。D62−D81为：`A-old=-0.0067pp[-0.0556,+0.0423]`、`A row-floor=-0.0533pp[-0.1881,+0.0814]`、`N=-0.0040pp[-0.0377,+0.0297]`、`H=-0.0028pp[-0.0287,+0.0232]`、`Forget=-0.0356pp[-0.1267,+0.0556]`，95%CI全部跨0。注册后375个场景状态中仅24个真正激活D62，K10/new20和K5/new20各只有1/75场景保留D62行；K1为75/75精确回退。因此D62的主要问题是formal decision surface几乎没有生效，不是均值波动不足。

### 4.2 D92相对D81

|slice|ΔA-old|ΔA row-floor|ΔN|ΔH|ΔForget|
|---|---:|---:|---:|---:|---:|
|K1/new20|0.000|0.000|0.000|0.000|0.000|
|K5/new20|+2.311|+2.400|−0.410|+0.920|−2.311|
|K10/new5|−0.133|−0.867|+0.520|+0.197|+0.133|
|K10/new10|+1.000|+1.933|−0.340|+0.291|−1.000|
|K10/new20|+2.622|+4.600|−0.653|+0.964|−2.622|

D92总体相对D81为：`A-old=+1.160pp[+0.893,+1.427]`、`A row-floor=+1.613pp[+1.032,+2.195]`、`N=-0.1767pp[-0.3243,-0.0290]`、`H=+0.4743pp[+0.3514,+0.5971]`、`Forget=-1.160pp[-1.427,-0.893]`。它在大注册规模下确实改善旧类和row-floor，但每个旧类正向slice都牺牲新类；K1逐值不变。因此它是“有机制作用但任务交换伤害”的阴性基线，而不是所有指标均最强的基线；K10/new20的D62`N=68.780%`仍高于D92的`68.150%`。

### 4.3 SVRN相对D62

125/125row严格匹配。SVRN−D62为：`A-old=-21.360pp[-23.184,-19.536]`、`A row-floor=-23.933pp[-27.135,-20.731]`、`N=-35.643pp[-38.410,-32.877]`、`H=-31.838pp[-33.937,-29.740]`、`Forget=+12.956pp[+11.850,+14.061]`；`N/H`为0/125row更好。SVRN的完整125已回答稳定性问题，无理由重复运行同一r4.2矩阵。

### 4.4 D91开发cell

D91在15个K10/new5 development row上为`B-old=92.778%`、`A-old=82.222%`、`A row-floor=50.000%`、聚合旧类class-floor`53.333%`、`N=84.667%`、`H=82.624%`、`Forget=10.556pp`。相对D89，`A-old=-0.556pp[-1.747,+0.636]`、`A row-floor=0`、`N=0`、`H=-0.313pp[-0.984,+0.358]`、`Forget=+0.556pp[-0.636,+1.747]`，仅1/15row改变；相对同cell D62则15/15row全部逐值一致。D91没有125证据，不能用其较高development绝对值压过D62/D92/SVRN正式矩阵。

## 5.相对当前D106门的差距

当前门为K10`A-old≥92%`、`pooled F-old≥85%`，且new5/new10/new20的`N≥92%/90%/86%`。不能用五seed逐row平均表直接计算这一单seed门。对固定seed713102、固定slice按当前定义重算：

|方法|slice|A-old|pooled F-old|N|H|
|---|---|---:|---:|---:|---:|
|D62|K10/new5|76.111|59.000|76.000|75.803|
|D62|K10/new10|72.333|51.667|67.067|69.336|
|D62|K10/new20|69.833|49.333|69.167|69.271|
|D92|K10/new5|75.778|58.000|76.267|75.751|
|D92|K10/new10|73.278|52.000|66.433|69.446|
|D92|K10/new20|72.167|51.667|68.633|70.160|
|SVRN|K10/new5|52.056|21.667|42.600|45.530|
|SVRN|K10/new10|44.111|15.667|30.167|35.111|
|SVRN|K10/new20|40.722|13.000|18.583|25.105|

就K10/new20而言，D92距当前门仍差`A-old=19.833pp`、`pooled F-old=33.333pp`、`N=17.367pp`。D92在该slice的旧类与H联合表现最好，但D62的新类准确率更高，不能笼统称D92所有指标最强。

K5/new20相对同seed、同方法K10/new20的下降为：

|方法|A-old下降|pooled F-old下降|N下降|H下降|结论|
|---|---:|---:|---:|---:|---|
|D62|6.056|8.000|8.000|7.091|四项均失败|
|D92|6.000|6.000|7.567|6.830|四项均失败|
|SVRN|0.000|2.000|2.183|2.110|退化门通过，但K10绝对门严重失败|

D92在seed713102的K1/new20基线为`A-old=44.889%`、`pooled F-old=20.667%`、`N=26.267%`、`H=32.596%`、old+new正确数`2384`。因此D106的汇总必要条件至少为`A-old≥44.889%`、`F-old≥22.667%`、`N≥26.267%`、`H≥34.596%`且正确数≥2385；正式通过仍须逐row配对，不能只比较这些汇总阈值。

因此D106不能把“略优于某个历史均值”当作成功。它必须先证明：

1. DA在同一四臂row中形成可测、方向一致的简单效应；
2. HEAD改善`N/H/pooled F-old`而不把旧类收益换成新类退化；
3. K1相对同row D92同时满足`ΔH≥2pp`、`Δpooled F-old≥2pp`、`ΔA-old≥0`、`ΔN≥0`和总正确数严格增加；
4. 完整Target25后再与D62、D92、SVRN按相同receiver、seed、K、new-count配对，D91只保留机制级开发参照。

## 6.资源与部署含义

|方法|适配资源|持久状态|query侧|证据口径|
|---|---|---:|---|---|
|D62|2,016参数；20epoch/20step；0.107–42.152B MAC/job，均值26.056B|8,583–18,503B，均值15,194B|6,624–15,264MAC/query；均值0.00857ms|125row资源审计|
|D91|2,159参数；20epoch；40optimizer step；估算25.427B MAC；实际16次crossfit LDA约0.499B，另有386,672 consensus MAC|14,399B；另有116,304B瞬时反量化ground|6,624MAC/query；D91额外query MAC=0|15row development资源闭包|
|D92|K10为88次闭式分量fit；new5/new10/new20约11.153/11.346/11.741GMAC等价上界|7.46/10.34/16.11KiB核心数组，另加元数据|编译头3,168/4,608/7,488MAC/query；保守流水线6,624/9,504/15,264|公式化保守审计，不是硬件实测|
|SVRN-r4.2|0参数/0epoch/0step|峰值208,069B，均值126,981B|均值0.07868ms，最大0.09543ms；未闭合统一MAC|125row资源审计|
|D106 RDCE DA|rank3；0query-fit|INT8 basis＋FP16 attenuation；数值payload<1KiB|仅DA为960MAC/query，不含HEAD、主干和完整方法|本地G0资源闭包，无性能结果|

不同方法的资源证据并非同一硬件实测口径，不能据此直接给出速度排名。

## 7.证据文件

- `automation_reports/CV-SincNet/d62_comprehensive_125_20260720/report.md`
- `automation_reports/CV-SincNet/d91_crossfit_consensus_sigma_margin_20260720/d91_full_performance_summary.json`
- `code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_registration_balanced_125_20260720/report.md`
- `automation_reports/CV-SincNet/svrn_qknn_bcrr_125_r4_retry2_20260724/report.md`

以上历史数字不替代D106的四臂同row预测、独立truth-side评分、逐类下尾和paired CI。
