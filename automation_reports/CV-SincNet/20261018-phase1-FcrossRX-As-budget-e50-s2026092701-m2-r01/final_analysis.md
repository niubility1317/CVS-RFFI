# E50 匹配预算：单种子正式结果

**已完成训练、七视图预测、独立 truth-last 评分和完整分析。** seed2026092701 的原完整 As50 相对匹配 NoAs50：clean +0.365476 pp、mid urban +29.864881 pp、low urban +22.508929 pp，urban 三场均值 +30.707738 pp。本种子满足 clean 下降≤0.1 个绝对百分点；clean 净修复614包，不存在 clean 下降代价。

相对 As 自己的 E30，clean／mid／low／urban 均值分别 +0.319643／+1.095238／+1.143452／+1.071627 pp。NoAs 的同等预算变化为 +0.047619／−1.095833／−0.261310／−1.267659 pp。因此 E50 的巨大 As−NoAs 星地差值既包含原 As 的长期收益，也包含本次 NoAs 星地下滑，不能全部归因于新增20轮。预算交互 clean +0.272024 pp、mid +2.191071 pp、low +1.404762 pp。

源证据独立支持保留这个固定预算假设：full V27000 As50−NoAs50 clean −0.074074 pp；As50 相对 As30 的 full V clean +0.118519 pp，源面板两 draw 的 mid／low +1.972222／+1.638889 pp，两场均值 +1.805556 pp。源 mid／low 预算交互 +1.472222 pp。全 V 原生指标与 V1800 panel 分开，源 panel 的 clean 配对变化 −0.166667 pp，不用大样本指标遮盖该小 panel 结果；该面板并非独立模型种子。

**仅1个模型种子，尚未证明可重复双提升。** 原E30三种子 clean 均值−0.3210 pp 的历史负结果保持；不会用01的E50结果改写其他种子。当前run固定 automatic_seed_extension=false，未启动额外种子、未按正式结果挑参数或选择性重跑。下一唯一待检验的命题是该固定预算收益能否在原定其他种子复现，不能从本次结果推定成立。

As50 的三个 suburban 场景与自身 clean 差距分别为 3.421429／6.104167／8.884524 pp；全部绝对准确率较 As30 上升，但 low suburban 最差 RX 比 As30 低0.116667 pp，不能声称所有分层都改善。low urban 绝对准确率52.387500%；没有把旧>60%探索目标写成已完成。

## 完整匹配读出

20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01；单种子2026092701。原As完整KL/dropout保留，F/defaultcos/joint160/NoExplicitCFO与原L6300/U56700隐藏/V27000不变。

E50是各自固定FINAL30 student/EMA追加20epoch，resetAdamW/cos20；不能把额外预算全部归因于As。阶段目标为同预算As−NoAs clean下降不超过0.1个绝对百分点，最终仍需可重复clean正收益。

|视图|NOAS30 %|AS30 %|NOAS50 %|AS50 %|As50−NoAs50 pp|As50−As30 pp|预算交互 pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.295238|85.660714|0.365476|0.319643|0.272024|
|practical_high|42.685119|81.405357|39.952381|82.239286|42.286905|0.833929|3.566667|
|practical_mid|40.995238|78.661310|39.140476|79.556548|40.416071|0.895238|2.750000|
|practical_low_suburban|39.472619|75.861905|38.218452|76.776190|38.557738|0.914286|2.168452|
|practical_high_urban|41.439881|77.767262|38.994048|78.743452|39.749405|0.976190|3.422024|
|practical_mid_urban|35.132738|62.806548|34.036905|63.901786|29.864881|1.095238|2.191071|
|practical_low_urban|30.139881|51.244048|29.878571|52.387500|22.508929|1.143452|1.404762|
|urban三场景均值|35.570833|63.939286|34.303175|65.010913|30.707738|1.071627|2.339286|

正式 E50 clean 配对变化 +0.365476 pp（提高，无下降代价），净修复614包；0.1pp整数判定net×1000≥−N：PASS。low/mid urban相对同预算NoAs分别+22.508929/+29.864881pp。

预算交互=(As50−NoAs50)−(As30−NoAs30)。NOAS50−NOAS30、AS50−AS30和同预算As−NoAs均保留，不混为机制效应。

|比较|视图|修复|新增错误|净修复|
|---|---|---:|---:|---:|
|AS50−NOAS50|clean|4153|3539|614|
|AS50−NOAS50|practical_mid_urban|56661|6488|50173|
|AS50−NOAS50|practical_low_urban|46642|8827|37815|
|AS30−NOAS30|clean|3868|3711|157|
|AS30−NOAS30|practical_mid_urban|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|44297|8842|35455|
|AS50−AS30|clean|2432|1895|537|
|AS50−AS30|practical_mid_urban|5920|4080|1840|
|AS50−AS30|practical_low_urban|6711|4790|1921|
|NOAS50−NOAS30|clean|2104|2024|80|
|NOAS50−NOAS30|practical_mid_urban|7490|9331|-1841|
|NOAS50−NOAS30|practical_low_urban|7810|8249|-439|

|模型|视图|macro-F1 %|最差RX %|
|---|---|---:|---:|
|NOAS30|clean|84.979636|78.450000|
|NOAS30|practical_high|39.311416|34.304167|
|NOAS30|practical_mid|37.954813|33.520833|
|NOAS30|practical_low_suburban|36.758731|32.054167|
|NOAS30|practical_high_urban|38.481030|34.000000|
|NOAS30|practical_mid_urban|33.446603|29.495833|
|NOAS30|practical_low_urban|29.184589|25.750000|
|AS30|clean|85.044474|78.362500|
|AS30|practical_high|81.084230|71.908333|
|AS30|practical_mid|78.359706|70.212500|
|AS30|practical_low_suburban|75.572741|67.741667|
|AS30|practical_high_urban|77.423208|69.108333|
|AS30|practical_mid_urban|62.450463|56.458333|
|AS30|practical_low_urban|50.887035|46.379167|
|NOAS50|clean|85.009461|77.479167|
|NOAS50|practical_high|34.167858|33.212500|
|NOAS50|practical_mid|34.087783|32.204167|
|NOAS50|practical_low_suburban|33.750550|31.241667|
|NOAS50|practical_high_urban|33.810253|32.766667|
|NOAS50|practical_mid_urban|30.802839|28.666667|
|NOAS50|practical_low_urban|27.842578|25.845833|
|AS50|clean|85.381047|79.025000|
|AS50|practical_high|82.031208|72.279167|
|AS50|practical_mid|79.342102|70.287500|
|AS50|practical_low_suburban|76.523952|67.625000|
|AS50|practical_high_urban|78.521008|69.708333|
|AS50|practical_mid_urban|63.536348|57.245833|
|AS50|practical_low_urban|51.848684|47.233333|

源完整V27000指标来自原生冻结读出，未保存逐包fullV预测，不宣称逐包独立重算。固定源V1800两draw的raw/三practical logits独立核对；无源highurban，不用mid/low冒称urban三场景均值。source面板fc2.45GHz、训练/正式2.462GHz既有差异保留披露。

源As50−NoAs50 fullV clean：-0.074074pp；源面板mid/low预算交互：+1.472222pp。source结果仅描述，不增加晋级门槛、不跳过正式收尾。

全4440新增step与20epoch CSV/JSONL、6L RDrop尺度、原U/teacher、物理序列、actual student/teacher counters均完整读取；旧30轮使用既有完整recount，不重复扫描旧step。source与formal技术缺项不得冒充完整。

|行|student train NN|student eval NN|teacher eval NN|native NN|训练段wall s|
|---|---:|---:|---:|---:|---:|
|NOAS30|6078780|918004|963900|7960684|2146.190|
|AS30|6078780|918004|963900|7960684|2860.909|
|NOAS50|4543920|594002|1134000|6271922|1602.126|
|AS50|4543920|594002|1134000|6271922|2069.076|

新增两行40模型epoch/8880更新，训练段+source+formal共14921048NN，另共享smoke1538NN；父两行60模型epoch/13320更新及其原读出18298572NN单列。teacher_train_NN=0；独立分析0NN/0CP/0IQ。wall是各模型实际段时间，不能从并行差异推断加速。

正式独立重算504个CM、84次RX/day/TX分区合并，与固定argmax和独立scorer逐项一致。全部4logical rows×7view预测固定及独立评分完成后才连接truth；测试IQ/SNR/信道/困难样本均未改变。

单模型seed无种子标准差，不证明重复性；小于0.1pp的正变化只记微小观察。阶段clean约束通过也不等于最终可重复双提升。测试不回流调参、候选或选择性重跑；本报告不据正式分数提出后续参数建议。

## 执行证据与缺项

训练release提交 `0f399c910324e21a8b1a60346d879be5c2964a9c`；独立分析提交 `ab16c3848794124f8c7fdee6013e01b2a46aae73`。全部原训练／分析PID正常退出，failure均为空。分析CPU2／CUDA隐藏，0NN／0IQ／0CP／0更新；全日志8880新增step、40个新增epoch完整核对，物理序列8880记录与4440条普通增强MSE对齐，mismatch均0。504正式CM、84分区合并与独立scorer一致；源600个分层CM单列保存，不作部署结果替代。

新增训练、源／正式读出和共享smoke共14,922,586 NN；父两行历史18,298,572 NN单列，累计33,221,158 NN包含本次共享smoke，未把历史共享smoke任意分摊。新段NoAs／As原生wall分别1602.126／2069.076 s，正式预测47.669／46.841 s。完整新增段不是连续cosine50，而是20+10+20／resetAdamW+cos20。

一次最终回收已VERIFIED：11项公开汇总＋manifest，archive SHA256 `c9ec3ca193837febd2d635d219bb5069c923beed056911754269f94d176f12fc`。完整CSV／CM／source与成本原件位于本地 `local_artifacts/F_As_budget_E50_20261018/collection-final-r01/aggregates/`，不提交原始日志／IDs／NPZ／IQ／checkpoint。没有技术缺项；科学缺项为E50其他模型种子的重复性。
